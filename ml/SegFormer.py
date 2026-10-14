from __future__ import annotations

import time
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from transformers import (
    SegformerConfig,
    SegformerImageProcessor,
    TFSegformerForSemanticSegmentation,
)

from config.settings import TILE_SIZE
from ml.segmentation_core import InferenceTimings

ImageArray = NDArray[np.uint8]
SEGFORMER_BASE_CONFIG_PATH = Path("models/mit-b5")


class SegFormer:
    """SegFormer model and image processor loaded from one local checkpoint."""

    def __init__(
        self,
        weights_path: str | PathLike[str],
        base_config_path: str | PathLike[str] = SEGFORMER_BASE_CONFIG_PATH,
    ) -> None:
        id2label = {0: "non_slum", 1: "slum"}
        label2id = {label: label_id for label_id, label in id2label.items()}
        base_config = str(base_config_path)
        self.image_processor = SegformerImageProcessor.from_pretrained(
            base_config, local_files_only=True
        )
        config = SegformerConfig.from_pretrained(
            base_config,
            local_files_only=True,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        self.model = TFSegformerForSemanticSegmentation(config)
        _ = self.model(
            pixel_values=tf.zeros((1, 3, TILE_SIZE, TILE_SIZE)),
            training=False,
        )
        self.model.load_weights(weights_path)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 3, TILE_SIZE, TILE_SIZE], tf.float32),
            tf.TensorSpec([2], tf.int32),
        ]
    )
    def infer_masks(self, pixel_values: tf.Tensor, output_size: tf.Tensor) -> tf.Tensor:
        """Reuse one graph for forward inference and tile-sized uint8 masks."""
        outputs = self.model(pixel_values=pixel_values, training=False)
        logits = tf.transpose(outputs.logits, perm=(0, 2, 3, 1))
        logits = tf.image.resize(logits, output_size, method="bilinear")
        masks = tf.argmax(logits, axis=-1)
        return tf.cast(masks * 255, tf.uint8)


def load_segformer(weights_path: str | PathLike[str]) -> SegFormer:
    """Build MiT-B5 and load the fine-tuned SegFormer weights."""
    return SegFormer(weights_path)


def run_segformer_inference(
    model: SegFormer,
    tiles: list[Any],
    batch_size: int,
    timings: InferenceTimings | None = None,
) -> list[ImageArray]:
    """Preprocess tiles and run graph inference to produce binary uint8 masks.

    The image processor retains its existing resize, rescale, and normalization.
    The graph accepts float32 NCHW batches with a variable batch dimension, so
    partial batches reuse its trace. Output size is a tensor to avoid retracing
    for different tile dimensions. Timings separate image processing, completed
    graph execution and output download, and mask collection.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    all_masks: list[ImageArray] = []
    for index in range(0, len(tiles), batch_size):
        started = time.perf_counter()
        tile_batch = [np.asarray(tile) for tile in tiles[index : index + batch_size]]
        inputs = model.image_processor(
            images=tile_batch,
            return_tensors="tf",
        )
        output_size = tf.convert_to_tensor(tile_batch[0].shape[:2], dtype=tf.int32)
        if timings is not None:
            timings.preprocessing_seconds += time.perf_counter() - started

        started = time.perf_counter()
        masks = model.infer_masks(inputs["pixel_values"], output_size).numpy()
        if timings is not None:
            timings.model_forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        all_masks.extend(masks)
        if timings is not None:
            timings.postprocessing_seconds += time.perf_counter() - started

    return all_masks
