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
        self.image_processor = SegformerImageProcessor.from_pretrained(base_config)
        config = SegformerConfig.from_pretrained(
            base_config,
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


def load_segformer(weights_path: str | PathLike[str]) -> SegFormer:
    """Build MiT-B5 and load the fine-tuned SegFormer weights."""
    return SegFormer(weights_path)


def run_segformer_inference(
    model: SegFormer,
    tiles: list[Any],
    batch_size: int,
    timings: InferenceTimings | None = None,
) -> list[ImageArray]:
    """Run batched SegFormer inference and return tile-sized binary masks."""
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
        if timings is not None:
            timings.preprocessing_seconds += time.perf_counter() - started

        started = time.perf_counter()
        outputs = model.model(pixel_values=inputs["pixel_values"], training=False)
        if timings is not None:
            timings.model_forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        logits = tf.transpose(outputs.logits, perm=(0, 2, 3, 1))
        logits = tf.image.resize(logits, tile_batch[0].shape[:2], method="bilinear")
        masks = tf.argmax(logits, axis=-1)
        masks = (masks * 255).numpy().astype(np.uint8)
        all_masks.extend(masks)
        if timings is not None:
            timings.postprocessing_seconds += time.perf_counter() - started

    return all_masks
