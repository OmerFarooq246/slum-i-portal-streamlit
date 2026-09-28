"""Shared, dependency-light segmentation pipeline operations."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ml.ENB5_Segmenter import EfficientNet_Segmenter_Config, EfficientNetB5_Segmenter


@dataclass
class InferenceTimings:
    """Accumulated timings for the operations inside ``run_inference``."""

    preprocessing_seconds: float = 0.0
    model_forward_seconds: float = 0.0
    postprocessing_seconds: float = 0.0


def load_enb5_seg(weights_path: str) -> EfficientNetB5_Segmenter:
    """Build ENB5-Seg without external pretrained weights and load a checkpoint."""
    config = EfficientNet_Segmenter_Config()
    model = EfficientNetB5_Segmenter(config)
    dummy = tf.zeros((1, *config.input_shape))
    _ = model(dummy, training=False)
    model.load_weights(weights_path)
    return model


def resize_to_minimum(img_np, min_size):
    """Upscale an image when either dimension is smaller than ``min_size``."""
    height, width, _ = img_np.shape
    if width < min_size or height < min_size:
        scale = max(min_size / width, min_size / height)
        image = tf.image.resize(
            img_np,
            [round(height * scale), round(width * scale)],
            method=tf.image.ResizeMethod.BILINEAR,
        )
    else:
        image = tf.constant(img_np)
    return tf.cast(image, tf.uint8)


def slice_image(raster, tile_size):
    """Slice an image into the same edge-aligned, in-memory tiles used by the portal."""
    raster_height, raster_width = raster.shape[0], raster.shape[1]
    tiles, names = [], []
    for top in range(0, raster_height, tile_size):
        for left in range(0, raster_width, tile_size):
            tile_left = min(left, raster_width - tile_size)
            tile_top = min(top, raster_height - tile_size)
            tiles.append(
                raster[
                    tile_top : tile_top + tile_size,
                    tile_left : tile_left + tile_size,
                    :,
                ]
            )
            names.append(f"tile_{tile_left}_{tile_top}")
    return tiles, names


def run_inference(model, tiles, batch_size, timings: InferenceTimings | None = None):
    """Run normalized, batched inference and return binary uint8 masks.

    When ``timings`` is supplied, durations are accumulated separately for batch
    preparation, the model forward calls, and mask postprocessing. The benchmark
    enables synchronous TensorFlow execution before calling this function so that
    forward-call timings represent completed CPU work.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    all_masks = []
    for index in range(0, len(tiles), batch_size):
        started = time.perf_counter()
        batch = np.stack(tiles[index : index + batch_size]).astype(np.float32) / 255.0
        if timings is not None:
            timings.preprocessing_seconds += time.perf_counter() - started

        started = time.perf_counter()
        predictions = model(batch, training=False)
        if timings is not None:
            timings.model_forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        masks = tf.argmax(predictions, axis=-1)
        masks = (masks * 255).numpy().astype(np.uint8)
        all_masks.extend(masks)
        if timings is not None:
            timings.postprocessing_seconds += time.perf_counter() - started
    return all_masks


def merge_masks(masks, tile_names):
    """Merge overlapping masks using tile names formatted as ``tile_X_Y``."""
    coords = [tuple(map(int, name.split("_")[1:3])) for name in tile_names]
    full_width = max(x + mask.shape[1] for mask, (x, _) in zip(masks, coords, strict=True))
    full_height = max(y + mask.shape[0] for mask, (_, y) in zip(masks, coords, strict=True))
    merged = np.zeros((full_height, full_width), dtype=np.uint8)
    for mask, (x, y) in zip(masks, coords, strict=True):
        height, width = mask.shape
        merged[y : y + height, x : x + width] = np.maximum(
            merged[y : y + height, x : x + width], mask
        )
    return merged


def create_overlay_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Return an RGB array with the positive mask blended over the image."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = image.astype(np.float32)
    mask_bool = mask > 0
    colored = np.zeros_like(image)
    colored[mask_bool] = color
    output = image.copy()
    output[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * colored[mask_bool]
    return output.astype(np.uint8)
