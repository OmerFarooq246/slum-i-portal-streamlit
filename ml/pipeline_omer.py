import os
from typing import Any

import leafmap
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from config.models import ModelConfig
from config.settings import FIXED_ZOOM_LEVEL, TILE_SIZE
from ml.ENB5_Segmenter import EfficientNet_Segmenter_Config, EfficientNetB5_Segmenter

PolygonCoords = list[list[float]]
SpatialResult = dict[str, Any]


@st.cache_resource
def load_ENB5_Seg(weights_path: str) -> EfficientNetB5_Segmenter:
    """Build one cached ENB5-Seg model and load its trained weights."""
    config = EfficientNet_Segmenter_Config()
    model = EfficientNetB5_Segmenter(config)
    dummy = tf.zeros((1, *config.input_shape))
    _ = model(dummy)
    model.load_weights(weights_path)
    return model


def perform_segmentation(
    polygon_coords: PolygonCoords,
    model_config: ModelConfig,
    model: EfficientNetB5_Segmenter,
) -> None:
    try:
        from utils.files import create_session_dir

        session_id, session_dir = create_session_dir()

        raster_np, date_dir, bbox = _download_raster(polygon_coords, session_dir)
        img_path = os.path.join(date_dir, "raster.png")
        mask_path = os.path.join(date_dir, "mask.png")

        img = _resize_and_save(raster_np, TILE_SIZE, img_path)

        if img.shape[:2] != (TILE_SIZE, TILE_SIZE):
            tiles, tile_names = _slice(img, TILE_SIZE)
        else:
            tiles = [img]
            tile_names = None

        masks = run_inference(model, tiles, model_config.batch_size)

        mask_merged = merge_masks(masks, tile_names) if tile_names else masks[0]

        tf.keras.utils.save_img(mask_path, tf.expand_dims(mask_merged, axis=-1))

        overlay = create_overlay_image(img, mask_merged)

        result: SpatialResult = {
            "session_id": session_id,
            "session_dir": session_dir,
            "primary_result": {
                "original_image": Image.fromarray(img.numpy().astype(np.uint8)),
                "stitched_mask": Image.fromarray(mask_merged.astype(np.uint8)),
                "overlay_image": Image.fromarray(overlay.astype(np.uint8)),
            },
            "bbox": bbox,
            "polygon_coords": polygon_coords,
        }

        st.session_state.processing_results[session_id] = result
        st.success("Spatial analysis complete.")
        st.rerun()

    except Exception as e:
        st.error(f"Analysis failed: {e!s}")
        raise


def _download_raster(polygon_coords, session_dir):

    date_dir = os.path.join(session_dir, "analysis")
    os.makedirs(date_dir, exist_ok=True)

    lons = [c[1] for c in polygon_coords]
    lats = [c[0] for c in polygon_coords]
    bbox = [min(lats), min(lons), max(lats), max(lons)]

    tiff_path = os.path.join(date_dir, "raster.tif")
    leafmap.map_tiles_to_geotiff(
        output=tiff_path,
        bbox=bbox,
        zoom=FIXED_ZOOM_LEVEL,
        source="SATELLITE",
        overwrite=True,
        quiet=False,
        options=["BIGTIFF=YES"],
    )

    Image.MAX_IMAGE_PIXELS = None
    raster = Image.open(tiff_path).convert("RGB")
    return np.array(raster), date_dir, bbox


def _resize_and_save(img_np, min_size, save_path):
    h, w, _ = img_np.shape
    if w < min_size or h < min_size:
        scale = max(min_size / w, min_size / h)
        img_t = tf.image.resize(
            img_np,
            [round(h * scale), round(w * scale)],
            method=tf.image.ResizeMethod.BILINEAR,
        )
    else:
        img_t = tf.constant(img_np)
    img_t = tf.cast(img_t, tf.uint8)
    tf.keras.utils.save_img(save_path, img_t.numpy())
    return img_t


def _slice(raster, tile_size):
    raster_h, raster_w = raster.shape[0], raster.shape[1]
    tiles, names = [], []
    for j in range(0, raster_h, tile_size):
        for i in range(0, raster_w, tile_size):
            i_t = min(i, raster_w - tile_size)
            j_t = min(j, raster_h - tile_size)
            tiles.append(raster[j_t : j_t + tile_size, i_t : i_t + tile_size, :])
            names.append(f"tile_{i_t}_{j_t}")
    return tiles, names


def run_inference(model, tiles, batch_size):
    """Run batched ENB5-Seg inference and return binary uint8 masks."""
    all_masks = []
    for i in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[i : i + batch_size]).astype(np.float32) / 255.0
        preds = model(batch, training=False)
        masks = tf.argmax(preds, axis=-1)
        masks = (masks * 255).numpy().astype(np.uint8)
        all_masks.extend(masks)
    return all_masks


def merge_masks(masks, tile_names):
    """Merge overlapping masks using tile names formatted as tile_X_Y."""
    coords = [tuple(map(int, n.split("_")[1:3])) for n in tile_names]
    full_w = max(x + m.shape[1] for m, (x, _) in zip(masks, coords, strict=True))
    full_h = max(y + m.shape[0] for m, (_, y) in zip(masks, coords, strict=True))
    merged = np.zeros((full_h, full_w), dtype=np.uint8)
    for mask, (x, y) in zip(masks, coords, strict=True):
        h, w = mask.shape
        merged[y : y + h, x : x + w] = np.maximum(merged[y : y + h, x : x + w], mask)
    return merged


def create_overlay_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Return an RGB array with the positive mask blended over the image."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = image.astype(np.float32)
    mask_bool = mask > 0
    colored = np.zeros_like(image)
    colored[mask_bool] = color
    out = image.copy()
    out[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * colored[mask_bool]
    return out.astype(np.uint8)
