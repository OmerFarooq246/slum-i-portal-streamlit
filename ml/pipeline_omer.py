import os
from typing import Any

import leafmap
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from config.models import ModelConfig
from config.settings import FIXED_ZOOM_LEVEL, TILE_SIZE
from ml.ENB5_Segmenter import EfficientNetB5_Segmenter
from ml.segmentation_core import (
    create_overlay_image,
    load_enb5_seg,
    merge_masks,
    resize_to_minimum,
    run_inference,
    slice_image,
)
from utils.files import create_session_dir

PolygonCoords = list[list[float]]
SpatialResult = dict[str, Any]


@st.cache_resource
def load_ENB5_Seg(weights_path: str) -> EfficientNetB5_Segmenter:
    """Build one cached ENB5-Seg model and load its trained weights."""
    return load_enb5_seg(weights_path)


def perform_segmentation(
    polygon_coords: PolygonCoords,
    model_config: ModelConfig,
    model: EfficientNetB5_Segmenter,
) -> None:
    try:
        session_id, session_dir = create_session_dir()

        raster_np, date_dir, bbox = _download_raster(polygon_coords, session_dir)
        img_path = os.path.join(date_dir, "raster.png")
        mask_path = os.path.join(date_dir, "mask.png")

        img = resize_to_minimum(raster_np, TILE_SIZE)
        tf.keras.utils.save_img(img_path, img.numpy())

        if img.shape[:2] != (TILE_SIZE, TILE_SIZE):
            tiles, tile_names = slice_image(img, TILE_SIZE)
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
