from pathlib import Path
from typing import Any

import leafmap
import numpy as np
import streamlit as st
from numpy.typing import NDArray
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

        raster_np, bbox = _download_raster(polygon_coords, session_dir)

        img = resize_to_minimum(raster_np, TILE_SIZE)

        tiles, tile_names = slice_image(img, TILE_SIZE)
        masks = run_inference(model, tiles, model_config.batch_size)
        mask_merged = merge_masks(masks, tile_names)

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

    except Exception as exc:
        st.error(f"Analysis failed: {exc!s}")
        raise


def _download_raster(
    polygon_coords: PolygonCoords, session_dir: str
) -> tuple[NDArray[np.uint8], list[float]]:
    date_dir = Path(session_dir) / "analysis"
    date_dir.mkdir(parents=True, exist_ok=True)

    lons = [c[1] for c in polygon_coords]
    lats = [c[0] for c in polygon_coords]
    bbox = [min(lats), min(lons), max(lats), max(lons)]

    tiff_path = date_dir / "raster.tif"
    leafmap.map_tiles_to_geotiff(
        output=str(tiff_path),
        bbox=bbox,
        zoom=FIXED_ZOOM_LEVEL,
        source="SATELLITE",
        overwrite=True,
        quiet=False,
        options=["BIGTIFF=YES"],
    )

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tiff_path) as raster:
        raster_rgb = np.array(raster.convert("RGB"))
    return raster_rgb, bbox
