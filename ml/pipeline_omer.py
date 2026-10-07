import math
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
    run_inference,
    slice_image,
)
from utils.files import create_session_dir

PolygonCoords = list[list[float]]
SpatialResult = dict[str, Any]
WEB_MERCATOR_MAX_LATITUDE = 85.05112878
WEB_MAP_TILE_SIZE = 256


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

        raster_np, bbox, requested_size = _download_raster(polygon_coords, session_dir)
        original_height, original_width = raster_np.shape[:2]

        tiles, tile_names = slice_image(raster_np, TILE_SIZE)
        masks = run_inference(model, tiles, model_config.batch_size)
        mask_merged = merge_masks(masks, tile_names)
        requested_width, requested_height = requested_size
        bbox_expanded = requested_width < TILE_SIZE or requested_height < TILE_SIZE
        size_details = f"Raster: {original_width} x {original_height} px"
        if bbox_expanded:
            size_details = (
                f"Original requested raster: {requested_width} x {requested_height} px; "
                f"expanded raster: {original_width} x {original_height} px"
            )
        print(f"[segmentation] {size_details}; tiles processed: {len(tiles)}", flush=True)

        overlay = create_overlay_image(raster_np, mask_merged)

        result: SpatialResult = {
            "session_id": session_id,
            "session_dir": session_dir,
            "primary_result": {
                "original_image": Image.fromarray(raster_np),
                "stitched_mask": Image.fromarray(mask_merged.astype(np.uint8)),
                "overlay_image": Image.fromarray(overlay.astype(np.uint8)),
            },
            "bbox": bbox,
            "polygon_coords": polygon_coords,
            "bbox_expanded": bbox_expanded,
            "requested_size": requested_size,
        }

        st.session_state.processing_results[session_id] = result
        st.success("Spatial analysis complete.")
        st.rerun()

    except Exception as exc:
        st.error(f"Analysis failed: {exc!s}")
        raise


def _download_raster(
    polygon_coords: PolygonCoords, session_dir: str
) -> tuple[NDArray[np.uint8], list[float], tuple[int, int]]:
    date_dir = Path(session_dir) / "analysis"
    date_dir.mkdir(parents=True, exist_ok=True)

    longitudes = [coordinate[0] for coordinate in polygon_coords]
    latitudes = [coordinate[1] for coordinate in polygon_coords]
    requested_bbox = [
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
    ]
    requested_width, requested_height = _bbox_pixel_dimensions(
        requested_bbox,
        zoom=FIXED_ZOOM_LEVEL,
    )
    requested_size = (round(requested_width), round(requested_height))
    bbox = _ensure_minimum_bbox_size(
        requested_bbox,
        zoom=FIXED_ZOOM_LEVEL,
        minimum_size=TILE_SIZE,
    )

    tiff_path = date_dir / "raster.tif"
    Image.MAX_IMAGE_PIXELS = None
    while True:
        leafmap.map_tiles_to_geotiff(
            output=str(tiff_path),
            bbox=bbox,
            zoom=FIXED_ZOOM_LEVEL,
            source="SATELLITE",
            overwrite=True,
            quiet=False,
            options=["BIGTIFF=YES"],
        )

        with Image.open(tiff_path) as raster:
            raster_rgb = np.array(raster.convert("RGB"))

        raster_height, raster_width = raster_rgb.shape[:2]
        if raster_height >= TILE_SIZE and raster_width >= TILE_SIZE:
            return raster_rgb, bbox, requested_size

        bbox = _expand_bbox_for_raster_shortfall(
            bbox,
            zoom=FIXED_ZOOM_LEVEL,
            raster_width=raster_width,
            raster_height=raster_height,
            minimum_size=TILE_SIZE,
        )


def _ensure_minimum_bbox_size(
    bbox: list[float],
    zoom: int,
    minimum_size: int,
) -> list[float]:
    """Expand a Web Mercator bbox to a minimum pixel width and height."""
    return _ensure_minimum_bbox_dimensions(
        bbox,
        zoom=zoom,
        minimum_width=minimum_size,
        minimum_height=minimum_size,
    )


def _expand_bbox_for_raster_shortfall(
    bbox: list[float],
    zoom: int,
    raster_width: int,
    raster_height: int,
    minimum_size: int,
) -> list[float]:
    pixel_width, pixel_height = _bbox_pixel_dimensions(bbox, zoom)
    return _ensure_minimum_bbox_dimensions(
        bbox,
        zoom=zoom,
        minimum_width=pixel_width + max(0, minimum_size - raster_width),
        minimum_height=pixel_height + max(0, minimum_size - raster_height),
    )


def _bbox_pixel_dimensions(bbox: list[float], zoom: int) -> tuple[float, float]:
    west, south, east, north = bbox
    width = _longitude_to_pixel_x(east, zoom) - _longitude_to_pixel_x(west, zoom)
    height = _latitude_to_pixel_y(south, zoom) - _latitude_to_pixel_y(north, zoom)
    return width, height


def _ensure_minimum_bbox_dimensions(
    bbox: list[float],
    zoom: int,
    minimum_width: float,
    minimum_height: float,
) -> list[float]:
    west, south, east, north = bbox
    left = _longitude_to_pixel_x(west, zoom)
    right = _longitude_to_pixel_x(east, zoom)
    top = _latitude_to_pixel_y(north, zoom)
    bottom = _latitude_to_pixel_y(south, zoom)

    left, right = _expand_pixel_axis(left, right, minimum_width)
    top, bottom = _expand_pixel_axis(top, bottom, minimum_height)

    return [
        _pixel_x_to_longitude(left, zoom),
        _pixel_y_to_latitude(bottom, zoom),
        _pixel_x_to_longitude(right, zoom),
        _pixel_y_to_latitude(top, zoom),
    ]


def _expand_pixel_axis(start: float, end: float, minimum_size: float) -> tuple[float, float]:
    if end - start >= minimum_size:
        return start, end

    expanded_start = math.floor((start + end - minimum_size) / 2)
    return float(expanded_start), float(expanded_start + minimum_size)


def _longitude_to_pixel_x(longitude: float, zoom: int) -> float:
    world_size = WEB_MAP_TILE_SIZE * 2**zoom
    return (longitude + 180.0) / 360.0 * world_size


def _latitude_to_pixel_y(latitude: float, zoom: int) -> float:
    world_size = WEB_MAP_TILE_SIZE * 2**zoom
    bounded_latitude = max(
        -WEB_MERCATOR_MAX_LATITUDE,
        min(WEB_MERCATOR_MAX_LATITUDE, latitude),
    )
    latitude_radians = math.radians(bounded_latitude)
    return (1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2.0 * world_size


def _pixel_x_to_longitude(pixel_x: float, zoom: int) -> float:
    world_size = WEB_MAP_TILE_SIZE * 2**zoom
    return pixel_x / world_size * 360.0 - 180.0


def _pixel_y_to_latitude(pixel_y: float, zoom: int) -> float:
    world_size = WEB_MAP_TILE_SIZE * 2**zoom
    mercator_y = math.pi * (1.0 - 2.0 * pixel_y / world_size)
    return math.degrees(math.atan(math.sinh(mercator_y)))
