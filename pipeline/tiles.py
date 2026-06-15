import os
from functools import partial
from typing import Dict, List, Optional, Tuple, TypedDict

import leafmap.foliumap as leafmap
import numpy as np
import rasterio
import streamlit as st
from PIL import Image, ImageDraw

from config.models import ModelConfig
from ml.images import create_overlay_image, generate_mask
from utils.files import create_session_dir


PolygonCoords = List[List[float]]
TileCoords = Tuple[int, int]


class TileInfo(TypedDict):
    path: str
    coords: TileCoords
    size: int


class InferenceResult(TypedDict):
    mask: np.ndarray
    tile_path: str
    coords: TileCoords


class TiffData(TypedDict):
    data: np.ndarray
    transform: object
    crs: object
    shape: Tuple[int, int]


class AnalysisResult(TypedDict):
    tiles_info: List[TileInfo]
    inference_results: Dict[TileCoords, InferenceResult]
    stitched_mask: Optional[Image.Image]
    clean_mask: Optional[Image.Image]
    overlay_image: Optional[Image.Image]
    original_image: Optional[Image.Image]
    tiff_path: str


class SpatialResult(TypedDict):
    session_id: str
    session_dir: str
    primary_result: AnalysisResult
    bbox: List[float]
    polygon_coords: PolygonCoords


def process_polygon(
    polygon_coords: PolygonCoords,
    zoom_level: int,
    tile_size: int,
    model_config: ModelConfig,
) -> Optional[SpatialResult]:
    """Main processing function for spatial analysis."""
    st.info("Processing spatial analysis...")
    try:
        session_id, session_dir = create_session_dir()
        lons = [coord[1] for coord in polygon_coords]
        lats = [coord[0] for coord in polygon_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]

        primary_result = process_single_date(
            polygon_coords, zoom_level, tile_size, model_config, session_dir,
        )

        if not primary_result:
            st.error("Failed to process spatial analysis.")
            return None

        spatial_result: SpatialResult = {
            "session_id": session_id,
            "session_dir": session_dir,
            "primary_result": primary_result,
            "bbox": bbox,
            "polygon_coords": polygon_coords,
        }

        if "processing_results" not in st.session_state:
            st.session_state.processing_results = {}

        st.session_state.processing_results[session_id] = spatial_result
        st.success("Spatial analysis complete!")
        st.rerun()
        return spatial_result

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def process_single_date(
    polygon_coords: PolygonCoords,
    zoom_level: int,
    tile_size: int,
    model_config: ModelConfig,
    session_dir: str,
) -> Optional[AnalysisResult]:
    """Process analysis for a given polygon."""
    try:
        date_dir = os.path.join(session_dir, "analysis")
        os.makedirs(date_dir, exist_ok=True)

        lons = [coord[1] for coord in polygon_coords]
        lats = [coord[0] for coord in polygon_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]

        tmp_tiff = os.path.join(date_dir, "tiles.tif")
        if not download_tiles(tmp_tiff, bbox, zoom_level):
            return None
        
        tiff_data = read_tiff_data(tmp_tiff)
        if not tiff_data:
            return None
        
        tiles_info = extract_and_process_tiles(
            tiff_data,
            date_dir,
            tile_size,
            "tile",
        )

        if not tiles_info:
            return None

        inference_results = run_inference_on_tiles(tiles_info, model_config)

        clean_mask, display_mask = stitch_masks(tiles_info, inference_results, tile_size)

        overlay_image = None
        if clean_mask:
            original_image = get_original_image(tmp_tiff)
            if original_image:
                overlay_image = create_overlay_image(original_image, clean_mask)

        return {
            "tiles_info": tiles_info,
            "inference_results": inference_results,
            "stitched_mask": display_mask,
            "clean_mask": clean_mask,
            "overlay_image": overlay_image,
            "original_image": get_original_image(tmp_tiff),
            "tiff_path": tmp_tiff,
        }

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None


def add_division_lines(
    stitched_mask_img: Image.Image,
    tiles_info: List[TileInfo],
    tile_size: int,
) -> Image.Image:
    """Add dotted lines to show tile divisions."""
    if stitched_mask_img.mode != "RGBA":
        img = stitched_mask_img.convert("RGBA")
    else:
        img = stitched_mask_img.copy()

    draw = ImageDraw.Draw(img)

    coords_list = [tile["coords"] for tile in tiles_info]
    min_i = min(coord[0] for coord in coords_list)
    min_j = min(coord[1] for coord in coords_list)

    unique_i = sorted({tile["coords"][0] for tile in tiles_info})
    unique_j = sorted({tile["coords"][1] for tile in tiles_info})

    white_color = (255, 255, 255, 220)

    for i in unique_i:
        if i > min_i:
            y = i - min_i
            for thickness_offset in range(-1, 2):
                y_thick = y + thickness_offset
                if 0 <= y_thick < img.height:
                    for x in range(0, img.width, 6):
                        for dot_width in range(3):
                            if x + dot_width < img.width:
                                draw.point((x + dot_width, y_thick), fill=white_color)
    
    for j in unique_j:
        if j > min_j:
            x = j - min_j
            for thickness_offset in range(-1, 2):
                x_thick = x + thickness_offset
                if 0 <= x_thick < img.width:
                    for y in range(0, img.height, 6):
                        for dot_height in range(3):
                            if y + dot_height < img.height:
                                draw.point((x_thick, y + dot_height), fill=white_color)

    return img


def download_tiles(
    tmp_tiff: str,
    bbox: List[float],
    zoom: int,
) -> bool:
    """Download live satellite tiles and convert to 512 multiples."""
    try:
        tmp_raw_tiff = tmp_tiff.replace(".tif", "_raw.tif")

        leafmap.map_tiles_to_geotiff(
            output=tmp_raw_tiff,
            bbox=bbox,
            zoom=zoom,
            source="SATELLITE",
            overwrite=True,
        )

        if os.path.exists(tmp_raw_tiff) and os.path.getsize(tmp_raw_tiff) > 1000:
            with rasterio.open(tmp_raw_tiff) as test_src:
                if test_src.read(1).size > 0:
                    if process_and_save_512_multiple(tmp_raw_tiff, tmp_tiff):
                        try:
                            os.remove(tmp_raw_tiff)
                        except Exception:
                            pass
                        return True
    except Exception as e:
        st.error(f"Failed to download tiles: {str(e)}")
    
    return False


def process_and_save_512_multiple(input_tiff: str, output_tiff: str) -> bool:
    """Process the downloaded image to 512 multiples and save it."""
    try:
        with rasterio.open(input_tiff) as src:
            image_data = src.read()
            
            if image_data.size == 0:
                st.error("Downloaded image is empty. Please select a larger area.")
                return False
            if image_data.shape[1] < 512 or image_data.shape[2] < 512:
                st.error("Downloaded image is too small. Please select a larger area.")
                return False
            
            resized_data = resize_to_multiple_of_512(image_data)

            if len(resized_data.shape) == 3:
                new_height, new_width = resized_data.shape[1], resized_data.shape[2]
                channels = resized_data.shape[0]
            else:
                new_height, new_width = resized_data.shape
                channels = 1
                resized_data = resized_data[np.newaxis, :, :]
            
            original_transform = src.transform
            height_scale = new_height / src.height
            width_scale = new_width / src.width
            new_transform = rasterio.Affine(
                original_transform.a / width_scale,
                original_transform.b,
                original_transform.c,
                original_transform.d,
                original_transform.e / height_scale,
                original_transform.f,
            )
            
            with rasterio.open(
                output_tiff,
                "w",
                driver="GTiff",
                height=new_height,
                width=new_width,
                count=channels,
                dtype=resized_data.dtype,
                crs=src.crs,
                transform=new_transform,
            ) as dst:
                dst.write(resized_data)

            return True

    except Exception as e:
        st.error(f"Error processing image to 512 multiples: {str(e)}")
        return False


def extract_and_process_tiles(
    tiff_data: TiffData,
    date_dir: str,
    tile_size: int,
    prefix: str,
) -> Optional[List[TileInfo]]:
    """Extract and process tiles from tiff data."""
    shape = tiff_data["shape"]
    if shape[0] < tile_size or shape[1] < tile_size:
        st.error(f"Image too small for tile size {tile_size} after resizing to multiples of 512")
        return None
    
    if shape[0] % 512 != 0 or shape[1] % 512 != 0:
        st.warning(f"Image dimensions ({shape[1]}x{shape[0]}) are not multiples of 512")
    
    image_tensor = np.array(tiff_data["data"])
    tiles_info = process_tiles(image_tensor, date_dir, prefix, tile_size, shape)
    
    return tiles_info


def get_original_image(tiff_path: str) -> Optional[Image.Image]:
    """Extract original image from TIFF file."""
    try:
        with rasterio.open(tiff_path) as src:
            image_data = src.read()
            if len(image_data.shape) == 3:
                image_data = np.transpose(image_data, (1, 2, 0))
            if image_data.shape[-1] == 1:
                image_data = image_data.squeeze(axis=2)
            
            return Image.fromarray(image_data.astype(np.uint8))
    except Exception as e:
        st.error(f"Error reading original image: {str(e)}")
        return None


def resize_to_multiple_of_512(image_array: np.ndarray) -> np.ndarray:
    """Resize image to dimensions that are multiples of 512 by trimming."""
    if len(image_array.shape) == 3:
        if image_array.shape[0] < image_array.shape[2]:
            channels, height, width = image_array.shape
            new_height = (height // 512) * 512
            new_width = (width // 512) * 512
            resized_array = image_array[:, :new_height, :new_width]
        else:
            height, width, channels = image_array.shape
            new_height = (height // 512) * 512
            new_width = (width // 512) * 512
            resized_array = image_array[:new_height, :new_width, :]
    else:
        height, width = image_array.shape
        new_height = (height // 512) * 512
        new_width = (width // 512) * 512
        resized_array = image_array[:new_height, :new_width]
    
    return resized_array


def read_tiff_data(tmp_tiff: str) -> Optional[TiffData]:
    """Read and process TIFF data."""
    try:
        with rasterio.open(tmp_tiff) as src:
            if src.count == 0:
                return None
            
            image_data = src.read()
            if image_data is None or image_data.size == 0:
                return None
            
            if len(image_data.shape) == 3:
                new_shape = (image_data.shape[1], image_data.shape[2])
            else:
                new_shape = image_data.shape

            return {
                "data": image_data,
                "transform": src.transform,
                "crs": src.crs,
                "shape": new_shape,
            }
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


def run_inference_on_tiles(
    tiles_info: List[TileInfo],
    model_config: ModelConfig,
) -> Dict[TileCoords, InferenceResult]:
    results: Dict[TileCoords, InferenceResult] = {}
    progress_bar = st.progress(0)

    for idx, tile_info in enumerate(tiles_info):
        mask = generate_mask(tile_info["path"], model_config)
        results[tile_info["coords"]] = {
            "mask": mask,
            "tile_path": tile_info["path"],
            "coords": tile_info["coords"],
        }
        progress_bar.progress((idx + 1) / len(tiles_info))

    return results


def stitch_masks(
    tiles_info: List[TileInfo],
    inference_results: Dict[TileCoords, InferenceResult],
    tile_size: int,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Stitch individual tile masks into a complete mask."""
    if not tiles_info or not inference_results:
        return None, None
    
    height, width, min_i, min_j = calculate_stitched_dimensions(tiles_info, tile_size)
    stitched_mask = np.zeros((height, width), dtype=np.uint8)
    
    for tile in tiles_info:
        i, j = tile["coords"]
        if tile["coords"] in inference_results:
            mask = inference_results[tile["coords"]]["mask"]

            rel_i = i - min_i
            rel_j = j - min_j

            stitched_mask[rel_i:rel_i + tile_size, rel_j:rel_j + tile_size] = np.maximum(
                stitched_mask[rel_i:rel_i + tile_size, rel_j:rel_j + tile_size],
                mask * 255,
            )
    
    clean_mask = Image.fromarray(stitched_mask)
    display_mask = add_division_lines(clean_mask, tiles_info, tile_size)

    return clean_mask, display_mask


def calculate_stitched_dimensions(
    tiles_info: List[TileInfo],
    tile_size: int,
) -> Tuple[int, int, int, int]:
    """Calculate dimensions for stitched mask."""
    coords_list = [tile["coords"] for tile in tiles_info]
    min_i = min(coord[0] for coord in coords_list)
    max_i = max(coord[0] for coord in coords_list)
    min_j = min(coord[1] for coord in coords_list)
    max_j = max(coord[1] for coord in coords_list)

    height = max_i - min_i + tile_size
    width = max_j - min_j + tile_size

    return height, width, min_i, min_j


def process_tile_patch(
    index: TileCoords,
    image_tensor: np.ndarray,
    output_dir: str,
    name: str,
    size: int,
) -> Optional[TileInfo]:
    """Process individual tile patch."""
    i, j = index
    image_patch = image_tensor[:, i:i + size, j:j + size]

    if image_patch.shape[1] != size or image_patch.shape[2] != size:
        return None

    image_patch = np.transpose(image_patch, (1, 2, 0))
    if image_patch.shape[-1] == 1:
        image_patch = image_patch.squeeze(axis=2)

    image_patch = Image.fromarray(image_patch.astype(np.uint8))
    tile_path = f"{output_dir}/{name}_{i}_{j}.png"
    image_patch.save(tile_path, format="PNG")

    return {
        "path": tile_path,
        "coords": (i, j),
        "size": size,
    }


def create_tile_indexes(shape: Tuple[int, int], tile_size: int) -> List[TileCoords]:
    """Create tile indexes for processing."""
    indexes: List[TileCoords] = []
    for i in range(0, shape[0], tile_size):
        for j in range(0, shape[1], tile_size):
            indexes.append((i, j))
    return indexes


def process_tiles(
    image_tensor: np.ndarray,
    output_dir: str,
    session_id: str,
    tile_size: int,
    shape: Tuple[int, int],
) -> List[TileInfo]:
    """Process tiles from image tensor."""
    indexes = create_tile_indexes(shape, tile_size)
    
    process_fn = partial(
        process_tile_patch,
        image_tensor=image_tensor,
        output_dir=output_dir,
        name=session_id,
        size=tile_size,
    )

    progress_bar = st.progress(0)
    tiles_info: List[TileInfo] = []

    for idx, tile_info in enumerate(map(process_fn, indexes)):
        if tile_info:
            tiles_info.append(tile_info)
        progress_bar.progress((idx + 1) / len(indexes))

    return tiles_info
