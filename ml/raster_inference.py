"""Sequential raster inference with bounded batches and disk-backed mask merging."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from config.settings import TILE_SIZE
from ml.segmentation_core import ImageArray, InferenceTimings, run_inference

GDAL_CACHE_BYTES = 32 * 1024 * 1024
PREVIEW_MAX_SIZE = 1024


@dataclass
class RasterInferenceResult:
    width: int
    height: int
    tile_count: int = 0
    reading_seconds: float = 0.0
    inference_seconds: float = 0.0
    writing_seconds: float = 0.0
    inference_timings: InferenceTimings = field(default_factory=InferenceTimings)


def tile_windows(width: int, height: int, tile_size: int = TILE_SIZE) -> Iterator[Window]:
    """Yield integer windows, shifting boundary tiles back without padding."""
    if tile_size < 1 or min(width, height) < tile_size:
        raise ValueError("raster dimensions must be at least tile_size, and tile_size positive")
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            yield Window(
                min(left, width - tile_size), min(top, height - tile_size), tile_size, tile_size
            )


def _rgb(data: ImageArray) -> ImageArray:
    """Match RGB conversion for grayscale, grayscale-alpha, and RGB(A) TIFFs."""
    if data.shape[0] < 3:
        data = np.repeat(data[:1], 3, axis=0)
    return np.moveaxis(data[:3], 0, -1)


def infer_raster(
    raster_path: Path,
    mask_path: Path,
    model: Any,
    batch_size: int,
    inference_runner: Callable[..., list[ImageArray]] = run_inference,
    *,
    region: Window | None = None,
    tile_size: int = TILE_SIZE,
) -> RasterInferenceResult:
    """Read one batch, infer, maximum-merge into a TIFF, then release the batch.

    The source must be uint8 imagery. ``region`` is an optional integer pixel ROI
    for benchmarks. The output retains its CRS and region-adjusted transform.
    A temporary file is promoted only after all batches complete successfully.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    raster_path, mask_path = Path(raster_path), Path(mask_path)
    if raster_path.resolve() == mask_path.resolve():
        raise ValueError("input raster and output mask must have different paths")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mask_path.with_suffix(".partial.tif")
    try:
        with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES), rasterio.open(raster_path) as src:
            roi = region or Window(0, 0, src.width, src.height)
            if any(int(v) != v for v in (roi.col_off, roi.row_off, roi.width, roi.height)):
                raise ValueError("region coordinates and dimensions must be integers")
            if (
                roi.col_off < 0
                or roi.row_off < 0
                or roi.col_off + roi.width > src.width
                or roi.row_off + roi.height > src.height
            ):
                raise ValueError("region must be inside the source raster")
            width, height = int(roi.width), int(roi.height)
            if min(width, height) < tile_size:
                raise ValueError("raster dimensions must be at least tile_size")
            if any(dtype != "uint8" for dtype in src.dtypes):
                raise ValueError("inference requires a uint8 raster")
            result = RasterInferenceResult(width, height)
            windows = tile_windows(width, height, tile_size)
            bands = list(range(1, min(src.count, 3) + 1))
            with rasterio.open(
                temporary,
                "w+",
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                dtype="uint8",
                crs=src.crs,
                transform=src.window_transform(roi),
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="deflate",
                BIGTIFF="IF_SAFER",
            ) as dst:
                while coordinates := list(islice(windows, batch_size)):
                    started = time.perf_counter()
                    tiles = [
                        _rgb(
                            src.read(
                                bands,
                                window=Window(
                                    w.col_off + roi.col_off,
                                    w.row_off + roi.row_off,
                                    w.width,
                                    w.height,
                                ),
                            )
                        )
                        for w in coordinates
                    ]
                    result.reading_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    masks = inference_runner(
                        model, tiles, batch_size, timings=result.inference_timings
                    )
                    result.inference_seconds += time.perf_counter() - started
                    del tiles
                    started = time.perf_counter()
                    for mask, window in zip(masks, coordinates, strict=True):
                        if mask.shape != (tile_size, tile_size) or mask.dtype != np.uint8:
                            raise ValueError("inference must return tile-sized uint8 masks")
                        existing = dst.read(1, window=window)
                        np.maximum(existing, mask, out=existing)
                        dst.write(existing, 1, window=window)
                    del masks, mask, existing
                    result.writing_seconds += time.perf_counter() - started
                    result.tile_count += len(coordinates)
        temporary.replace(mask_path)
        return result
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def raster_previews(
    raster_path: Path,
    mask_path: Path,
    *,
    region: Window | None = None,
    max_size: int = PREVIEW_MAX_SIZE,
) -> tuple[ImageArray, ImageArray]:
    """Read bounded RGB/mask previews; nearest resampling preserves mask labels."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES), rasterio.open(raster_path) as src:
        roi = region or Window(0, 0, src.width, src.height)
        scale = min(1.0, max_size / max(roi.width, roi.height))
        height, width = max(1, round(roi.height * scale)), max(1, round(roi.width * scale))
        bands = list(range(1, min(src.count, 3) + 1))
        image = _rgb(
            src.read(
                bands,
                window=roi,
                out_shape=(len(bands), height, width),
                resampling=Resampling.bilinear,
            )
        )
        with rasterio.open(mask_path) as mask:
            labels = mask.read(1, out_shape=(height, width), resampling=Resampling.nearest)
    return image, labels
