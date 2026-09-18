#!/usr/bin/env python3
"""Controlled ENB5-Seg system benchmark using only a local raster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rasterio
import tensorflow as tf
from numpy.typing import NDArray
from PIL import Image
from rasterio.windows import Window

from config.settings import TILE_SIZE
from ml.pipeline_omer import create_overlay_image, load_ENB5_Seg, run_inference

ROI_SPECS: Sequence[tuple[str, int]] = (
    ("small", 2),
    ("medium", 4),
    ("large", 6),
)
TIMING_METRICS = (
    "raster_load_preprocess_s",
    "tiling_s",
    "inference_s",
    "stitching_s",
    "output_generation_s",
    "total_latency_s",
)
OTHER_METRICS = ("tiles_per_second", "peak_process_memory_mb")
ImageArray = NDArray[np.uint8]


class PeakMemorySampler:
    """Sample this process's resident memory during one pipeline run."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample_once(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample_once()
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        self._sample_once()
        return self.peak_bytes / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", required=True, type=Path, help="Local source GeoTIFF/raster")
    parser.add_argument("--weights", required=True, type=Path, help="ENB5-Seg .h5 weights")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--crop-x", type=int, default=0, help="Deterministic left pixel offset")
    parser.add_argument("--crop-y", type=int, default=0, help="Deterministic top pixel offset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--cpu-limit", type=float, default=2.0)
    parser.add_argument("--ram-limit-gb", type=float, default=4.0)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.raster.is_file():
        raise FileNotFoundError(f"Source raster not found: {args.raster}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"ENB5-Seg weights not found: {args.weights}")
    if args.batch_size < 1 or args.warmup_runs != 1 or args.measured_runs != 5:
        raise ValueError(
            "Controlled protocol requires batch size >= 1, exactly 1 warm-up, and exactly 5 measured runs"
        )
    required_extent = max(grid for _, grid in ROI_SPECS) * TILE_SIZE
    with rasterio.open(args.raster) as src:
        if src.count < 1:
            raise ValueError("Source raster has no bands")
        if args.crop_x < 0 or args.crop_y < 0:
            raise ValueError("Crop offsets must be non-negative")
        if src.width < args.crop_x + required_extent or src.height < args.crop_y + required_extent:
            raise ValueError(
                f"Raster must cover a {required_extent}x{required_extent} pixel crop at "
                f"offset ({args.crop_x}, {args.crop_y}); got {src.width}x{src.height}"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detected_cgroup_limits() -> dict[str, Any]:
    detected: dict[str, Any] = {}
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    memory_max = Path("/sys/fs/cgroup/memory.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text().strip().split()
        detected["detected_cpu_limit"] = (
            "unlimited" if quota == "max" else float(quota) / float(period)
        )
    if memory_max.is_file():
        value = memory_max.read_text().strip()
        detected["detected_ram_limit_bytes"] = "unlimited" if value == "max" else int(value)
    detected["detected_cpu_affinity_count"] = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    )
    return detected


def load_raster_roi(raster_path: Path, crop_x: int, crop_y: int, pixels: int) -> ImageArray:
    with rasterio.open(raster_path) as src:
        bands = list(range(1, min(src.count, 3) + 1))
        data = src.read(bands, window=Window(crop_x, crop_y, pixels, pixels))
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate((data, data[-1:, :, :]), axis=0)
    return np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)


def tile_roi(image: ImageArray, tile_dir: Path, grid_size: int) -> list[tuple[Path, int, int]]:
    tile_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[Path, int, int]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            top = row * TILE_SIZE
            left = col * TILE_SIZE
            tile = image[top : top + TILE_SIZE, left : left + TILE_SIZE]
            tile_path = tile_dir / f"tile_{row}_{col}.png"
            Image.fromarray(tile).save(tile_path, format="PNG")
            tiles.append((tile_path, row, col))
    return tiles


def infer_tiles(
    model: tf.keras.Model, tiles: Sequence[tuple[Path, int, int]], batch_size: int
) -> list[ImageArray]:
    tile_arrays = []
    for tile_path, _, _ in tiles:
        encoded = tf.io.read_file(str(tile_path))
        tile = tf.image.decode_image(encoded, channels=3, expand_animations=False)
        tile_arrays.append(tile.numpy())
    return run_inference(model, tile_arrays, batch_size)


def stitch_masks(masks: Sequence[ImageArray], grid_size: int) -> Image.Image:
    stitched = np.zeros((grid_size * TILE_SIZE, grid_size * TILE_SIZE), dtype=np.uint8)
    for index, mask in enumerate(masks):
        row, col = divmod(index, grid_size)
        top = row * TILE_SIZE
        left = col * TILE_SIZE
        stitched[top : top + TILE_SIZE, left : left + TILE_SIZE] = np.maximum(
            stitched[top : top + TILE_SIZE, left : left + TILE_SIZE],
            mask,
        )
    return Image.fromarray(stitched)


def generate_outputs(image: ImageArray, mask: Image.Image, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(create_overlay_image(image, np.asarray(mask)))
    mask.save(output_dir / "mask.png", format="PNG")
    overlay.save(output_dir / "overlay.png", format="PNG")


def run_pipeline(
    raster_path: Path,
    model: tf.keras.Model,
    roi_name: str,
    grid_size: int,
    crop_x: int,
    crop_y: int,
    batch_size: int,
    work_dir: Path,
) -> dict[str, float]:
    sampler = PeakMemorySampler()
    sampler.start()
    total_start = time.perf_counter()

    started = time.perf_counter()
    pixels = grid_size * TILE_SIZE
    image = load_raster_roi(raster_path, crop_x, crop_y, pixels)
    raster_latency = time.perf_counter() - started

    started = time.perf_counter()
    tiles = tile_roi(image, work_dir / roi_name / "tiles", grid_size)
    tiling_latency = time.perf_counter() - started

    started = time.perf_counter()
    masks = infer_tiles(model, tiles, batch_size)
    inference_latency = time.perf_counter() - started

    started = time.perf_counter()
    stitched = stitch_masks(masks, grid_size)
    stitching_latency = time.perf_counter() - started

    started = time.perf_counter()
    generate_outputs(image, stitched, work_dir / roi_name / "outputs")
    output_latency = time.perf_counter() - started

    total_latency = time.perf_counter() - total_start
    peak_memory = sampler.stop()
    tile_count = grid_size * grid_size
    return {
        "raster_load_preprocess_s": raster_latency,
        "tiling_s": tiling_latency,
        "inference_s": inference_latency,
        "stitching_s": stitching_latency,
        "output_generation_s": output_latency,
        "total_latency_s": total_latency,
        "tiles_per_second": tile_count / inference_latency,
        "peak_process_memory_mb": peak_memory,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for roi_name, grid_size in ROI_SPECS:
        roi_rows = [row for row in rows if row["roi"] == roi_name]
        item: dict[str, Any] = {
            "roi": roi_name,
            "width_px": grid_size * TILE_SIZE,
            "height_px": grid_size * TILE_SIZE,
            "tile_count": grid_size * grid_size,
            "measured_runs": len(roi_rows),
        }
        for metric in (*TIMING_METRICS, *OTHER_METRICS):
            values = [float(row[metric]) for row in roi_rows]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_stddev"] = statistics.stdev(values)
        summary.append(item)
    return summary


def write_paper_table(path: Path, summary: Sequence[dict[str, Any]]) -> None:
    headers = (
        "ROI",
        "Tiles",
        "Load/preprocess (s)",
        "Tiling (s)",
        "Inference (s)",
        "Stitch (s)",
        "Output (s)",
        "Total (s)",
        "Tiles/s",
        "Peak RSS (MB)",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    metric_order = (
        *TIMING_METRICS[:-1],
        "total_latency_s",
        "tiles_per_second",
        "peak_process_memory_mb",
    )
    for row in summary:
        values = [str(row["roi"]).title(), str(row["tile_count"])]
        values.extend(
            f"{row[f'{metric}_median']:.3f} ± {row[f'{metric}_stddev']:.3f}"
            for metric in metric_order
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.results_dir / "artifacts"

    tf.config.threading.set_intra_op_parallelism_threads(max(1, int(args.cpu_limit)))
    tf.config.threading.set_inter_op_parallelism_threads(1)

    metadata: dict[str, Any] = {
        "protocol": "ENB5-Seg local-raster system benchmark",
        "cpu_limit": args.cpu_limit,
        "ram_limit_gb": args.ram_limit_gb,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tile_size": TILE_SIZE,
        "batch_size": args.batch_size,
        "warmup_runs_per_roi": args.warmup_runs,
        "measured_runs_per_roi": args.measured_runs,
        "crop_x": args.crop_x,
        "crop_y": args.crop_y,
        "source_raster": str(args.raster.resolve()),
        "source_raster_sha256": file_sha256(args.raster),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": file_sha256(args.weights),
        "tiles_per_roi": {name: grid * grid for name, grid in ROI_SPECS},
        "model_initialization_timed": False,
        "imagery_acquisition_timed": False,
    }
    metadata.update(detected_cgroup_limits())
    (args.results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # This call initializes and loads weights once. Streamlit caches separately by
    # weights_path; the same resident object is reused for every warm-up/measured run.
    model = load_ENB5_Seg(str(args.weights))

    raw_rows: list[dict[str, Any]] = []
    for roi_name, grid_size in ROI_SPECS:
        print(f"Warm-up: {roi_name} ({grid_size * grid_size} tiles)", flush=True)
        run_pipeline(
            args.raster,
            model,
            roi_name,
            grid_size,
            args.crop_x,
            args.crop_y,
            args.batch_size,
            work_dir,
        )
        for run_number in range(1, args.measured_runs + 1):
            print(f"Measured run {run_number}/{args.measured_runs}: {roi_name}", flush=True)
            metrics = run_pipeline(
                args.raster,
                model,
                roi_name,
                grid_size,
                args.crop_x,
                args.crop_y,
                args.batch_size,
                work_dir,
            )
            raw_rows.append(
                {
                    "roi": roi_name,
                    "run": run_number,
                    "width_px": grid_size * TILE_SIZE,
                    "height_px": grid_size * TILE_SIZE,
                    "tile_count": grid_size * grid_size,
                    "tile_size": TILE_SIZE,
                    "batch_size": args.batch_size,
                    "cpu_limit": args.cpu_limit,
                    "ram_limit_gb": args.ram_limit_gb,
                    "architecture": platform.machine(),
                    "tensorflow_version": tf.__version__,
                    "python_version": platform.python_version(),
                    **metrics,
                }
            )

    raw_fields = list(raw_rows[0].keys())
    write_csv(args.results_dir / "raw_runs.csv", raw_rows, raw_fields)
    summary_rows = summarize(raw_rows)
    write_csv(args.results_dir / "summary.csv", summary_rows, summary_rows[0].keys())
    write_paper_table(args.results_dir / "paper_table.md", summary_rows)
    print(f"Results written to {args.results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Benchmark preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
