#!/usr/bin/env python3
"""Controlled segmentation system benchmark using only a local raster."""

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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
from ml.segmentation_core import (
    InferenceTimings,
    create_overlay_image,
    load_enb5_seg,
    merge_masks,
    resize_to_minimum,
    run_inference,
    slice_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_PATH = PROJECT_ROOT / "benchmarks/data/raster_benchmark.tif"
RESULTS_ROOT = PROJECT_ROOT / "benchmarks/results"
ModelLoader = Callable[[str], Any]
InferenceRunner = Callable[..., list[Any]]


@dataclass(frozen=True)
class ModelAdapter:
    """Model-specific loading and inference operations used by the benchmark."""

    default_weights: Path
    load: ModelLoader
    infer: InferenceRunner


MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    "enb5_seg_islamabad": ModelAdapter(
        default_weights=PROJECT_ROOT / "models/enb5_seg_islamabad.h5",
        load=load_enb5_seg,
        infer=run_inference,
    ),
}
ROI_SPECS: Sequence[tuple[str, int]] = (
    ("small", 2),
    ("medium", 4),
    ("large", 6),
)
TIMING_METRICS = (
    "raster_load_preprocess_ms",
    "tiling_ms",
    "inference_preprocessing_ms",
    "model_forward_ms",
    "inference_postprocessing_ms",
    "inference_ms",
    "stitching_ms",
    "output_generation_ms",
    "total_latency_ms",
)
OTHER_METRICS = (
    "tiles_per_second",
    "model_tiles_per_second",
    "peak_process_memory_mb",
)
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_ADAPTERS),
        default="enb5_seg_islamabad",
        help="Model adapter to benchmark; add future adapters to MODEL_ADAPTERS",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Output directory; defaults to benchmarks/results/<model>",
    )
    parser.add_argument("--crop-x", type=int, default=0, help="Deterministic left pixel offset")
    parser.add_argument("--crop-y", type=int, default=0, help="Deterministic top pixel offset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--cpu-limit", type=float, default=2.0)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve command-line paths relative to the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_weights_path(args: argparse.Namespace) -> Path:
    return MODEL_ADAPTERS[args.model].default_weights


def selected_results_dir(args: argparse.Namespace) -> Path:
    requested = args.results_dir or RESULTS_ROOT / args.model
    return project_path(requested)


def validate_inputs(args: argparse.Namespace, weights_path: Path) -> None:
    if not RASTER_PATH.is_file():
        raise FileNotFoundError(f"Source raster not found: {RASTER_PATH}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if args.batch_size < 1 or args.cpu_limit <= 0:
        raise ValueError("Batch size and CPU limit must both be greater than zero")
    if args.warmup_runs != 1 or args.measured_runs != 5:
        raise ValueError(
            "Controlled protocol requires exactly 1 warm-up and exactly 5 measured runs"
        )
    required_extent = max(grid for _, grid in ROI_SPECS) * TILE_SIZE
    with rasterio.open(RASTER_PATH) as src:
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


def generate_outputs(image, mask: ImageArray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(create_overlay_image(image, mask))
    Image.fromarray(mask).save(output_dir / "mask.png", format="PNG")
    overlay.save(output_dir / "overlay.png", format="PNG")


def run_pipeline(
    raster_path: Path,
    model: Any,
    roi_name: str,
    grid_size: int,
    crop_x: int,
    crop_y: int,
    batch_size: int,
    work_dir: Path,
    inference_runner: InferenceRunner = run_inference,
) -> dict[str, float]:
    sampler = PeakMemorySampler()
    sampler.start()
    total_start = time.perf_counter()

    started = time.perf_counter()
    pixels = grid_size * TILE_SIZE
    image = resize_to_minimum(
        load_raster_roi(raster_path, crop_x, crop_y, pixels),
        TILE_SIZE,
    )
    raster_latency = time.perf_counter() - started

    started = time.perf_counter()
    tiles, tile_names = slice_image(image, TILE_SIZE)
    tiling_latency = time.perf_counter() - started

    inference_timings = InferenceTimings()
    started = time.perf_counter()
    masks = inference_runner(model, tiles, batch_size, timings=inference_timings)
    inference_latency = time.perf_counter() - started

    started = time.perf_counter()
    stitched = merge_masks(masks, tile_names)
    stitching_latency = time.perf_counter() - started

    started = time.perf_counter()
    generate_outputs(image, stitched, work_dir / roi_name / "outputs")
    output_latency = time.perf_counter() - started

    total_latency = time.perf_counter() - total_start
    peak_memory = sampler.stop()
    tile_count = len(tiles)
    return {
        "raster_load_preprocess_ms": raster_latency * 1_000,
        "tiling_ms": tiling_latency * 1_000,
        "inference_preprocessing_ms": inference_timings.preprocessing_seconds * 1_000,
        "model_forward_ms": inference_timings.model_forward_seconds * 1_000,
        "inference_postprocessing_ms": inference_timings.postprocessing_seconds * 1_000,
        "inference_ms": inference_latency * 1_000,
        "stitching_ms": stitching_latency * 1_000,
        "output_generation_ms": output_latency * 1_000,
        "total_latency_ms": total_latency * 1_000,
        "tiles_per_second": tile_count / inference_latency,
        "model_tiles_per_second": tile_count / inference_timings.model_forward_seconds,
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
        "Load/preprocess (ms)",
        "Tiling (ms)",
        "Inference prep (ms)",
        "Model forward (ms)",
        "Inference post (ms)",
        "Inference total (ms)",
        "Stitch (ms)",
        "Output (ms)",
        "Total (ms)",
        "Tiles/s",
        "Model tiles/s",
        "Peak RSS (MB)",
    )
    metric_order = (
        "raster_load_preprocess_ms",
        "tiling_ms",
        "inference_preprocessing_ms",
        "model_forward_ms",
        "inference_postprocessing_ms",
        "inference_ms",
        "stitching_ms",
        "output_generation_ms",
        "total_latency_ms",
        "tiles_per_second",
        "model_tiles_per_second",
        "peak_process_memory_mb",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
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
    weights_path = selected_weights_path(args)
    results_dir = selected_results_dir(args)
    validate_inputs(args, weights_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = results_dir / "artifacts"

    tf.config.threading.set_intra_op_parallelism_threads(max(1, int(args.cpu_limit)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.set_synchronous_execution(True)

    metadata: dict[str, Any] = {
        "protocol": "Local-raster segmentation system benchmark",
        "model": args.model,
        "cpu_limit": args.cpu_limit,
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
        "source_raster": str(RASTER_PATH),
        "source_raster_sha256": file_sha256(RASTER_PATH),
        "weights_path": str(weights_path),
        "weights_sha256": file_sha256(weights_path),
        "tiles_per_roi": {name: grid * grid for name, grid in ROI_SPECS},
        "model_initialization_timed": False,
        "imagery_acquisition_timed": False,
        "tensorflow_synchronous_execution": True,
    }
    metadata.update(detected_cgroup_limits())
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Initialize and load once, then reuse the same resident model for all runs.
    model_adapter = MODEL_ADAPTERS[args.model]
    model = model_adapter.load(str(weights_path))

    raw_rows: list[dict[str, Any]] = []
    for roi_name, grid_size in ROI_SPECS:
        print(f"Warm-up: {roi_name} ({grid_size * grid_size} tiles)", flush=True)
        run_pipeline(
            RASTER_PATH,
            model,
            roi_name,
            grid_size,
            args.crop_x,
            args.crop_y,
            args.batch_size,
            work_dir,
            model_adapter.infer,
        )
        for run_number in range(1, args.measured_runs + 1):
            print(f"Measured run {run_number}/{args.measured_runs}: {roi_name}", flush=True)
            metrics = run_pipeline(
                RASTER_PATH,
                model,
                roi_name,
                grid_size,
                args.crop_x,
                args.crop_y,
                args.batch_size,
                work_dir,
                model_adapter.infer,
            )
            raw_rows.append(
                {
                    "model": args.model,
                    "roi": roi_name,
                    "run": run_number,
                    "width_px": grid_size * TILE_SIZE,
                    "height_px": grid_size * TILE_SIZE,
                    "tile_count": grid_size * grid_size,
                    "tile_size": TILE_SIZE,
                    "batch_size": args.batch_size,
                    "cpu_limit": args.cpu_limit,
                    "architecture": platform.machine(),
                    "tensorflow_version": tf.__version__,
                    "python_version": platform.python_version(),
                    **metrics,
                }
            )

    raw_fields = list(raw_rows[0].keys())
    write_csv(results_dir / "raw_runs.csv", raw_rows, raw_fields)
    summary_rows = summarize(raw_rows)
    write_csv(results_dir / "summary.csv", summary_rows, summary_rows[0].keys())
    write_paper_table(results_dir / "paper_table.md", summary_rows)
    print(f"Results written to {results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Benchmark preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
