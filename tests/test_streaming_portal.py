from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.transform import from_origin

from benchmarks.system_benchmark import run_pipeline
from ml import pipeline_omer
from ui.results import _build_zip


def test_portal_streams_download_and_archives_full_resolution(tmp_path: Path, monkeypatch) -> None:
    def download(output, **kwargs):
        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            width=1025,
            height=513,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(100, 200, 1, 1),
        ) as dataset:
            dataset.write(np.full((3, 513, 1025), 255, np.uint8))

    class Model:
        def infer_masks(self, batch):
            return (
                tf.cast(tf.reduce_mean(tf.cast(batch, tf.float32), axis=-1) > 127, tf.uint8) * 255
            )

    monkeypatch.setattr(pipeline_omer.leafmap, "map_tiles_to_geotiff", download)
    monkeypatch.setattr(pipeline_omer, "create_session_dir", lambda: ("session123", str(tmp_path)))
    state = SimpleNamespace(processing_results={})
    monkeypatch.setattr(pipeline_omer.st, "session_state", state)
    monkeypatch.setattr(pipeline_omer.st, "success", lambda *args: None)
    monkeypatch.setattr(pipeline_omer.st, "rerun", lambda: None)
    model: Any = Model()
    config: Any = SimpleNamespace(batch_size=4)
    pipeline_omer.perform_segmentation([[74.3, 31.5], [74.31, 31.51]], config, model)
    result = state.processing_results["session123"]
    primary = result["primary_result"]
    assert primary["raster_size"] == (1025, 513)
    assert primary["tile_count"] == 6
    for key in ["original_image", "stitched_mask", "overlay_image"]:
        assert max(primary[key].size) == 1024
    with rasterio.open(primary["mask_path"]) as mask:
        assert (mask.width, mask.height) == (1025, 513)
        np.testing.assert_array_equal(mask.read(1), 255)
    assert not list(tmp_path.glob("*.zip"))
    archive = _build_zip("session123", result)
    with ZipFile(archive) as contents:
        assert set(contents.namelist()) == {
            "original.tif",
            "mask.tif",
            "original_session1.png",
            "mask_session1.png",
            "overlay_session1.png",
        }
        assert contents.read("mask.tif") == Path(primary["mask_path"]).read_bytes()

    for pipeline in ["streaming", "in_memory"]:
        metrics = run_pipeline(
            Path(primary["raster_path"]),
            model,
            "smoke",
            1,
            0,
            0,
            4,
            tmp_path / pipeline,
            pipeline=pipeline,
        )
        assert metrics["total_latency_ms"] > 0
        assert metrics["model_forward_ms"] > 0
        assert metrics["peak_process_memory_mb"] > 0
