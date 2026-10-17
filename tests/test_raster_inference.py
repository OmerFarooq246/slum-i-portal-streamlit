import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from ml.raster_inference import infer_raster, raster_previews, tile_windows
from ml.segmentation_core import ImageArray, merge_masks, slice_image


def write_raster(path: Path, image: ImageArray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=image.shape[2],
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(100, 200, 2, 2),
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0))


def context_masks(tiles):
    # Depend on tile context so overlapping predictions can disagree.
    return [np.full(tile.shape[:2], 255 if tile[0, 0, 0] > 127 else 0, np.uint8) for tile in tiles]


def test_streaming_matches_legacy_overlap_and_releases_batches(tmp_path: Path) -> None:
    image = np.random.default_rng(7).integers(0, 256, (513, 1025, 3), dtype=np.uint8)
    source, output = tmp_path / "input.tif", tmp_path / "mask.tif"
    write_raster(source, image)
    tiles, names = slice_image(image, 512)
    expected = merge_masks(context_masks(tiles), names)
    previous: list[Any] = []
    sizes = []

    def infer(model, batch, batch_size, timings):
        assert all(ref() is None for ref in previous)
        assert batch_size == 4
        sizes.append(len(batch))
        masks = context_masks(batch)
        previous[:] = [weakref.ref(array) for array in [*batch, *masks]]
        return masks

    result = infer_raster(source, output, None, 4, infer)
    assert sizes == [4, 2]
    assert result.tile_count == 6
    assert all(ref() is None for ref in previous)
    with rasterio.open(output) as dataset, rasterio.open(source) as original:
        np.testing.assert_array_equal(dataset.read(1), expected)
        assert dataset.count == 1
        assert dataset.crs == original.crs
        assert dataset.transform == original.transform
    assert not output.with_suffix(".partial.tif").exists()


def test_windows_preserve_shifted_edges() -> None:
    assert [(w.col_off, w.row_off) for w in tile_windows(1025, 513)] == [
        (0, 0),
        (512, 0),
        (513, 0),
        (0, 1),
        (512, 1),
        (513, 1),
    ]


def test_region_transform_and_preview_dimensions(tmp_path: Path) -> None:
    image = np.full((600, 1100, 3), 255, np.uint8)
    source, output = tmp_path / "input.tif", tmp_path / "mask.tif"
    write_raster(source, image)
    region = Window(10, 20, 1025, 513)
    infer_raster(
        source,
        output,
        None,
        3,
        lambda model, tiles, batch_size, timings: context_masks(tiles),
        region=region,
    )
    with rasterio.open(output) as dataset, rasterio.open(source) as original:
        assert (dataset.width, dataset.height) == (1025, 513)
        assert dataset.transform == original.window_transform(region)
    rgb, mask = raster_previews(source, output, region=region, max_size=100)
    assert rgb.shape == (50, 100, 3)
    assert mask.shape == (50, 100)
    assert set(np.unique(mask)) == {255}
    np.testing.assert_array_equal(rgb, 255)


@pytest.mark.parametrize("bands", [1, 2, 4])
def test_rgb_conversion(tmp_path: Path, bands: int) -> None:
    image = np.full((512, 512, bands), 160, np.uint8)
    source, output = tmp_path / "input.tif", tmp_path / "mask.tif"
    write_raster(source, image)

    def infer(model, tiles, batch_size, timings):
        assert tiles[0].shape == (512, 512, 3)
        return context_masks(tiles)

    infer_raster(source, output, None, 1, infer)
    rgb, mask = raster_previews(source, output)
    assert rgb.shape == (512, 512, 3)
    assert mask.shape == (512, 512)


def test_failed_inference_does_not_replace_completed_output(tmp_path: Path) -> None:
    source, output = tmp_path / "input.tif", tmp_path / "mask.tif"
    write_raster(source, np.zeros((512, 512, 3), np.uint8))
    output.write_bytes(b"previous result")

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        infer_raster(source, output, None, 2, fail)
    assert output.read_bytes() == b"previous result"
    assert not output.with_suffix(".partial.tif").exists()


@pytest.mark.parametrize(
    "batch_size,region",
    [
        (0, None),
        (1, Window(-1, 0, 512, 512)),
        (1, Window(0, 0, 513, 512)),
        (1, Window(0.5, 0, 511, 512)),
        (1, Window(0, 0, 511, 512)),
    ],
)
def test_rejects_invalid_inputs(tmp_path: Path, batch_size: int, region: Window | None) -> None:
    source, output = tmp_path / "input.tif", tmp_path / "mask.tif"
    write_raster(source, np.zeros((512, 512, 3), np.uint8))
    with pytest.raises(ValueError):
        infer_raster(source, output, None, batch_size, region=region)
    assert not output.exists()
