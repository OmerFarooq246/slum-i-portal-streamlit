from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.transform import from_origin
from tensorflow import keras

from config.settings import TILE_SIZE
from ml.ENB5_Segmenter import EfficientNetB5_Segmenter
from ml.raster_inference import infer_raster
from ml.segmentation_core import InferenceTimings, merge_masks, run_inference, slice_image


class ThresholdModel(EfficientNetB5_Segmenter):
    def __init__(self):
        # Use the production infer_masks method with a lightweight forward pass.
        keras.Model.__init__(self)

    def call(self, batch, training=False):
        del training
        foreground = tf.reduce_mean(batch, axis=-1)
        return tf.stack((1.0 - foreground, foreground), axis=-1)


def test_slice_and_merge_use_portal_coordinates_in_memory() -> None:
    image = tf.reshape(tf.range(4 * 4 * 3, dtype=tf.int32), (4, 4, 3))

    tiles, names = slice_image(image, tile_size=2)
    masks = [np.full((2, 2), index, dtype=np.uint8) for index in range(1, 5)]
    merged = merge_masks(masks, names)

    assert names == ["tile_0_0", "tile_2_0", "tile_0_2", "tile_2_2"]
    assert len(tiles) == 4
    np.testing.assert_array_equal(
        merged,
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.uint8,
        ),
    )


def test_run_inference_profiles_shared_batch_path() -> None:
    tiles = [
        np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8),
        np.full((TILE_SIZE, TILE_SIZE, 3), 255, dtype=np.uint8),
    ]
    timings = InferenceTimings()

    masks = run_inference(ThresholdModel(), tiles, batch_size=2, timings=timings)

    np.testing.assert_array_equal(masks[0], np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8))
    np.testing.assert_array_equal(masks[1], np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8))
    assert timings.preprocessing_seconds > 0
    assert timings.model_forward_seconds > 0
    assert timings.postprocessing_seconds > 0


def test_graph_matches_eager_masks_and_reuses_trace_for_partial_batches(tmp_path: Path) -> None:
    model = ThresholdModel()
    rng = np.random.default_rng(42)
    tiles = [
        rng.integers(0, 256, (TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8),
        np.full((TILE_SIZE, TILE_SIZE, 3), 127, dtype=np.uint8),
        np.full((TILE_SIZE, TILE_SIZE, 3), 128, dtype=np.uint8),
    ]
    predictions = model(np.stack(tiles).astype(np.float32) / 255.0, training=False)
    expected = (tf.argmax(predictions, axis=-1) * 255).numpy().astype(np.uint8)

    run_inference(model, tiles[:2], batch_size=2)
    inference = model.infer_masks
    trace_count = inference.experimental_get_tracing_count()
    actual = run_inference(model, tiles, batch_size=2)

    np.testing.assert_array_equal(np.stack(actual), expected)
    assert all(mask.dtype == np.uint8 for mask in actual)
    assert model.infer_masks is inference
    assert inference.experimental_get_tracing_count() == trace_count == 1

    # The disk-backed path uses the same graph and preserves the partial batch.
    image = np.concatenate(tiles, axis=1)
    source, output = tmp_path / "image.tif", tmp_path / "mask.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=TILE_SIZE,
        width=3 * TILE_SIZE,
        count=3,
        dtype="uint8",
        transform=from_origin(100, 200, 1, 1),
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0))
    infer_raster(source, output, model, batch_size=2)
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), np.concatenate(expected, axis=1))
    assert inference.experimental_get_tracing_count() == 1
