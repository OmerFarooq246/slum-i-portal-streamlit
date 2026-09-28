import numpy as np
import tensorflow as tf

from ml.segmentation_core import InferenceTimings, merge_masks, run_inference, slice_image


class ThresholdModel:
    def __call__(self, batch, training=False):
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
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    timings = InferenceTimings()

    masks = run_inference(ThresholdModel(), tiles, batch_size=2, timings=timings)

    np.testing.assert_array_equal(masks[0], np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(masks[1], np.full((2, 2), 255, dtype=np.uint8))
    assert timings.preprocessing_seconds > 0
    assert timings.model_forward_seconds > 0
    assert timings.postprocessing_seconds > 0
