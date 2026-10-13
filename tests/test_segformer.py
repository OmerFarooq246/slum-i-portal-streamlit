import numpy as np
import tensorflow as tf
from transformers import (
    SegformerConfig,
    SegformerImageProcessor,
    TFSegformerForSemanticSegmentation,
)

from ml.SegFormer import SEGFORMER_BASE_CONFIG_PATH, SegFormer, run_segformer_inference
from ml.segmentation_core import InferenceTimings


def test_segformer_graph_matches_eager_and_reuses_partial_batch_trace() -> None:
    # Exercise the real processor and model without a checkpoint or network access.
    segmenter = SegFormer.__new__(SegFormer)
    segmenter.image_processor = SegformerImageProcessor.from_pretrained(
        str(SEGFORMER_BASE_CONFIG_PATH), local_files_only=True
    )
    segmenter.model = TFSegformerForSemanticSegmentation(
        SegformerConfig(
            num_labels=2,
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            num_attention_heads=[1, 2, 4, 8],
            decoder_hidden_size=16,
        )
    )
    rng = np.random.default_rng(42)
    tiles = [rng.integers(0, 256, (32, 48, 3), dtype=np.uint8) for _ in range(3)]
    inputs = segmenter.image_processor(images=tiles, return_tensors="tf")
    outputs = segmenter.model(pixel_values=inputs["pixel_values"], training=False)
    logits = tf.image.resize(tf.transpose(outputs.logits, (0, 2, 3, 1)), (32, 48))
    expected = (tf.argmax(logits, axis=-1) * 255).numpy().astype(np.uint8)

    run_segformer_inference(segmenter, tiles[:2], batch_size=2)
    trace_count = segmenter.infer_masks.experimental_get_tracing_count()
    timings = InferenceTimings()
    actual = run_segformer_inference(segmenter, tiles, batch_size=2, timings=timings)

    np.testing.assert_array_equal(np.stack(actual), expected)
    assert all(mask.dtype == np.uint8 for mask in actual)
    assert segmenter.infer_masks.experimental_get_tracing_count() == trace_count == 1
    assert timings.preprocessing_seconds > 0
    assert timings.model_forward_seconds > 0
    assert timings.postprocessing_seconds > 0

    resized_tile = rng.integers(0, 256, (48, 32, 3), dtype=np.uint8)
    resized_masks = run_segformer_inference(segmenter, [resized_tile], batch_size=2)
    assert resized_masks[0].shape == (48, 32)
    assert segmenter.infer_masks.experimental_get_tracing_count() == trace_count
