# Segmentation benchmark

This benchmark measures the portal's local-raster segmentation path without imagery acquisition or Streamlit rendering. The portal and benchmark use the same in-memory operations from `ml/segmentation_core.py`.

## Inputs

Provide the artifacts registered by the selected `MODEL_ADAPTERS` entry in `system_benchmark.py`:

- ENB5-Seg: `models/enb5_seg_islamabad.h5`
- SegFormer MiT-B5 configuration and image processor: a local copy at `models/mit-b5/`
- SegFormer fine-tuned weights: `models/segformer_b5_islamabad.h5`

Create the small local MiT-B5 configuration bundle once, before running the network-isolated benchmark:

```bash
python - <<'PY'
from transformers import SegformerConfig, SegformerImageProcessor

destination = "models/mit-b5"
SegformerConfig.from_pretrained("nvidia/mit-b5").save_pretrained(destination)
SegformerImageProcessor.from_pretrained("nvidia/mit-b5").save_pretrained(destination)
PY
```

Generate the local benchmark raster from the repository root:

```bash
python -m scripts.benchmark_download_raster
```

The download script writes `benchmarks/data/raster_benchmark.tif` using its configured bounding box and the fixed zoom level from `config/settings.py`. The current configuration has been verified to produce a 4096 x 4096 raster, which safely covers the benchmark's largest 3072 x 3072 ROI. See `data/README.md` for verification and crop-offset details. Model checkpoints and the generated raster are intentionally excluded from Git.

## Build and run

The Compose files contain the build context, platform, resource limits, network policy, mounts, and benchmark arguments. Run from the repository root on Apple Silicon.

ENB5-Seg:

```bash
docker compose -f benchmarks/compose.enb5_seg.yaml build
docker compose -f benchmarks/compose.enb5_seg.yaml run --rm benchmark
```

SegFormer:

```bash
docker compose -f benchmarks/compose.segformer.yaml build
docker compose -f benchmarks/compose.segformer.yaml run --rm benchmark
```

The SegFormer adapter reconstructs the same two-class MiT-B5 architecture used for training and then calls `load_weights()` with the fine-tuned weights. The `models/mit-b5/` directory only needs `config.json` and `preprocessor_config.json`; a second copy of the base model weights is unnecessary. The configuration, processor metadata, and fine-tuned weights must be local because benchmark containers run without network access.

Transformers 4.57.3 uses the legacy TensorFlow Keras API for this model, so the benchmark environment pins `tf-keras==2.20.1` alongside `tensorflow==2.20.0`.

The ENB5-Seg configuration inherits the image's `CMD`. The SegFormer configuration replaces it with SegFormer arguments. Do not run the ARM64 protocol through AMD64 emulation.

## Sources of truth

- Protocol, model adapters, metrics, and result files: `system_benchmark.py`
- Container entry point, defaults, and build smoke test: `Dockerfile`
- Per-model runtime configuration: `compose.enb5_seg.yaml`, `compose.segformer.yaml`
- Container dependencies: `environment.yml`
- Raster requirements: `data/README.md`
- Shared portal and benchmark operations: `../ml/segmentation_core.py`
