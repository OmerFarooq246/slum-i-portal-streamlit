# Segmentation benchmark

This benchmark measures the portal's local-raster segmentation path without imagery acquisition or Streamlit rendering. The portal and benchmark use the same in-memory operations from `ml/segmentation_core.py`.

## Inputs

Provide the model checkpoint registered by the selected `MODEL_ADAPTERS` entry in `run_enb5_system_benchmark.py`. Provide the local benchmark raster described in `data/README.md`. These binary inputs are intentionally excluded from Git.

## Build and run

The Compose files contain the build context, platform, resource limits, network policy, mounts, and benchmark arguments. Run from the repository root on Apple Silicon.

ENB5-Seg:

```bash
docker compose -f benchmarks/compose.enb5_seg.yaml build
docker compose -f benchmarks/compose.enb5_seg.yaml run --rm benchmark
```

SegFormer, after its adapter, dependencies, and checkpoint are added:

```bash
docker compose -f benchmarks/compose.segformer.yaml build
docker compose -f benchmarks/compose.segformer.yaml run --rm benchmark
```

The ENB5-Seg configuration inherits the image's `CMD`. The SegFormer configuration replaces it with SegFormer arguments. Do not run the ARM64 protocol through AMD64 emulation.

## Sources of truth

- Protocol, model adapters, metrics, and result files: `run_enb5_system_benchmark.py`
- Container entry point, defaults, and build smoke test: `Dockerfile`
- Per-model runtime configuration: `compose.enb5_seg.yaml`, `compose.segformer.yaml`
- Container dependencies: `environment.yml`
- Raster requirements: `data/README.md`
- Shared portal and benchmark operations: `../ml/segmentation_core.py`
