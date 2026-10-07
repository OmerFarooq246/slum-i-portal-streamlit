# Slum-i Portal Repository Context

## Purpose

Slum-i is a Streamlit portal for detecting informal settlements in satellite imagery. It also includes a controlled, Docker-based benchmark for measuring the shared segmentation pipeline and comparing model performance.

This file records only repository-level relationships and boundaries. Implementation details, configuration values, commands, and benchmark protocol belong in their authoritative files listed below.

## Architecture

The application and benchmark share the dependency-light operations in `ml/segmentation_core.py`, including model loading, image preparation, in-memory tiling, inference, mask merging, and overlay creation.

The interactive application is assembled by `app.py` and `ml/pipeline_omer.py`. Portal-only concerns such as Streamlit state, imagery acquisition, session directories, and user feedback remain outside the shared core.

The benchmark is implemented by `benchmarks/system_benchmark.py`. It uses the shared core but owns benchmark-specific concerns such as local raster cropping, timing, memory sampling, model adapters, aggregation, and result files. It does not benchmark imagery acquisition or Streamlit rendering.

## Sources of truth

- Project setup and portal usage: `README.md`
- Application entry point and UI flow: `app.py`, `ui/`
- Model registry, checkpoints, and portal batch sizes: `config/models.py`
- Shared constants and map presets: `config/settings.py`
- ENB5-Seg architecture: `ml/ENB5_Segmenter.py`
- Shared segmentation operations: `ml/segmentation_core.py`
- Portal-specific pipeline adapter: `ml/pipeline_omer.py`
- Benchmark protocol, Docker commands, inputs, and outputs: `benchmarks/README.md`
- Benchmark implementation and model adapters: `benchmarks/system_benchmark.py`
- Benchmark container dependencies: `benchmarks/environment.yml`
- Application and development dependencies: `environment.yml`
- Quality-tool configuration and commands: `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`
- Local benchmark raster requirements: `benchmarks/data/README.md`

## Repository boundaries

- Model checkpoints, benchmark rasters, benchmark results, and temporary portal sessions are local/generated artifacts and are not committed.
- The benchmark keeps tiles in memory and saves only final inspection artifacts and result files.
- Each benchmark model is registered through its model adapter so model-specific loading and inference can change without duplicating the shared spatial pipeline.
- Documentation should reference the sources of truth above instead of copying values or implementation details into this file.
