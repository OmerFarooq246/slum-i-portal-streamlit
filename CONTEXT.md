# Slum-i Portal - Current Repository Context

## Purpose

Slum-i is a single-page Streamlit application for detecting informal settlements in satellite imagery. A user selects Lahore or Islamabad, draws a rectangle on a satellite map, and runs an EfficientNet-B5 semantic-segmentation model. The application displays the downloaded image, a binary detection mask, and a red overlay, and lets the user download those outputs as a ZIP archive.

The repository also contains a controlled local-raster benchmark. The application and benchmark share model loading, batched inference, and overlay generation through `ml/pipeline_omer.py`.

## Current structure

```text
.
├── app.py                              # Streamlit entry point
├── ml/
│   ├── ENB5_Segmenter.py               # EfficientNet-B5 encoder/decoder model
│   ├── pipeline_omer.py                # Active download, tiling, inference, merge, overlay pipeline
│   └── __init__.py
├── config/
│   ├── models.py                       # ModelConfig and Lahore/Islamabad registry
│   ├── settings.py                     # Temp directory, map presets, shared constants
│   └── __init__.py
├── ui/
│   ├── map.py                          # Folium satellite map and rectangle drawing
│   ├── results.py                      # Result display, removal, and ZIP download
│   └── __init__.py
├── utils/
│   ├── files.py                        # UUID-based temporary session directories
│   ├── state.py                        # Streamlit session-state initialization
│   └── __init__.py
├── benchmarks/
│   ├── run_enb5_system_benchmark.py    # Controlled local-raster benchmark
│   ├── Dockerfile                      # Micromamba-based Linux ARM64 benchmark image
│   ├── environment.yml                 # Benchmark-only Conda dependencies
│   ├── README.md                       # Benchmark protocol and commands
│   ├── data/README.md                  # Local benchmark raster instructions
│   └── results/.gitkeep                # Output-directory placeholder
├── tests/                               # Model-independent unit tests
├── .github/workflows/ci.yml             # Quality and dependency-audit CI
├── .pre-commit-config.yaml              # File hygiene and Ruff Git hooks
├── assets/screenshot.jpg               # README screenshot
├── environment.yml                     # App and development Conda environment
├── Makefile                             # Local run and quality commands
├── pyproject.toml                       # Ruff, mypy, pytest, and coverage configuration
└── README.md                            # Setup and usage summary
```

Model weights, temporary sessions, generated masks, and benchmark data/results are intentionally excluded from Git.

## Application execution path

The application is launched with:

```bash
conda activate ./.venv
python -m streamlit run app.py
```

The active flow is:

1. `app.py` initializes Streamlit state through `utils/state.py`.
2. `ui/map.py` renders a Google satellite Folium map and returns drawn rectangles.
3. `app.py` selects a `ModelConfig` from `config/models.py` based on the chosen city.
4. `ml.pipeline_omer.load_ENB5_Seg()` constructs the model architecture without ImageNet weights, loads the selected trained segmentation weights, and caches that initialized model by weights path.
5. When the user selects **Analyse area**, `ml.pipeline_omer.perform_segmentation()`:
   - creates `tmp/session_<uuid>/analysis/`;
   - converts the drawn coordinates into a bounding box;
   - downloads zoom-19 satellite imagery with `leafmap.map_tiles_to_geotiff()`;
   - saves the source raster and a PNG copy;
   - upscales images smaller than 512 pixels on either axis;
   - slices larger images into overlapping 512×512 edge-aligned tiles;
   - normalizes tile values to `[0, 1]` and runs batched inference;
   - converts class predictions into a binary `uint8` mask containing 0 or 255;
   - merges overlapping masks with pixel-wise maximum;
   - creates a 50%-alpha red overlay;
   - stores PIL images and metadata in `st.session_state.processing_results`.
6. `ui/results.py` displays the original image, mask, and overlay and creates an in-memory ZIP download.

## Models and configuration

`config/models.py` defines two active configurations:

| Registry key | Region | Weights path | Batch size |
|---|---|---|---:|
| `enb5_lahore` | Lahore | `models/enb5_seg_lahore.h5` | 2 |
| `enb5_islamabad` | Islamabad | `models/enb5_seg_islamabad.h5` | 2 |

`ml/ENB5_Segmenter.py` builds an EfficientNet-B5 encoder with a custom decoder. Its configured input and output resolution is 512×512 with two output classes. The encoder is created with `weights=None`; the complete trained segmentation weights are loaded immediately afterward, so no separate ImageNet EfficientNet file or Keras download is needed.

The `.h5` model files are not committed to the repository.

## Shared pipeline API

`ml/pipeline_omer.py` is the only active segmentation pipeline module.

- `load_ENB5_Seg(weights_path)` builds the architecture without pretrained encoder weights, loads the supplied trained segmentation weights, and caches the initialized model by path. Both the app and benchmark use this function.
- `perform_segmentation(polygon_coords, model_config, model)` runs the complete Streamlit spatial-analysis workflow.
- `run_inference(model, tiles, batch_size)` performs the shared normalized, batched inference.
- `merge_masks(masks, tile_names)` reconstructs a full mask from `tile_X_Y` coordinates.
- `create_overlay_image(image, mask, color, alpha)` creates the shared RGB overlay.

Leafmap and temporary-session helpers are imported only by the complete application workflow. This keeps the benchmark able to import the shared model and inference helpers without adding application-only geospatial dependencies to its container.

## Session and output shape

Each application result is stored under a UUID key in `st.session_state.processing_results`:

```text
session_id
session_dir
bbox
polygon_coords
primary_result
├── original_image
├── stitched_mask
└── overlay_image
```

Removing a result deletes its temporary session directory. **Clear all** deletes every recorded session directory. The `tmp/` directory itself is created by `config/settings.py` and ignored by Git.

## Benchmark

`benchmarks/run_enb5_system_benchmark.py` operates on a supplied local raster and weights file. It does not download imagery. It uses `ml.pipeline_omer` for model loading, inference, and overlays while retaining benchmark-specific raster cropping, tile-file generation, timing, memory sampling, result aggregation, and output writing.

The fixed benchmark protocol uses:

- 512×512 tiles;
- 2×2, 4×4, and 6×6 tile grids;
- one warm-up and five measured runs per grid size;
- CSV, Markdown, JSON metadata, mask, and overlay outputs;
- SHA-256 hashes for the raster and weights inputs.

See `benchmarks/README.md` for the ARM64 Docker build and run commands.

## Python environment and quality checks

Conda manages the Python 3.11 application and development environment from `environment.yml`. The local environment is created at `.venv/` in the repository with `conda env create --prefix ./.venv --file environment.yml` and activated from the repository root with `conda activate ./.venv`. Conda-forge supplies GDAL, its Python bindings, PROJ, and the rest of the geospatial stack so their native libraries are solved together on Apple Silicon and Linux. TensorFlow 2.20 and its Keras dependency are installed into that environment from TensorFlow's official pip wheel because conda-forge does not publish the requested TensorFlow build for Apple Silicon. PyArrow 23 also comes from its official wheel: loading TensorFlow with conda-forge's PyArrow 24 native build segfaults on Apple Silicon, while the wheel combination is compatible. NumPy is pinned in both dependency sections so pip cannot replace Conda's intended 1.26.4 version. The benchmark Docker image uses Micromamba and the smaller `benchmarks/environment.yml` environment with the same Conda-plus-pip split.

`pyproject.toml` contains only Ruff, mypy, pytest, and coverage configuration. Dependency declarations live only in the two Conda environment files so the application and benchmark definitions do not drift.

After activating `./.venv`, run `make check` for the text policy, formatting/lint verification, type checking, and unit tests. `make coverage` emits terminal coverage and `coverage.xml`, while `make audit` checks installed Python packages for known vulnerabilities. Pre-commit runs repository hygiene and Ruff checks, pre-push blocks direct pushes to `main`, and both hook stages reject em dashes. GitHub Actions creates the Conda environment and repeats the complete quality suite and dependency audit on pull requests and pushes to `main`.

## Generated and ignored paths

- `models/`: local model weights
- `tmp/`: application session data
- `public/outputs/`: legacy/generated output location retained in ignore rules
- `benchmarks/data/*`: local benchmark rasters, except its README
- `benchmarks/results/*`: generated benchmark results, except `.gitkeep`
- `__pycache__/` and `*.pyc`: Python bytecode
- `.venv/` and tool caches: local environment and quality-tool state
- `.coverage` and `coverage.xml`: local coverage reports
