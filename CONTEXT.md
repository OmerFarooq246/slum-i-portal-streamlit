# Slum-i Portal — Current Repository Context

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
│   ├── Dockerfile                      # Linux ARM64 benchmark image
│   ├── requirements.txt                # Benchmark-only dependencies
│   ├── README.md                       # Benchmark protocol and commands
│   ├── data/README.md                  # Local benchmark raster instructions
│   └── results/.gitkeep                # Output-directory placeholder
├── assets/screenshot.jpg               # README screenshot
├── requirements.txt                    # Application dependencies
└── README.md                            # Setup and usage summary
```

Model weights, temporary sessions, generated masks, and benchmark data/results are intentionally excluded from Git.

## Application execution path

The application is launched with:

```bash
streamlit run app.py
```

The active flow is:

1. `app.py` initializes Streamlit state through `utils/state.py`.
2. `ui/map.py` renders a Google satellite Folium map and returns drawn rectangles.
3. `app.py` selects a `ModelConfig` from `config/models.py` based on the chosen city.
4. `ml.pipeline_omer.load_ENB5_Seg()` constructs and caches the model architecture.
5. When the user selects **Analyse area**, `ml.pipeline_omer.perform_segmentation()`:
   - creates `tmp/session_<uuid>/analysis/`;
   - converts the drawn coordinates into a bounding box;
   - downloads zoom-19 satellite imagery with `leafmap.map_tiles_to_geotiff()`;
   - saves the source raster and a PNG copy;
   - upscales images smaller than 512 pixels on either axis;
   - slices larger images into overlapping 512×512 edge-aligned tiles;
   - loads the selected city's model weights;
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

`ml/ENB5_Segmenter.py` builds an EfficientNet-B5 encoder with a custom decoder. Its configured input and output resolution is 512×512 with two output classes. Because the encoder is created with `weights="imagenet"`, Keras resolves its EfficientNet-B5 encoder weights through the Keras model cache and may download them when networking is available. The benchmark mounts `efficientnetb5_notop.h5` directly into that cache and disables networking.

The `.h5` model files are not committed to the repository.

## Shared pipeline API

`ml/pipeline_omer.py` is the only active segmentation pipeline module.

- `load_ENB5_Seg(weights_path=None)` builds and caches a model. The app omits the path and loads city weights when analysis starts; the benchmark supplies a path and loads weights once before timing.
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

## Generated and ignored paths

- `models/` — local model weights
- `tmp/` — application session data
- `public/outputs/` — legacy/generated output location retained in ignore rules
- `benchmarks/data/*` — local benchmark rasters, except its README
- `benchmarks/results/*` — generated benchmark results, except `.gitkeep`
- `__pycache__/` and `*.pyc` — Python bytecode
