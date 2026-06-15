# Slum-i — Project Context

## Overview

Slum-i is a single-page Streamlit dashboard for detecting informal settlements in satellite imagery using a TensorFlow segmentation model. The app lets users draw a bounding box on a map, downloads the corresponding satellite tiles, runs slum detection, and displays the original image, binary mask, and overlay side-by-side.

---

## Project Structure

```
slum-i/
├── app.py                  # Entry point
├── ui/
│   ├── sidebar.py          # Controls: model, date, location, tile size
│   ├── map.py              # Folium map with draw plugin + KML support
│   └── results.py          # Summary expander, 3-column image display, zip download
├── pipeline/
│   └── tiles.py            # Core pipeline: download → slice → infer → stitch
├── ml/
│   ├── loader.py           # @st.cache_resource TF model loader
│   └── images.py           # generate_mask + create_overlay_image
├── config/
│   ├── settings.py         # Zoom level, supported dates, preset locations, basemaps
│   └── models.py           # ModelConfig dataclass + model registry
├── utils/
│   ├── state.py            # Session state init/cleanup
│   └── files.py            # UUID-based session dirs under tmp/
├── models/
│   └── model.h5            # Placeholder — place real TF model here
└── requirements.txt
```

---

## Pipeline

1. User draws a rectangle on the Folium map
2. Clicks "Process Spatial Analysis" → `pipeline/tiles.py:process_polygon`
3. **Download** — `leafmap.map_tiles_to_geotiff` fetches tiles at zoom=19 for the selected date
4. **Resize** — image trimmed to nearest 512px multiple, saved as GeoTIFF
5. **Tile** — GeoTIFF sliced into 512×512 (or 256×256) PNG patches
6. **Inference** — each tile passed through the TF model via `ml/images.py:generate_mask` → binary mask
7. **Stitch** — masks reassembled into a full-region mask; dotted lines mark tile boundaries
8. **Overlay** — mask blended semi-transparently (red, α=0.5) over original image
9. Results stored in `st.session_state.processing_results` keyed by UUID

---

## Configuration

| Setting | Value |
|---|---|
| Fixed zoom level | 19 |
| Default tile size | 512px (256px also available) |
| Models | `enb5_seg_lahore.h5` (Lahore), `enb5_seg_islamabad.h5` (Islamabad) — ~109 MB each |
| Supported dates | Live (2025-06-26), Dec 2024, Jun 2024, Dec 2023 — via ArcGIS Wayback WMS |
| Preset locations | Lahore, Islamabad, Karachi, Mumbai |
| Basemaps | OpenStreetMap, Google Satellite |

---

## Type System

All modules use full type annotations. Key TypedDicts:

- `ModelConfig` — frozen dataclass: `name`, `path`, `regions`, `model`
- `SidebarData`, `LocationInfo`, `DateSelection`
- `TileInfo`, `InferenceResult`, `DateResult`, `SpatialResult`
- `SessionState`, `TiffData`

---

## Open TODOs (`ml/images.py:generate_mask`)

1. Confirm model input size — add resizing if the model expects a fixed resolution
2. Confirm normalization — currently raw float32 with no scaling
3. Confirm output format — currently assumes single-channel output thresholded at 0.5

---

## Setup

```bash
conda activate <gdal-env>
streamlit run app.py
```

Dependencies: `tensorflow`, `streamlit`, `rasterio`, `geopandas`, `pillow`, `gdal`, `streamlit-folium`, `folium`, `leafmap`, `owslib`, `numpy==1.26.4`

---

## Refactor History (April 2026)

- Renamed from `main.py` → `app.py`; `components/` → `ui/`; `core/` → `pipeline/`; added `ml/`
- Removed `views/` (home, single-image inference, spatio-temporal analysis)
- Removed temporal analysis and PyTorch dependencies
- Switched to TensorFlow `.h5` models
- Added full typing across all modules
- Converted multi-view router to single-page spatial analysis app
