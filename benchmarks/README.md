# ENB5-Seg system-performance benchmark

This benchmark measures the portal's ENB5-Seg path from an already-downloaded local raster. Model loading, batched inference, and overlay generation use the same `ml/pipeline_omer.py` helpers as the Streamlit app. It performs no imagery acquisition and does not import or call `leafmap.map_tiles_to_geotiff()`.

For each deterministic, nested top-left crop it runs one unmeasured warm-up followed by five measured runs. Model construction and weight loading happen once before the warm-ups; the initialized model remains resident. The three crops are exactly 1024×1024 (4 tiles), 2048×2048 (16 tiles), and 3072×3072 (36 tiles), using 512×512 tiles.

## Required local files

The repository does not include the required binary inputs. Supply:

- `models/enb5_seg_islamabad.h5`: the trained ENB5-Seg weights.
- `benchmarks/data/raster_benchmark.tif`: a local raster with at least three useful image bands and dimensions of at least 3072×3072 pixels. Larger rasters are deterministically cropped from pixel offset `(0, 0)` by default.

Record the provenance/licensing of the source raster separately. The benchmark records SHA-256 hashes of both supplied benchmark inputs in `metadata.json`.

## Build and run on Apple Silicon

Run these commands from the repository root. They explicitly build and run Linux ARM64, constrain the container to 2 CPUs and 4 GB RAM, disable GPU visibility, and disable runtime networking:

```bash
test "$(uname -m)" = "arm64"
test "$(docker info --format '{{.Architecture}}')" = "aarch64"

mkdir -p benchmarks/results

docker buildx build \
  --platform linux/arm64 \
  --load \
  -f benchmarks/Dockerfile \
  -t slum-i-enb5-benchmark:arm64 \
  .

docker run --rm \
  --platform linux/arm64 \
  --cpus 2 \
  --memory 4g \
  --memory-swap 4g \
  --network none \
  --env CUDA_VISIBLE_DEVICES=-1 \
  --mount type=bind,src="$(pwd)/models",dst=/app/models,readonly \
  --mount type=bind,src="$(pwd)/benchmarks/data",dst=/app/benchmarks/data,readonly \
  --mount type=bind,src="$(pwd)/benchmarks/results",dst=/app/benchmarks/results \
  slum-i-enb5-benchmark:arm64 \
  --batch-size 1 \
  --cpu-limit 2
```

The raster, weights, and results paths are fixed in the runner. The benchmark constructs the architecture without downloading ImageNet weights, then loads the trained segmentation weights from `/app/models`. The container therefore remains fully offline at runtime. Do not add `--platform linux/amd64`; that would invoke emulation and invalidate this protocol.

To use a different deterministic crop origin, pass both `--crop-x N --crop-y N`. Keep those values fixed across compared runs.

## Results

The runner writes:

- `raw_runs.csv`: all 15 measured runs and every requested stage metric.
- `summary.csv`: median and sample standard deviation for each metric by ROI.
- `paper_table.md`: concise median ± SD table.
- `metadata.json`: limits, detected cgroup settings, platform/architecture, versions, input hashes, tile/batch configuration, and tile counts.
- `artifacts/`: latest tile, mask, and overlay outputs for inspection (overwritten on each run).

All latency fields in `raw_runs.csv`, `summary.csv`, and `paper_table.md` are reported in milliseconds (`*_ms`). Throughput remains reported as tiles per second.

Peak memory is sampled process RSS every 10 ms across the complete per-ROI pipeline. Total latency starts before local raster reading and ends after mask and overlay PNGs are written. It excludes model initialization, the unmeasured warm-up, result CSV aggregation, and all imagery acquisition.
