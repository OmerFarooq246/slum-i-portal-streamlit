# Benchmark raster

Generate `raster_benchmark.tif` from the repository root:

```bash
python -m scripts.benchmark_download_raster
```

The downloader uses the bounding box configured in the script and the fixed zoom level from `config/settings.py`. It overwrites an existing benchmark raster. Its bounding-box order is west, south, east, north.

The resulting raster must be at least 3072 x 3072 pixels. The current downloader configuration has been verified to produce a 4096 x 4096 raster:

```bash
gdalinfo benchmarks/data/raster_benchmark.tif | grep "Size is"
# Size is 4096, 4096
```

The benchmark reads fixed nested crops from pixel offset `(0, 0)` by default. To center the largest 3072 x 3072 crop within this raster, pass `--crop-x 512 --crop-y 512`. Keep the selected offsets consistent between benchmark runs; they are recorded in `metadata.json`.

The generated raster is excluded from Git.
