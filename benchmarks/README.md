# Segmentation benchmark

This benchmark measures the portal's local-raster segmentation path without imagery acquisition or Streamlit rendering. By default, the portal and benchmark use the sequential windowed path in `ml/raster_inference.py`, with model inference from `ml/segmentation_core.py` or `ml/SegFormer.py`.

## Streaming and baseline comparison

`--pipeline streaming` (default) reads at most one batch of TIFF windows, infers it,
and writes predictions into a tiled, compressed `mask.tif`. Boundary windows shift
backwards without padding. Overlapping predictions use the original pixelwise
maximum merge. Source CRS and the crop transform are preserved. The GDAL block
cache is limited to 32 MiB during inference and preview generation.

`--pipeline in_memory` retains the original implementation as a baseline. Use
separate result directories (the default is `results_10_mruns/<model>/<pipeline>`); pipeline mode is recorded
in metadata and raw results. Historical runs used the in-memory path.

In streaming mode, `raster_load_preprocess_ms` sums batch window reads,
`inference_ms` sums inference calls, and `stitching_ms` includes mask window reads,
merges, and writes. `tiling_ms` is zero because window enumeration is lazy and not
separately timed. File setup, closing/compression, and enumeration overhead remain
included in total latency. The model timing definitions are unchanged.

Streaming PNG outputs are previews capped at 1024 pixels on the longest side;
`mask.tif` contains the full-resolution mask. The baseline generates full-resolution
PNGs, so end-to-end latency comparisons include this output-generation difference.
The benchmark excludes downloading imagery, portal rendering, and downloads.
In the portal, ZIP archives are created on disk only on request; Streamlit still
loads the requested archive into memory to serve its download.

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

Transformers 4.57.6 uses the legacy TensorFlow Keras API for this model, so the benchmark environment pins `tf-keras==2.20.1` alongside `tensorflow==2.20.0`.

Both Compose configurations supply their model and runtime arguments explicitly. Do not run the ARM64 protocol through AMD64 emulation.

The runner defaults to three excluded warm-up runs and five measured runs per ROI, and accepts at least one warm-up and five measurements. The local ENB5 Compose configuration explicitly uses one warm-up and five measurements to match the historical baseline; the Pegasus sweep explicitly uses three warm-ups and five measurements. Summary CSV columns use `_mean` and `_stddev`, and paper tables report arithmetic mean ± sample standard deviation across the measured runs. Historical result sets retain their original measurements and warm-up counts; their summaries can be recalculated from the saved raw runs without rerunning inference.

Ten-run results are stored in `results_10_mruns/`. Five-run results are preserved in `results_5_mruns/` with their original raw measurements and mean-based summaries. The `enb5_seg_islamabad_cpu2_b1_new` and `enb5_seg_islamabad_cpu2_b5_new` directories record graph-inference runs for comparison with the corresponding `enb5_seg_islamabad_b1` and `enb5_seg_islamabad_b5` baselines. Their metadata matches except for CPU affinity (six available CPUs versus four previously, with the same two-CPU quota), which limits attribution of the observed gains solely to graph execution. Generated PNG artifacts are excluded from Git.

The latest local ENB5 streaming batch-one run is saved in
`results_5_mruns_stream/enb5_seg_islamabad_cpu2_b1/`; the local Compose command
currently selects this configuration. Mean peak RSS is 1853, 1856, and 1877 MiB
for small, medium, and large ROIs, with total latency of 12.82, 49.70, and 112.07
seconds. Compared with the saved streaming batch-five session, this is about
60-65% less peak RSS. Against the historical in-memory graph batch-one session,
peak RSS changes by +3.9%, +2.1%, and -1.0%, respectively. These separate sessions
support reducing inference batch size on the local two-CPU setup, but do not
establish a substantial streaming memory benefit at the tested ROI sizes. The
pipelines also generate different PNG resolutions, as described above. The
Pegasus sweep retains its independent CPU and batch-size matrix.

## Pegasus CPU/batch sweep

`pegasus_sweep.sbatch` runs both models across CPUs **2, 4, 6** and batch sizes
**1-5**, with **16 GiB RAM**, **zero GPUs**, **3 excluded warm-ups**, and **5
measured runs per ROI**. One submission runs all 60 configurations sequentially
on one node. For each CPU/model/batch setting, `streaming` and `in_memory` run
back-to-back with a fresh Python process per pipeline. This keeps both pipelines
on the same CPU hardware and input staging. The local Compose batch setting
does not control this sweep.

### Build once

Keep the checkout under `/netscratch/$USER/<PROJECT>/`. From the repository
root on Pegasus, submit the automated CPU-only build:

```bash
sbatch benchmarks/pegasus_build.sbatch
```

Slurm queues the build and runs it automatically on a compute node when resources
are available. It requests four CPUs, 16 GiB RAM, zero GPUs, and four hours. You
can disconnect after submission; an interactive shell is unnecessary. The job
uses `/enroot/podman+enroot.sqsh` to run Podman and export the benchmark image.
Model checkpoints and the raster are needed for the sweep, but not for building.

The submission returns a `BUILD_JOB_ID`. Inspect it with bounded checks:

```bash
squeue -j BUILD_JOB_ID
tail -n 50 slurm-slum-i-build-BUILD_JOB_ID.out
```

On success, the log prints the image path and sweep submission command. The
image is saved as `benchmark-BUILD_JOB_ID.sqsh` in the repository root. Each job
uses its own image tag and `.pegasus-build/BUILD_JOB_ID/` directory for Podman
storage, caches, and Enroot files, all inside the project's scratch directory.
The storage driver is `vfs`, which avoids overlay requirements but uses more
scratch space. Build storage remains after completion for inspection; remove
only the relevant job directory when it is no longer needed.

The job stops if the build or export fails. It publishes the final image only
after a successful, non-empty export and refuses to overwrite an existing image.
A failed job can leave partial files under its build-storage directory. Submit
a new build job after correcting the problem. The build does not launch the
benchmark or perform the pending cluster smoke test automatically.

The Dockerfile copies the benchmark and model code into `/app`. Updating the
repository alone does not update an existing image. Rebuild after code or
dependency changes; each build job uses a new tag and `.sqsh` filename. Do not
reuse the original `benchmark-v1.sqsh` for this protocol unless it was rebuilt
from code supporting `--pipeline` and `--work-dir`. Build natively on Pegasus
using the commands above rather than reusing the local Apple Silicon image.

### Run the sweep

After the cluster smoke test, submit once from the repository root. Replace
`BUILD_JOB_ID` with the successful build job ID:

```bash
sbatch benchmarks/pegasus_sweep.sbatch "$PWD/benchmark-BUILD_JOB_ID.sqsh"
```

This runs both pipelines in one allocation; do not submit them separately for a
paired comparison. The image must support `--work-dir`, so rebuild older images.
The `.pegasus-build/` directory is excluded from Git and the image build context.
Keep generated `.sqsh` files out of version control as well.

Each submission returns one `JOB_ID`. Inspect it with one bounded status check:

```bash
squeue -j JOB_ID
tail -n 50 slurm-slum-i-sweep-JOB_ID.out
```

Results and logs are saved in `benchmarks/pegasus_results/JOB_ID/PIPELINE/`, with
the pipeline also included in configuration names and the `status.csv` column.
Check that file for completion (`exit_code=0`); failed configurations are
recorded and the remaining sweep continues. A missing status row indicates an
unfinished configuration. `allocation.txt` at the job root records both
pipelines, the image hash, repository revision, and allocation details.

Inputs and timed image outputs are staged under node-local `/tmp`. Metadata and
`raw_runs.csv` are written directly to shared result storage. Each completed
measurement is flushed and synced outside its timed region, so an OOM or timeout
retains completed rows even when a configuration never finishes. Summary CSVs
and paper tables are produced only after all measurements finish. Inspection
artifacts are copied to shared storage after each configuration; artifacts from
an interrupted step may remain incomplete or be lost. The runner's optional
`--work-dir` separates timed artifacts from durable results; local runs retain
the default `<results-dir>/artifacts` location.

The 16 GiB budget includes container and staging overhead; RAM is never increased
automatically. Keep the image and inputs fixed across repeats, and copy results
off scratch storage for safekeeping.

The job has a two-day (48-hour) time limit covering both pipelines. Actual
runtime and whether every model and batch size fits in 16 GiB must be checked
on Pegasus; local Mac measurements do not establish either. A cluster smoke
test remains pending before the full sweep. End-to-end comparisons also include
the different output resolutions described above.

## Sources of truth

- Protocol, model adapters, metrics, and result files: `system_benchmark.py`
- Container entry point, defaults, and build smoke test: `Dockerfile`
- Per-model runtime configuration: `compose.enb5_seg.yaml`, `compose.segformer.yaml`
- Container dependencies: `environment.yml`
- Raster requirements: `data/README.md`
- Shared portal and benchmark operations: `../ml/raster_inference.py` and `../ml/segmentation_core.py`
- Automated Pegasus image build: `pegasus_build.sbatch`
- Pegasus paired CPU/batch sweep and fixed 16 GiB allocation: `pegasus_sweep.sbatch`
