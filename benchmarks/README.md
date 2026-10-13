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

Both Compose configurations supply their model and runtime arguments explicitly. Do not run the ARM64 protocol through AMD64 emulation.

Each ROI uses one excluded warm-up run and at least 10 measured runs (the default). Set `--measured-runs` to a larger value to collect more measurements; values below 10 are rejected. Summary CSV columns use `_mean` and `_stddev`, and paper tables report arithmetic mean ± sample standard deviation across the measured runs. Historical result sets retain their original five measurements; their summaries can be recalculated from the saved raw runs without rerunning inference.

Results from the current protocol are stored in `results_10_mruns/`. Historical five-run results are preserved in `results_5_mruns/` with their original raw measurements and mean-based summaries. Generated PNG artifacts are excluded from Git.

## Pegasus CPU/batch sweep

`pegasus_sweep.sbatch` submits the complete CPU-only experiment with one command.
It reserves one node with six CPUs and **8 GiB total job memory**, then runs
Slurm steps sequentially, each bound to exactly two, four, or six CPUs. A named
Pyxis/Enroot container is reused across steps; each configuration gets a fresh
Python process so resident model state is not carried between configurations.

| Parameter | Values |
| --- | --- |
| Model | ENB5-Seg and SegFormer B5 |
| CPUs | 2, 4, 6 |
| RAM | 8 GiB throughout |
| Batch size | 1, 2, 3, 4, 5 |
| Measured runs | 10 per ROI |
| Excluded warm-up | 1 per ROI |
| Crop offset | (0, 0) |

This produces 30 configurations, each covering the small, medium, and large ROIs
(900 measured pipeline runs total). Configurations run sequentially to avoid
interference and to keep the total memory request at 8 GiB. No screen sessions
or separate image builds per configuration are required.

### Prepare the image once

After pulling the repository on Pegasus, change to its root. The checkpoints,
MiT-B5 configuration bundle, and raster listed above are excluded from Git and
must be transferred separately. Store images and large inputs under
`/netscratch/$USER`, rather than the home directory.

Use a scheduled compute node to build the existing Dockerfile for the node's
native architecture; do not use the ARM64 Compose files on Pegasus:

```bash
srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  --time=04:00:00 --immediate=3600 \
  --container-image=/enroot/podman+enroot.sqsh \
  --container-mounts="/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,$PWD:$PWD" \
  --container-workdir="$PWD" --pty bash
```

Inside that build session:

```bash
uname -m
podman build --format=docker -f benchmarks/Dockerfile -t slum-i-benchmark:pegasus-v1 .
enroot import -o /netscratch/$USER/benchmark-v1.sqsh podman://slum-i-benchmark:pegasus-v1
exit
```

`--format=docker` preserves the Micromamba build shell. The sweep bypasses the
Docker entrypoint and activates the environment explicitly with
`micromamba run -n base`. Keep the same `.sqsh` image for all repeats; the
environment YAML has some unpinned dependencies, so rebuilding can change the
resolved environment. Rebuild if you intentionally change the benchmark code;
the image contains the benchmark code used for execution.

### Submit and inspect

From the repository root:

```bash
sbatch benchmarks/pegasus_sweep.sbatch /netscratch/$USER/benchmark-v1.sqsh
squeue -u "$USER"
# Replace JOB_ID with the ID printed by sbatch:
tail -f slurm-slum-i-sweep-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Results are stored under `benchmarks/pegasus_results/JOB_ID/`, with one directory
per configuration, e.g. `enb5_seg_islamabad_cpu2_ram8_b1/`. Each directory contains
the standard benchmark outputs plus CPU affinity and package records. The
top-level directory includes image SHA-256, node/CPU information, filesystem,
allocation details, per-configuration logs, and `status.csv`. Only rows with
`exit_code=0` represent completed configurations; a failed configuration may
leave partial artifacts. Failures are recorded, remaining configurations are
attempted, and the overall job exits unsuccessfully if any configuration failed.

The script stages inputs and timed output writes under the compute node's
`/tmp`, then copies each configuration's results back outside the measured
pipeline. Check the filesystem recorded in `allocation.txt`: a RAM-backed
`/tmp` also consumes memory and should be reported as such. All 30 configurations
use the same staging policy. This measures the pipeline with staged local data,
not shared-filesystem performance. Results are retained after each configuration
so a later timeout does not discard already copied results.

The 8 GiB request includes container startup, staging/page cache, and model
execution; it is not an 8 GiB allowance solely for Python. Pegasus documents
additional RAM requirements for Enroot images. An image or model that cannot
fit must be reported as failing under this budget; the script never increases
RAM automatically. Slurm's actual memory/swap enforcement depends on cluster
configuration, and its CPU binding is different from Docker's CPU time quota.
The runner rejects configurations whose visible CPU affinity count differs
from the requested count instead of silently producing mislabeled results.

The default 24-hour wall-clock limit is a starting point, not a runtime estimate.
All steps stay on the same allocated node. Other jobs may still share the node;
if your allocation permits exclusive-node CPU experiments, use
`sbatch --exclusive ...` with the same script. The memory request remains 8 GiB.
For independent repeats, record the node/CPU model, keep the image and inputs
fixed, and compare variability across jobs. Download results because scratch
storage is not backed up. The named container may remain on the compute node
after completion according to cluster cleanup policy; the script removes its
temporary working directory.

Pegasus references: [software stack](https://pegasus.dfki.de/docs/slurm-cluster/software-stack/),
[building images](https://pegasus.dfki.de/docs/slurm-cluster/custom-software/),
[resource allocation](https://pegasus.dfki.de/docs/slurm-cluster/resource-allocation/).
Runtime references: [Pyxis container reuse](https://github.com/NVIDIA/pyxis),
[Slurm step resource limits](https://slurm.schedmd.com/srun.html),
[Micromamba and Podman](https://micromamba-docker.readthedocs.io/en/stable/quick_start.html).

## Sources of truth

- Protocol, model adapters, metrics, and result files: `system_benchmark.py`
- Container entry point, defaults, and build smoke test: `Dockerfile`
- Per-model runtime configuration: `compose.enb5_seg.yaml`, `compose.segformer.yaml`
- Container dependencies: `environment.yml`
- Raster requirements: `data/README.md`
- Shared portal and benchmark operations: `../ml/segmentation_core.py`
- Pegasus CPU/batch sweep and fixed 8 GiB allocation: `pegasus_sweep.sbatch`
