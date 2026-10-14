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

Transformers 4.57.6 uses the legacy TensorFlow Keras API for this model, so the benchmark environment pins `tf-keras==2.20.1` alongside `tensorflow==2.20.0`.

Both Compose configurations supply their model and runtime arguments explicitly. Do not run the ARM64 protocol through AMD64 emulation.

The runner defaults to three excluded warm-up runs and five measured runs per ROI, and accepts at least one warm-up and five measurements. The local ENB5 Compose configuration explicitly uses one warm-up and five measurements to match the historical baseline; the Pegasus sweep explicitly uses three warm-ups and ten measurements. Summary CSV columns use `_mean` and `_stddev`, and paper tables report arithmetic mean ± sample standard deviation across the measured runs. Historical result sets retain their original measurements and warm-up counts; their summaries can be recalculated from the saved raw runs without rerunning inference.

Ten-run results are stored in `results_10_mruns/`. Five-run results are preserved in `results_5_mruns/` with their original raw measurements and mean-based summaries. The `enb5_seg_islamabad_cpu2_b1_new` and `enb5_seg_islamabad_cpu2_b5_new` directories record graph-inference runs for comparison with the corresponding `enb5_seg_islamabad_b1` and `enb5_seg_islamabad_b5` baselines. Their metadata matches except for CPU affinity (six available CPUs versus four previously, with the same two-CPU quota), which limits attribution of the observed gains solely to graph execution. Generated PNG artifacts are excluded from Git.

## Pegasus CPU/batch sweep

`pegasus_sweep.sbatch` runs both models across CPUs **2, 4, 6** and batch sizes
**1-5**, with **8 GiB RAM**, **3 excluded warm-ups**, and **10 measured runs per
ROI**. The 30 configurations run sequentially on one node, reusing one container
with a fresh Python process per configuration.

### Build once

From the repository root on Pegasus, provide the model files and raster listed
above, then start a compute-node build session:

```bash
srun --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  --time=04:00:00 --immediate=3600 \
  --container-image=/enroot/podman+enroot.sqsh \
  --container-mounts="/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,$PWD:$PWD" \
  --container-workdir="$PWD" --pty bash
```

Inside the session, build and save the image:

```bash
podman build --format=docker -f benchmarks/Dockerfile -t slum-i-benchmark:pegasus-v1 .
enroot import -o /netscratch/$USER/benchmark-v1.sqsh podman://slum-i-benchmark:pegasus-v1
exit
```

Reuse this image for repeat experiments; rebuild after changing benchmark code.

### Run the sweep

Submit from the repository root, replacing `JOB_ID` with the returned ID:

```bash
sbatch benchmarks/pegasus_sweep.sbatch /netscratch/$USER/benchmark-v1.sqsh
squeue -u "$USER"
tail -f slurm-slum-i-sweep-JOB_ID.out
```

Results and logs are saved in `benchmarks/pegasus_results/JOB_ID/`, with a separate
directory per configuration. Check `status.csv` for completion (`exit_code=0`);
failed configurations are recorded and the remaining sweep continues.

Inputs and timed outputs are staged under node-local `/tmp`. The 8 GiB budget
includes container and staging overhead; RAM is never increased automatically.
Keep the image and inputs fixed across repeats, and copy results off scratch
storage for safekeeping.

## Sources of truth

- Protocol, model adapters, metrics, and result files: `system_benchmark.py`
- Container entry point, defaults, and build smoke test: `Dockerfile`
- Per-model runtime configuration: `compose.enb5_seg.yaml`, `compose.segformer.yaml`
- Container dependencies: `environment.yml`
- Raster requirements: `data/README.md`
- Shared portal and benchmark operations: `../ml/segmentation_core.py`
- Pegasus CPU/batch sweep and fixed 8 GiB allocation: `pegasus_sweep.sbatch`
