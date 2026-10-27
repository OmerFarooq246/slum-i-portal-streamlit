#!/bin/bash
#SBATCH --job-name=slum-i-sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --gpus=0
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root:
# sbatch benchmarks/pegasus_sweep.sbatch ["/absolute/image.sqsh"]
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: sbatch benchmarks/pegasus_sweep.sbatch [/absolute/image.sqsh]" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO="$SLURM_SUBMIT_DIR"
IMAGE="${1:-$REPO/benchmark-current.sqsh}"
[[ "$IMAGE" = /* && -f "$IMAGE" ]] || {
    echo "Image must be an existing absolute .sqsh path: $IMAGE" >&2
    exit 2
}
# Resolve the default once so later successful builds cannot change this sweep.
IMAGE=$(realpath "$IMAGE")
for input in models/enb5_seg_islamabad.h5 models/segformer_b5_islamabad.h5 \
    models/mit-b5/config.json models/mit-b5/preprocessor_config.json \
    benchmarks/data/raster_benchmark.tif; do
    [[ -f "$REPO/$input" ]] || { echo "Missing input: $REPO/$input" >&2; exit 2; }
done

RESULTS="$REPO/benchmarks/pegasus_results/$SLURM_JOB_ID"
mkdir -p "$RESULTS"
CONTAINER="slum-i-sweep-$SLURM_JOB_ID"
WORK=$(mktemp -d /tmp/slum-i-sweep.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/models" "$WORK/benchmarks/data" "$WORK/benchmarks/pegasus_results"
cp "$REPO/models/enb5_seg_islamabad.h5" "$REPO/models/segformer_b5_islamabad.h5" "$WORK/models/"
cp -r "$REPO/models/mit-b5" "$WORK/models/mit-b5"
cp "$REPO/benchmarks/data/raster_benchmark.tif" "$WORK/benchmarks/data/"

{
    printf 'pipelines=streaming,in_memory\n'
    printf 'image=%s\n' "$IMAGE"
    date -u
    hostname
    lscpu
    df -T "$WORK"
    scontrol show job "$SLURM_JOB_ID"
    sha256sum "$IMAGE"
    git -C "$REPO" rev-parse HEAD || true
    git -C "$REPO" status --short || true
} > "$RESULTS/allocation.txt"
cp "$REPO/benchmarks/pegasus_sweep.sbatch" "$RESULTS/"
for pipeline in streaming in_memory; do
    mkdir -p "$RESULTS/$pipeline"
    printf 'model,pipeline,cpus,ram_gib,batch_size,measured_runs,exit_code\n' > "$RESULTS/$pipeline/status.csv"
done

failed=0
for cpus in 2 4 6; do
    for batch in 1 2 3 4 5; do
        for model in enb5_seg_islamabad segformer_b5_islamabad; do
            # Pair both pipelines on the same CPU allocation and input staging.
            for PIPELINE in streaming in_memory; do
                name="${model}_${PIPELINE}_cpu${cpus}_ram16_b${batch}"
                echo "Starting $name: 5 measurements per ROI"
                rc=0
                srun --nodes=1 --ntasks=1 --exact --cpus-per-task="$cpus" \
                    --hint=nomultithread --cpu-bind=verbose,threads --mem=16G --gpus=0 \
                    --container-image="$IMAGE" --container-name="$CONTAINER" \
                    --no-container-entrypoint --container-workdir=/work \
                    --container-mounts="$WORK:/work,$RESULTS:/work/benchmarks/pegasus_results" \
                    micromamba run -n base bash -c '
                        set -euo pipefail
                        cpus=$1; batch=$2; model=$3; name=$4; pipeline=$5
                        export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=/app PYTHONUNBUFFERED=1
                        export OMP_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS="$cpus"
                        export MKL_NUM_THREADS="$cpus" NUMEXPR_NUM_THREADS="$cpus"
                        export TF_NUM_INTRAOP_THREADS="$cpus" TF_NUM_INTEROP_THREADS=1
                        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
                        export PYTHONHASHSEED=0
                        out="benchmarks/pegasus_results/$pipeline/$name"
                        mkdir -p "$out"
                        python -c "import os, sys; a=sorted(os.sched_getaffinity(0)); print(a); sys.exit(0 if len(a)==int(sys.argv[1]) else 2)" \
                            "$cpus" > "$out/cpu-affinity.txt"
                        micromamba list -n base > "$out/packages.txt"
                        python /app/benchmarks/system_benchmark.py \
                            --model "$model" --cpu-limit "$cpus" --batch-size "$batch" \
                            --pipeline "$pipeline" \
                            --warmup-runs 3 --measured-runs 5 --crop-x 0 --crop-y 0 \
                            --results-dir "$out" --work-dir "artifacts/$name"
                    ' sweep "$cpus" "$batch" "$model" "$name" "$PIPELINE" \
                    > "$RESULTS/$PIPELINE/$name.log" 2>&1 || rc=$?

                # Raw measurements already live on shared storage. Preserve inspection
                # artifacts after each step, then release node-local disk space.
                if [[ -d "$WORK/artifacts/$name" ]]; then
                    cp -r "$WORK/artifacts/$name" "$RESULTS/$PIPELINE/$name/artifacts"
                    rm -rf "$WORK/artifacts/$name"
                fi
                printf '%s,%s,%s,16,%s,5,%s\n' "$model" "$PIPELINE" "$cpus" "$batch" "$rc" >> "$RESULTS/$PIPELINE/status.csv"
                if (( rc != 0 )); then
                    echo "FAILED $name (exit $rc); see $RESULTS/$PIPELINE/$name.log" >&2
                    failed=1
                fi
            done
        done
    done
done
echo "Sweep finished. Results: $RESULTS"
exit "$failed"
