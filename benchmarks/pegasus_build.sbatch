#!/bin/bash
#SBATCH --job-name=slum-i-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root: sbatch benchmarks/pegasus_build.sbatch
set -euo pipefail
if (( $# != 0 )); then
    echo "Usage: sbatch benchmarks/pegasus_build.sbatch" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"
for input in benchmarks/Dockerfile benchmarks/environment.yml .dockerignore; do
    [[ -f "$input" ]] || { echo "Missing build input: $input" >&2; exit 2; }
done
IMAGE="$REPO/benchmark-$SLURM_JOB_ID.sqsh"
BUILD_ROOT="$REPO/.pegasus-build/$SLURM_JOB_ID"
CURRENT="$REPO/benchmark-current.sqsh"
[[ ! -e "$IMAGE" && ! -L "$IMAGE" ]] || {
    echo "Refusing to replace existing image: $IMAGE" >&2
    exit 2
}

[[ ! -d "$CURRENT" && ( ! -e "$CURRENT" || -L "$CURRENT" ) ]] || {
    echo "Default image path must be a symlink or unused path: $CURRENT" >&2
    exit 2
}

echo "Building benchmark image on $(hostname) for job $SLURM_JOB_ID"
git rev-parse HEAD || true
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gpus=0 \
    --container-image=/enroot/podman+enroot.sqsh \
    --container-mounts="/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,$REPO:$REPO" \
    --container-workdir="$REPO" --no-container-entrypoint \
    bash -s -- "$BUILD_ROOT" "$SLURM_JOB_ID" <<'BUILD'
set -euo pipefail
BUILD_ROOT="$1"
TAG="slum-i-benchmark:pegasus-$2"
mkdir -p "$BUILD_ROOT"/{storage,runroot,tmp,cache,enroot-data,enroot-cache}
export TMPDIR="$BUILD_ROOT/tmp"
export XDG_CACHE_HOME="$BUILD_ROOT/cache"
export ENROOT_CACHE_PATH="$BUILD_ROOT/enroot-cache"
export ENROOT_DATA_PATH="$BUILD_ROOT/enroot-data"
export ENROOT_TEMP_PATH="$TMPDIR"
export CONTAINERS_STORAGE_CONF="$BUILD_ROOT/storage.conf"
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "vfs"
graphroot = "$BUILD_ROOT/storage"
runroot = "$BUILD_ROOT/runroot"
EOF
# vfs avoids overlay requirements but uses more scratch space.
podman --tmpdir="$TMPDIR" build --format=docker \
    -f benchmarks/Dockerfile -t "$TAG" .
enroot import -o "$BUILD_ROOT/image.sqsh" "podman://$TAG"
BUILD

# Publish only a completed, non-empty export. Hard linking refuses overwrites.
[[ -s "$BUILD_ROOT/image.sqsh" ]] || { echo "Exported image is empty or missing" >&2; exit 1; }
ln "$BUILD_ROOT/image.sqsh" "$IMAGE"
rm "$BUILD_ROOT/image.sqsh"
# Rename a temporary link atomically, leaving the previous default on failure.
ln -s "$(basename "$IMAGE")" "$BUILD_ROOT/current.sqsh"
mv -f "$BUILD_ROOT/current.sqsh" "$CURRENT"
echo "Build finished. Image: $IMAGE"
echo "Default image: $CURRENT"
printf 'After the cluster smoke test, submit from the repository root:\n'
printf 'sbatch benchmarks/pegasus_sweep.sbatch\n'
