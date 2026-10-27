"""Exercise the sweep orchestration without Slurm, containers, or model inference."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SWEEP = Path(__file__).resolve().parents[1] / "benchmarks/pegasus_sweep.sbatch"


def write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


@pytest.fixture
def sweep_environment(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "checkout with spaces"
    for name in (
        "models/enb5_seg_islamabad.h5",
        "models/segformer_b5_islamabad.h5",
        "models/mit-b5/config.json",
        "models/mit-b5/preprocessor_config.json",
        "benchmarks/data/raster_benchmark.tif",
        "benchmarks/pegasus_sweep.sbatch",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    image = tmp_path / "benchmark image.sqsh"
    image.touch()
    commands = tmp_path / "bin"
    commands.mkdir()
    write_executable(
        commands / "srun",
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "mount = next(a.split('=', 1)[1] for a in args "
        "if a.startswith('--container-mounts='))\n"
        "work, durable = mount.split(',')\n"
        "work = work.removesuffix(':/work')\n"
        "source = durable.removesuffix(':/work/benchmarks/pegasus_results')\n"
        "target = os.path.join(work, 'benchmarks/pegasus_results')\n"
        "if not os.path.islink(target):\n"
        "    os.rmdir(target)\n"
        "    os.symlink(source, target)\n"
        "assert '--mem=16G' in args and '--gpus=0' in args\n"
        "sys.exit(subprocess.run(args[args.index('micromamba'):], cwd=work).returncode)\n",
    )
    write_executable(
        commands / "micromamba",
        '#!/bin/bash\nif [[ "$1" == list ]]; then echo mock-packages; else exec "${@:4}"; fi\n',
    )
    write_executable(
        commands / "python",
        f"#!{sys.executable}\n"
        "import argparse, json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c':\n"
        "    print(list(range(int(sys.argv[3]))))\n"
        "    sys.exit(0)\n"
        "assert sys.argv[1] == '/app/benchmarks/system_benchmark.py'\n"
        "parser = argparse.ArgumentParser()\n"
        "for name in ('model', 'cpu-limit', 'batch-size', 'pipeline', 'warmup-runs', "
        "'measured-runs', 'crop-x', 'crop-y', 'results-dir', 'work-dir'):\n"
        "    parser.add_argument('--' + name, required=True)\n"
        "args = parser.parse_args(sys.argv[2:])\n"
        "out = Path(args.results_dir)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'metadata.json').write_text(json.dumps(vars(args)))\n"
        "artifacts = Path(args.work_dir)\n"
        "artifacts.mkdir(parents=True, exist_ok=True)\n"
        "(artifacts / 'mask.tif').write_text('mock-mask')\n"
        "case = f'{args.model}:{args.cpu_limit}:{args.batch_size}'\n"
        "sys.exit(7 if case == os.environ.get('FAIL_CASE') else 0)\n",
    )
    for command in ("lscpu", "scontrol", "sha256sum"):
        write_executable(commands / command, "#!/bin/bash\necho mock-allocation\n")
    env = dict(os.environ)
    env.update(
        PATH=f"{commands}{os.pathsep}{env['PATH']}",
        SLURM_JOB_ID="4242",
        SLURM_SUBMIT_DIR=str(repo),
    )
    return image, env


def test_sweep_pairs_pipelines_and_preserves_matrix(
    sweep_environment: tuple[Path, dict[str, str]],
) -> None:
    image, env = sweep_environment
    result = subprocess.run(
        ["bash", str(SWEEP), str(image)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    starts = [
        line.split()[1] for line in result.stdout.splitlines() if line.startswith("Starting ")
    ]
    assert len(starts) == 60
    for streaming, baseline in zip(starts[::2], starts[1::2], strict=True):
        assert streaming.replace("_streaming_", "_in_memory_") == baseline
    for pipeline in ("streaming", "in_memory"):
        root = Path(env["SLURM_SUBMIT_DIR"]) / "benchmarks/pegasus_results/4242" / pipeline
        with (root / "status.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            (model, str(cpus), str(batch))
            for model in ("enb5_seg_islamabad", "segformer_b5_islamabad")
            for cpus in (2, 4, 6)
            for batch in range(1, 6)
        }
        assert len(rows) == 30
        assert {(r["model"], r["cpus"], r["batch_size"]) for r in rows} == expected
        for row in rows:
            assert row["pipeline"] == pipeline
            assert row["exit_code"] == "0"
            assert row["ram_gib"] == "16"
            name = f"{row['model']}_{pipeline}_cpu{row['cpus']}_ram16_b{row['batch_size']}"
            metadata = json.loads((root / name / "metadata.json").read_text())
            assert metadata["pipeline"] == pipeline
            assert metadata["model"] == row["model"]
            assert metadata["work_dir"] == f"artifacts/{name}"
            assert (root / name / "artifacts/mask.tif").read_text() == "mock-mask"
            assert metadata["cpu_limit"] == row["cpus"]
            assert metadata["batch_size"] == row["batch_size"]
            assert metadata["warmup_runs"] == "3"
            assert metadata["measured_runs"] == row["measured_runs"] == "5"
            assert (root / f"{name}.log").is_file()


def test_sweep_continues_after_failed_configuration(
    sweep_environment: tuple[Path, dict[str, str]],
) -> None:
    image, env = sweep_environment
    env["FAIL_CASE"] = "enb5_seg_islamabad:2:1"
    result = subprocess.run(
        ["bash", str(SWEEP), str(image)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 1, result.stdout + result.stderr
    root = Path(env["SLURM_SUBMIT_DIR"]) / "benchmarks/pegasus_results/4242/streaming"
    with (root / "status.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert rows[0]["exit_code"] == "7"
    assert all(row["exit_code"] == "0" for row in rows[1:])
    assert (root / "enb5_seg_islamabad_streaming_cpu2_ram16_b1/metadata.json").is_file()


@pytest.mark.parametrize("image_args", [[], ["image", "streaming"]])
def test_sweep_requires_only_image_argument(
    sweep_environment: tuple[Path, dict[str, str]], image_args: list[str]
) -> None:
    _, env = sweep_environment
    result = subprocess.run(
        ["bash", str(SWEEP), *image_args], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not (Path(env["SLURM_SUBMIT_DIR"]) / "benchmarks/pegasus_results").exists()
