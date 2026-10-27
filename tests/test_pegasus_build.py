"""Check batch build execution and publication without Slurm or container tools."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "benchmarks/pegasus_build.sbatch"


@pytest.fixture
def build_environment(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "checkout with spaces"
    for name in ("benchmarks/Dockerfile", "benchmarks/environment.yml", ".dockerignore"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    commands = tmp_path / "bin"
    commands.mkdir()
    scripts = {
        "srun": (
            f"#!{sys.executable}\n"
            "import subprocess, sys\n"
            "args = sys.argv[1:]\n"
            "assert '--mem=16G' in args and '--gpus=0' in args\n"
            "assert '--pty' not in args\n"
            "sys.exit(subprocess.run(args[args.index('bash'):]).returncode)\n"
        ),
        "podman": (
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "root = Path(os.environ['SLURM_SUBMIT_DIR']) / '.pegasus-build/1234'\n"
            "for key in ('TMPDIR', 'XDG_CACHE_HOME', 'ENROOT_CACHE_PATH', "
            "'ENROOT_DATA_PATH', 'ENROOT_TEMP_PATH', 'CONTAINERS_STORAGE_CONF'):\n"
            "    assert Path(os.environ[key]).is_relative_to(root)\n"
            "config = Path(os.environ['CONTAINERS_STORAGE_CONF']).read_text()\n"
            "assert f'graphroot = \"{root}/storage\"' in config\n"
            "assert f'runroot = \"{root}/runroot\"' in config\n"
            "assert 'build' in sys.argv and 'slum-i-benchmark:pegasus-1234' in sys.argv\n"
            "(root / 'podman-called').touch()\n"
            "sys.exit(7 if os.environ.get('FAIL_BUILD') else 0)\n"
        ),
        "enroot": (
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "assert sys.argv[1] == 'import'\n"
            "assert sys.argv[-1] == 'podman://slum-i-benchmark:pegasus-1234'\n"
            "output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "output.write_text('partial' if os.environ.get('FAIL_EXPORT') else 'image')\n"
            "sys.exit(8 if os.environ.get('FAIL_EXPORT') else 0)\n"
        ),
    }
    for name, source in scripts.items():
        path = commands / name
        path.write_text(source)
        path.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{commands}{os.pathsep}{env['PATH']}",
        SLURM_JOB_ID="1234",
        SLURM_SUBMIT_DIR=str(repo),
    )
    return repo, env


def test_build_publishes_image_after_export(
    build_environment: tuple[Path, dict[str, str]],
) -> None:
    repo, env = build_environment
    result = subprocess.run(["bash", str(BUILD)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    image = repo / "benchmark-1234.sqsh"
    assert image.read_text() == "image"
    assert not (repo / ".pegasus-build/1234/image.sqsh").exists()
    assert f'sbatch benchmarks/pegasus_sweep.sbatch "{image}"' in result.stdout


@pytest.mark.parametrize(("failure", "exit_code"), [("FAIL_BUILD", 7), ("FAIL_EXPORT", 8)])
def test_failed_build_does_not_publish_image(
    build_environment: tuple[Path, dict[str, str]], failure: str, exit_code: int
) -> None:
    repo, env = build_environment
    env[failure] = "1"
    result = subprocess.run(["bash", str(BUILD)], env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert not (repo / "benchmark-1234.sqsh").exists()
    if failure == "FAIL_BUILD":
        assert not (repo / ".pegasus-build/1234/image.sqsh").exists()
    else:
        assert (repo / ".pegasus-build/1234/image.sqsh").read_text() == "partial"


def test_build_refuses_to_overwrite_existing_image(
    build_environment: tuple[Path, dict[str, str]],
) -> None:
    repo, env = build_environment
    image = repo / "benchmark-1234.sqsh"
    image.write_text("existing image")
    result = subprocess.run(["bash", str(BUILD)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Refusing to replace" in result.stderr
    assert image.read_text() == "existing image"
    assert not (repo / ".pegasus-build").exists()
