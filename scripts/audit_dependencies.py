"""Audit dependencies with advisory-specific exceptions for TensorFlow SegFormer."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# Reviewed unused features only; rationale and removal conditions: docs/dependency-audit.md.
TRANSFORMERS_VERSION = "4.57.6"
TRANSFORMERS_ADVISORIES = (
    "PYSEC-2025-217",
    "PYSEC-2026-2288",
    "PYSEC-2026-2289",
    "PYSEC-2026-2290",
    "PYSEC-2026-3929",
    "PYSEC-2026-4174",
)


def main() -> int:
    command = [sys.executable, "-m", "pip_audit"]
    try:
        installed_version = version("transformers")
    except PackageNotFoundError:
        installed_version = None
    if installed_version == TRANSFORMERS_VERSION:
        print(
            "Applying reviewed Transformers 4.57.6 advisory exceptions; "
            "see docs/dependency-audit.md. Other findings remain blocking.",
            flush=True,
        )
        for advisory in TRANSFORMERS_ADVISORIES:
            command.extend(("--ignore-vuln", advisory))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
