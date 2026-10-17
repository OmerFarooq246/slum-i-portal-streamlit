"""Audit dependencies with documented, version-specific advisory exceptions."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# Rationale, accepted risks, and removal conditions: docs/dependency-audit.md.
TRANSFORMERS_VERSION = "4.57.6"
TRANSFORMERS_ADVISORIES = (
    "PYSEC-2025-217",
    "PYSEC-2026-2288",
    "PYSEC-2026-2289",
    "PYSEC-2026-2290",
    "PYSEC-2026-3929",
    "PYSEC-2026-4174",
)
GDAL_VERSION = "3.12.3"
GDAL_ADVISORIES = (
    "PYSEC-2026-4",
    "PYSEC-2026-2155",
    "PYSEC-2026-2156",
    "PYSEC-2026-193",
    "PYSEC-2026-2154",
    "PYSEC-2026-2153",
    "PYSEC-2026-2157",
)


def main() -> int:
    command = [sys.executable, "-m", "pip_audit"]
    for package, reviewed_version, advisories in (
        ("transformers", TRANSFORMERS_VERSION, TRANSFORMERS_ADVISORIES),
        ("gdal", GDAL_VERSION, GDAL_ADVISORIES),
    ):
        try:
            installed_version = version(package)
        except PackageNotFoundError:
            continue
        if installed_version != reviewed_version:
            continue
        print(
            f"Applying documented {package} {reviewed_version} advisory exceptions; "
            "see docs/dependency-audit.md. Other findings remain blocking.",
            flush=True,
        )
        for advisory in advisories:
            command.extend(("--ignore-vuln", advisory))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
