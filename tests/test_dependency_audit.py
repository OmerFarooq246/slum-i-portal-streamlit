import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from scripts import audit_dependencies


@pytest.mark.parametrize("installed_version", ["4.57.6", "4.57.3", "5.10.0", None])
@pytest.mark.parametrize("gdal_version", ["3.12.3", "3.13.1", None])
def test_exceptions_apply_only_to_reviewed_version_and_preserve_failures(
    installed_version, gdal_version, monkeypatch
) -> None:
    def version(package):
        versions = {"transformers": installed_version, "gdal": gdal_version}
        result = versions[package]
        if result is None:
            raise PackageNotFoundError(package)
        return result

    commands = []

    def run(command, check):
        assert check is False
        commands.append(command)
        # A new advisory, audit service failure, or another vulnerable dependency
        # must still fail CI rather than being hidden by the compatibility policy.
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(audit_dependencies, "version", version)
    monkeypatch.setattr(audit_dependencies.subprocess, "run", run)
    assert audit_dependencies.main() == 1
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "pip_audit"]
    expected = []
    if installed_version == "4.57.6":
        expected.extend(
            [
                "PYSEC-2025-217",
                "PYSEC-2026-2288",
                "PYSEC-2026-2289",
                "PYSEC-2026-2290",
                "PYSEC-2026-3929",
                "PYSEC-2026-4174",
            ]
        )
    if gdal_version == "3.12.3":
        expected.extend(
            [
                "PYSEC-2026-4",
                "PYSEC-2026-2155",
                "PYSEC-2026-2156",
                "PYSEC-2026-193",
                "PYSEC-2026-2154",
                "PYSEC-2026-2153",
                "PYSEC-2026-2157",
            ]
        )
    assert command[3::2] == ["--ignore-vuln"] * len(expected)
    assert command[4::2] == expected
