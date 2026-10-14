import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from scripts import audit_dependencies


@pytest.mark.parametrize("installed_version", ["4.57.6", "4.57.3", "5.10.0", None])
def test_exceptions_apply_only_to_reviewed_version_and_preserve_failures(
    installed_version, monkeypatch
) -> None:
    def version(package):
        assert package == "transformers"
        if installed_version is None:
            raise PackageNotFoundError(package)
        return installed_version

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
    if installed_version == "4.57.6":
        assert command[3::2] == ["--ignore-vuln"] * 6
        assert command[4::2] == [
            "PYSEC-2025-217",
            "PYSEC-2026-2288",
            "PYSEC-2026-2289",
            "PYSEC-2026-2290",
            "PYSEC-2026-3929",
            "PYSEC-2026-4174",
        ]
    else:
        assert command[3:] == []
