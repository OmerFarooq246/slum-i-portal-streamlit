"""Reject em dashes in tracked text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which

EM_DASH = "\N{EM DASH}"


def tracked_files() -> list[Path]:
    git = which("git")
    if git is None:
        raise RuntimeError("git executable not found")

    result = subprocess.run(
        [git, "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(name) for name in result.stdout.decode().split("\0") if name]


def em_dash_locations(path: Path) -> list[int]:
    if not path.is_file():
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    return [number for number, line in enumerate(text.splitlines(), start=1) if EM_DASH in line]


def main() -> int:
    paths = [Path(argument) for argument in sys.argv[1:]] or tracked_files()
    failures = [(path, line_number) for path in paths for line_number in em_dash_locations(path)]

    for path, line_number in failures:
        print(f"{path}:{line_number}: em dash is not allowed", file=sys.stderr)

    if failures:
        print(
            "Replace each em dash with a hyphen, comma, colon, parentheses, or rewritten text.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
