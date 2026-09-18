"""Prevent direct pushes to the main branch."""

from __future__ import annotations

import os
import sys

MAIN_REF = "refs/heads/main"


def main() -> int:
    remote_branch = os.environ.get("PRE_COMMIT_REMOTE_BRANCH")
    if remote_branch != MAIN_REF:
        return 0

    print(
        "Direct pushes to main are not allowed. Create a branch and open a pull request.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
