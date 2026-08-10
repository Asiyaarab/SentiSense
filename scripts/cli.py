"""Tiny CLI wrapper so `sentisense-train` works after `pip install -e .`.

Thin shim over the root `train.py` for users who prefer the entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    # Lazy import so help is fast.
    import runpy

    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
