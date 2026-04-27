"""
Regenerate ``tests/golden_hashes.json`` from the current ``env.py``.

This script is deliberately separate from the test file so the test never
overwrites its own reference. Run it only when the simulator's stationary
behaviour has been intentionally changed (very rare — almost always when
fixing a bug in the base simulator). Always review the diff before committing
the new ``golden_hashes.json``.

Usage::

    python tests/regenerate_golden_hashes.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the same helpers used by the test, to guarantee the digest scheme
# stays identical.
from tests.test_byte_identical import (  # noqa: E402
    CONFIGS,
    SEEDS,
    N_STEPS,
    _stable_summary,
    _digest,
    GOLDEN_PATH,
)


def main():
    new_golden = {}
    for testbed, cfg_path in CONFIGS:
        for seed in SEEDS:
            key = f"{testbed}_seed{seed}"
            summary = _stable_summary(cfg_path, seed, N_STEPS)
            new_golden[key] = _digest(summary)
            print(f"  {key}: {new_golden[key][:24]}...")

    GOLDEN_PATH.write_text(json.dumps(new_golden, indent=2) + "\n")
    print(f"\nWrote {len(new_golden)} golden hashes to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
