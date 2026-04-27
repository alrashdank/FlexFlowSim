"""
Regression test: byte-identical reproduction of Paper 3/4 simulator output.

The non-stationarity extensions added to FlexFlowSim for Paper 5 (machine
breakdowns, demand surges, processing-cost drift) are implemented as opt-in
extensions that take a separate code path from the base simulator. Stationary
configurations — those without breakdowns, surge schedules, or cost-drift
schedules enabled — must therefore produce traces that are byte-identical to
the unmodified Paper 3/4 simulator.

This test verifies that property against a 20-seed golden reference. The
golden hashes were computed once from the unpatched (Paper 3/4) env.py at
the time the breakdown extension was introduced, and are stored in
``golden_hashes.json``.

Running
-------
    cd /path/to/FlexFlowSim
    pytest tests/test_byte_identical.py -v

The test passes if every (testbed, seed) cell produces the same SHA-256 digest
as the golden reference. A failure indicates that a recent change to ``env.py``
has altered behaviour for stationary configurations and must be investigated
before being committed.

Regenerating the golden reference
---------------------------------
If a change to the simulator is *intentionally* expected to alter stationary
behaviour (rare — most commonly when fixing a bug in the base simulator),
re-run::

    python tests/regenerate_golden_hashes.py

and review the diff carefully before committing the new ``golden_hashes.json``.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable so ``from env import FlexFlowSimEnv`` works.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from env import FlexFlowSimEnv  # noqa: E402

# ----------------------------------------------------------------------
# Test parameters — must match the regenerator script
# ----------------------------------------------------------------------
CONFIGS = [
    ("bakery",      REPO_ROOT / "configs" / "bakery_bk50.json"),
    ("electronics", REPO_ROOT / "configs" / "electronics_3stage.json"),
]
SEEDS = list(range(20))
N_STEPS = 480
GOLDEN_PATH = Path(__file__).parent / "golden_hashes.json"


def _stable_summary(config_path: Path, seed: int, n_steps: int) -> str:
    """Run one episode and return a stable text summary suitable for hashing.

    The summary must not depend on Python or NumPy versions: floats are
    formatted with %.10g, and array contents are joined with commas.
    """
    cfg = json.loads(config_path.read_text())
    env = FlexFlowSimEnv(cfg, seed=seed)
    init_obs, _ = env.reset(seed=seed)

    # Action sequence is deterministic given the seed.
    rng = np.random.default_rng(seed * 7919)
    actions = rng.integers(0, env.action_space.n, size=n_steps).tolist()

    rewards = []
    last_obs = init_obs
    last_info = {}
    for a in actions:
        result = env.step(int(a))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        rewards.append(float(reward))
        last_obs = obs
        last_info = info
        if done:
            break

    def _fmt_array(a):
        return ",".join(f"{float(x):.10g}" for x in a)

    departed = last_info.get("total_departed", last_info.get("totalDeparted", "NA"))
    total_cost = float(last_info.get("total_cost", last_info.get("totalCost", 0.0)))

    parts = [
        f"init_obs:{_fmt_array(init_obs)}",
        f"final_obs:{_fmt_array(last_obs)}",
        f"n_steps:{len(rewards)}",
        f"reward_sum:{sum(rewards):.10g}",
        f"reward_first10:{_fmt_array(rewards[:10])}",
        f"reward_last10:{_fmt_array(rewards[-10:])}",
        f"total_departed:{departed}",
        f"total_cost:{total_cost:.6g}",
    ]
    return "|".join(parts)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
GOLDEN = json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize(
    "testbed,config_path,seed",
    [(tb, cfg, s) for tb, cfg in CONFIGS for s in SEEDS],
    ids=[f"{tb}_seed{s}" for tb, _ in CONFIGS for s in SEEDS],
)
def test_stationary_byte_identical(testbed, config_path, seed):
    """Stationary trace for ``(testbed, seed)`` must match the golden hash."""
    key = f"{testbed}_seed{seed}"
    expected = GOLDEN.get(key)
    assert expected is not None, f"No golden hash recorded for {key}"
    summary = _stable_summary(config_path, seed, N_STEPS)
    actual = _digest(summary)
    assert actual == expected, (
        f"Trace for {key} has diverged from the Paper 3/4 reference.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"This means a recent change to env.py has altered stationary behaviour.\n"
        f"If the change is intentional, regenerate the golden reference with:\n"
        f"    python tests/regenerate_golden_hashes.py"
    )


def test_golden_reference_complete():
    """Sanity check that the golden file covers every (testbed, seed) cell."""
    expected_keys = {f"{tb}_seed{s}" for tb, _ in CONFIGS for s in SEEDS}
    actual_keys = set(GOLDEN.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing, f"Golden reference missing keys: {sorted(missing)}"
    assert not extra, f"Golden reference has stray keys: {sorted(extra)}"


def test_breakdowns_change_behaviour():
    """Negative test: enabling breakdowns must change the trace.

    Protects against a regression where the breakdown code path silently
    no-ops (e.g. a refactor accidentally drops the ``breakdowns`` key from
    the config). For at least one severity, the breakdown variant must
    produce a different hash from the stationary baseline.
    """
    base_path = REPO_ROOT / "configs" / "electronics_3stage.json"
    bd_path = REPO_ROOT / "configs" / "paper5_electronics" / "electronics_breakdowns_A70.json"
    assert base_path.exists() and bd_path.exists(), \
        "Test requires both stationary and breakdown configs"

    seed = 0
    base_summary = _stable_summary(base_path, seed, N_STEPS)
    bd_summary = _stable_summary(bd_path, seed, N_STEPS)
    assert base_summary != bd_summary, (
        "Stationary and A=0.70 breakdown traces are identical — "
        "the breakdown mechanism is not active when it should be."
    )
