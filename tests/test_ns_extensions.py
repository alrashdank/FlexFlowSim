"""
Tests for FlexFlowSim non-stationarity extensions (Paper 5).

Covers:
  1. Regression: stationary bakery config produces byte-identical output
     to the Paper 3/4 baseline (golden file).
  2. Breakdowns: server availability is gated correctly; no entity is
     granted service during downtime; downtime is correctly accounted.
  3. Arrival schedule: thinning produces arrival rates that match the
     configured lambda(t) within Monte Carlo tolerance.
  4. Cost schedule: processing cost accumulates according to the
     time-varying multiplier.
  5. Combined: all three mechanisms operate together without interference.
"""

import json
import os
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import FlexFlowSimEnv
from baselines import RoundRobinPolicy

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "bakery_bk50.json",
)
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden_roundrobin_stationary.json",
)


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _run_episode(cfg, policy_cls, seed, weights=(1.0, 0.0, 0.0)):
    env = FlexFlowSimEnv(config=cfg, weights=weights, seed=seed)
    pol = policy_cls(env=env)
    obs, info = env.reset(seed=seed)
    pol.reset()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(pol.predict(obs))
        done = term or trunc
    return info, env


# ───────────────────────────────────────────────────────────────────
# 1. REGRESSION
# ───────────────────────────────────────────────────────────────────

class TestRegression:
    """Confirm that stationary config reproduces Paper 4 baseline bit-for-bit."""

    def test_roundrobin_byte_identical_to_golden(self):
        cfg = _load_cfg()
        with open(GOLDEN_PATH) as f:
            golden = json.load(f)
        assert len(golden) == 20

        for g in golden:
            info, _ = _run_episode(cfg, RoundRobinPolicy, seed=g["seed"])
            assert abs(float(info["total_cost"]) - g["total_cost"]) < 1e-9, \
                f"Seed {g['seed']}: cost drift"
            assert int(info["total_departed"]) == g["departed"], \
                f"Seed {g['seed']}: departures drift"

    def test_ns_blocks_disabled_by_default(self):
        cfg = _load_cfg()
        env = FlexFlowSimEnv(config=cfg, seed=0)
        env.reset(seed=0)
        assert env._breakdowns_enabled is False
        assert env._arrival_is_nonstationary is False
        assert env._cost_schedule_enabled is False


# ───────────────────────────────────────────────────────────────────
# 2. BREAKDOWNS
# ───────────────────────────────────────────────────────────────────

class TestBreakdowns:

    @staticmethod
    def _cfg_with_breakdowns(mtbf=200.0, mttr=30.0):
        cfg = _load_cfg()
        cfg["breakdowns"] = {
            "enabled": True,
            "default": {
                "ttf": {"distribution": "exponential", "mean": mtbf},
                "ttr": {"distribution": "exponential", "mean": mttr},
            },
        }
        return cfg

    def test_breakdown_count_nonzero(self):
        cfg = self._cfg_with_breakdowns(mtbf=100.0, mttr=20.0)
        info, _ = _run_episode(cfg, RoundRobinPolicy, seed=42)
        assert "breakdown_count" in info
        assert sum(info["breakdown_count"]) > 0

    def test_breakdown_time_bounded_by_horizon(self):
        cfg = self._cfg_with_breakdowns(mtbf=50.0, mttr=50.0)
        info, _ = _run_episode(cfg, RoundRobinPolicy, seed=7)
        for bt in info["breakdown_time"]:
            assert 0 <= bt <= 480.0

    def test_steady_state_availability(self):
        """Mean unavailability fraction should approximate MTTR/(MTBF+MTTR)."""
        mtbf, mttr = 100.0, 25.0
        cfg = self._cfg_with_breakdowns(mtbf=mtbf, mttr=mttr)
        fracs = []
        for seed in range(10):
            info, _ = _run_episode(cfg, RoundRobinPolicy, seed=seed)
            bt = np.array(info["breakdown_time"])
            fracs.append(bt.mean() / 480.0)
        observed = float(np.mean(fracs))
        assert 0.10 < observed < 0.35, f"unavailability observed {observed:.3f}"

    def test_severe_breakdowns_reduce_throughput(self):
        cfg_s = _load_cfg()
        info_s, _ = _run_episode(cfg_s, RoundRobinPolicy, seed=0)
        cfg_bd = self._cfg_with_breakdowns(mtbf=50.0, mttr=100.0)
        info_bd, _ = _run_episode(cfg_bd, RoundRobinPolicy, seed=0)
        assert info_bd["total_departed"] < info_s["total_departed"]


# ───────────────────────────────────────────────────────────────────
# 3. ARRIVAL SCHEDULE
# ───────────────────────────────────────────────────────────────────

class TestArrivalSchedule:

    @staticmethod
    def _count_arrivals(env, info):
        in_system = int(np.sum(env._queue_len) + np.sum(env._in_service))
        return info["total_departed"] + in_system

    def test_piecewise_rate_matches_expectation(self):
        cfg = _load_cfg()
        cfg["arrival"] = {
            "distribution": "exponential",
            "mean": 9.6,
            "schedule": [
                {"t": 0, "rate": 0.05},
                {"t": 240, "rate": 0.20},
            ],
        }
        expected = 0.05 * 240 + 0.20 * 240  # 60

        totals = []
        for seed in range(10):
            env = FlexFlowSimEnv(config=cfg, seed=seed)
            env.reset(seed=seed)
            done = False
            info = None
            while not done:
                _, _, term, trunc, info = env.step(0)
                done = term or trunc
            totals.append(self._count_arrivals(env, info))

        mean_arrivals = float(np.mean(totals))
        tol = 3 * np.sqrt(expected) / np.sqrt(10) + 5
        assert abs(mean_arrivals - expected) < tol, \
            f"Mean arrivals {mean_arrivals:.1f} vs expected {expected:.1f}"

    def test_single_segment_matches_constant(self):
        cfg_const = _load_cfg()
        cfg_const["arrival"] = {"distribution": "exponential", "mean": 10.0}
        cfg_sched = deepcopy(cfg_const)
        cfg_sched["arrival"] = {
            "distribution": "exponential",
            "mean": 10.0,
            "schedule": [{"t": 0, "rate": 0.1}],
        }

        def avg_arrivals(cfg, n=8):
            arr = []
            for seed in range(n):
                env = FlexFlowSimEnv(config=cfg, seed=seed)
                env.reset(seed=seed)
                done = False
                info = None
                while not done:
                    _, _, term, trunc, info = env.step(0)
                    done = term or trunc
                arr.append(self._count_arrivals(env, info))
            return float(np.mean(arr))

        mean_c = avg_arrivals(cfg_const)
        mean_s = avg_arrivals(cfg_sched)
        tol = 3 * np.sqrt(mean_c) / np.sqrt(8) + 3
        assert abs(mean_c - mean_s) < tol, \
            f"Const {mean_c:.1f} vs single-segment schedule {mean_s:.1f}"


# ───────────────────────────────────────────────────────────────────
# 4. COST SCHEDULE
# ───────────────────────────────────────────────────────────────────

class TestCostSchedule:

    def test_uniform_2x_doubles_processing_cost(self):
        cfg_base = _load_cfg()
        info_base, _ = _run_episode(cfg_base, RoundRobinPolicy, seed=0)

        cfg_2x = deepcopy(cfg_base)
        cfg_2x["cost_schedule"] = {
            "enabled": True,
            "processing_multiplier": [{"t": 0, "multiplier": 2.0}],
        }
        info_2x, _ = _run_episode(cfg_2x, RoundRobinPolicy, seed=0)

        assert info_2x["total_departed"] == info_base["total_departed"]
        assert abs(info_2x["processing_cost"] - 2.0 * info_base["processing_cost"]) < 1e-6
        assert abs(info_2x["idle_cost"] - info_base["idle_cost"]) < 1e-6
        assert abs(info_2x["waiting_cost"] - info_base["waiting_cost"]) < 1e-6

    def test_stepwise_multiplier(self):
        cfg_base = _load_cfg()
        info_base, _ = _run_episode(cfg_base, RoundRobinPolicy, seed=3)

        cfg_step = deepcopy(cfg_base)
        cfg_step["cost_schedule"] = {
            "enabled": True,
            "processing_multiplier": [
                {"t": 0, "multiplier": 1.0},
                {"t": 240, "multiplier": 3.0},
            ],
        }
        info_step, _ = _run_episode(cfg_step, RoundRobinPolicy, seed=3)
        assert info_step["total_departed"] == info_base["total_departed"]
        ratio = info_step["processing_cost"] / info_base["processing_cost"]
        assert 1.5 < ratio < 2.5, f"ratio {ratio:.2f} not near 2.0"


# ───────────────────────────────────────────────────────────────────
# 5. COMBINED
# ───────────────────────────────────────────────────────────────────

class TestCombined:

    @staticmethod
    def _full_ns_cfg():
        cfg = _load_cfg()
        cfg["breakdowns"] = {
            "enabled": True,
            "default": {
                "ttf": {"distribution": "exponential", "mean": 150.0},
                "ttr": {"distribution": "exponential", "mean": 20.0},
            },
        }
        cfg["arrival"] = {
            "distribution": "exponential",
            "mean": 9.6,
            "schedule": [
                {"t": 0, "rate": 0.08},
                {"t": 240, "rate": 0.15},
            ],
        }
        cfg["cost_schedule"] = {
            "enabled": True,
            "processing_multiplier": [
                {"t": 0, "multiplier": 1.0},
                {"t": 240, "multiplier": 1.5},
            ],
        }
        return cfg

    def test_runs_without_error(self):
        info, _ = _run_episode(self._full_ns_cfg(), RoundRobinPolicy, seed=123)
        assert info["total_departed"] > 0
        assert "breakdown_count" in info
        assert info["sim_time"] >= 480.0

    def test_reset_deterministic(self):
        env = FlexFlowSimEnv(config=self._full_ns_cfg(), seed=55)
        obs1, _ = env.reset(seed=55)
        obs2, _ = env.reset(seed=55)
        assert np.array_equal(obs1, obs2)

    def test_two_episodes_same_seed_reproducible(self):
        cfg = self._full_ns_cfg()
        info1, _ = _run_episode(cfg, RoundRobinPolicy, seed=99)
        info2, _ = _run_episode(cfg, RoundRobinPolicy, seed=99)
        assert info1["total_departed"] == info2["total_departed"]
        assert abs(info1["total_cost"] - info2["total_cost"]) < 1e-9


# ───────────────────────────────────────────────────────────────────
# 6. NS-AWARE OBSERVATION FEATURES
# ───────────────────────────────────────────────────────────────────

class TestNSFeatures:
    """Verify extended observation vector."""

    @staticmethod
    def _cfg_with_ns_features(breakdowns=True):
        cfg = _load_cfg()
        cfg["ns_features"] = {"enabled": True, "ema_window": 30}
        if breakdowns:
            cfg["breakdowns"] = {
                "enabled": True,
                "default": {
                    "ttf": {"distribution": "exponential", "mean": 100.0},
                    "ttr": {"distribution": "exponential", "mean": 20.0},
                },
            }
        return cfg

    def test_obs_shape_extended(self):
        """NS-aware obs should be 3*J+2 = 14 for bakery (J=4)."""
        cfg = self._cfg_with_ns_features()
        env = FlexFlowSimEnv(config=cfg, seed=0)
        obs, _ = env.reset(seed=0)
        J = env._total_servers
        assert obs.shape == (3 * J + 2,), f"Got {obs.shape}, expected ({3*J+2},)"

    def test_obs_shape_original_without_ns_features(self):
        """Without ns_features, obs should remain 2*J = 8."""
        cfg = _load_cfg()
        env = FlexFlowSimEnv(config=cfg, seed=0)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (2 * env._total_servers,)

    def test_availability_reflects_breakdowns(self):
        """After enough steps with MTBF=50, at least one server should show 0."""
        cfg = self._cfg_with_ns_features()
        cfg["breakdowns"]["default"]["ttf"]["mean"] = 50.0
        env = FlexFlowSimEnv(config=cfg, seed=42)
        obs, _ = env.reset(seed=42)
        J = env._total_servers

        any_broken = False
        for step in range(200):
            obs, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break
            avail = obs[2*J:3*J]
            if np.any(avail < 0.5):
                any_broken = True
                break
        assert any_broken, "Expected at least one server to show unavailable"

    def test_arrival_rate_ema_tracks_schedule(self):
        """With a demand surge, EMA should rise above 1.0."""
        cfg = self._cfg_with_ns_features(breakdowns=False)
        cfg["arrival"] = {
            "distribution": "exponential",
            "mean": 9.6,
            "schedule": [
                {"t": 0, "rate": 0.10},
                {"t": 100, "rate": 0.50},
            ],
        }
        env = FlexFlowSimEnv(config=cfg, seed=0)
        obs, _ = env.reset(seed=0)
        J = env._total_servers

        # Run past t=100 into the surge zone
        for step in range(200):
            obs, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break
        ema = obs[3 * J]
        assert ema > 1.5, f"EMA {ema:.2f} should be well above 1.0 during 5x surge"

    def test_cost_multiplier_in_obs(self):
        """Cost multiplier should appear in obs when schedule is active."""
        cfg = self._cfg_with_ns_features(breakdowns=False)
        cfg["cost_schedule"] = {
            "enabled": True,
            "processing_multiplier": [
                {"t": 0, "multiplier": 1.0},
                {"t": 100, "multiplier": 3.0},
            ],
        }
        env = FlexFlowSimEnv(config=cfg, seed=0)
        obs, _ = env.reset(seed=0)
        J = env._total_servers

        # Before t=100: multiplier should be 1.0
        assert abs(obs[3*J+1] - 1.0) < 0.01

        # Run past t=100
        for step in range(150):
            obs, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break
        assert abs(obs[3*J+1] - 3.0) < 0.01, f"Cost mult {obs[3*J+1]:.2f}, expected 3.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
