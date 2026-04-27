"""
FlexFlowSim — N-Stage, M-Server Heterogeneous Flow Shop Environment
====================================================================

Gymnasium-compatible discrete-event simulation (SimPy backend) for
multi-objective reinforcement learning research in manufacturing routing.

Generalises the 2-stage/4-server environment from Paper 2 to:
  - Arbitrary number of stages (N >= 1)
  - Arbitrary number of servers per stage (M_i >= 1)
  - Configurable service time distributions (normal, exponential, lognormal, uniform)
  - JSON-driven parameterisation (no hard-coded values)

Action space:
  Discrete(prod(M_i)) — each action maps to a tuple of server indices,
  one per stage. E.g. with [2, 3] servers: 6 actions mapping to
  (stage1_server, stage2_server) tuples.

Observation space:
  Box(0, 1, shape=(2 * sum(M_i),)) — normalised queue lengths and
  in-service flags for every server across all stages.

Reward:
  Weighted linear scalarisation of three objectives (all continuous per step):
    r_cost = -(cost_rate × dt) / norm[0]        [minimise cost]
    r_tp   = (departure_rate × dt) / norm[1]     [maximise throughput]
    r_lt   = -(WIP × dt) / norm[2]               [minimise lead time / WIP]
    R = scale × (w_cost × r_cost + w_tp × r_tp + w_lt × r_lt)

Reference:
  Babor & Hitzmann (2022), DOI: 10.17632/dhgbssb8ns.2 (bakery config)
  Therkelsen et al. (2014), J. Food Engineering (cost ratios)
"""

import json
import warnings
from itertools import product as cartesian_product
from pathlib import Path

import gymnasium as gym
import numpy as np
import simpy
from gymnasium import spaces


# ═══════════════════════════════════════════════════════════════════
# SERVICE TIME SAMPLERS
# ═══════════════════════════════════════════════════════════════════

def _make_sampler(dist_cfg, rng):
    """Return a callable(rng) -> float for the given distribution config."""
    dist = dist_cfg["distribution"].lower()
    if dist == "exponential":
        mu = float(dist_cfg["mean"])
        return lambda: max(rng.exponential(mu), 1e-6)
    elif dist == "normal":
        mu = float(dist_cfg["mean"])
        sigma = float(dist_cfg["std"])
        min_val = float(dist_cfg.get("min", 0.1))
        return lambda: max(rng.normal(mu, sigma), min_val)
    elif dist == "lognormal":
        mu = float(dist_cfg["mean"])
        sigma = float(dist_cfg["std"])
        return lambda: max(rng.lognormal(mu, sigma), 1e-6)
    elif dist == "uniform":
        lo = float(dist_cfg["low"])
        hi = float(dist_cfg["high"])
        return lambda: rng.uniform(lo, hi)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


# ═══════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ═══════════════════════════════════════════════════════════════════

def load_config(path):
    """Load and validate a FlexFlowSim JSON config file."""
    with open(path) as f:
        cfg = json.load(f)

    required = ["stages", "arrival", "waiting_cost", "max_time", "dt",
                 "max_queue", "norm_constants"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing keys: {missing}")

    for i, stage in enumerate(cfg["stages"]):
        if "servers" not in stage or len(stage["servers"]) < 1:
            raise ValueError(f"Stage {i} must have at least 1 server")
        for j, srv in enumerate(stage["servers"]):
            if "service_time" not in srv:
                raise ValueError(f"Stage {i}, Server {j}: missing 'service_time'")
            if "processing_cost" not in srv:
                raise ValueError(f"Stage {i}, Server {j}: missing 'processing_cost'")

    return cfg


# ═══════════════════════════════════════════════════════════════════
# NON-STATIONARITY HELPERS (Paper 5)
# ═══════════════════════════════════════════════════════════════════

class _PiecewiseSchedule:
    """
    Piecewise-constant schedule with caching for fast lookup.

    breakpoints : list of (t, value) pairs, sorted by t, with t[0] = 0.
    value_at(t) returns the value of the segment containing t.
    """

    def __init__(self, breakpoints, value_key):
        if not breakpoints:
            raise ValueError("Schedule must have at least one breakpoint")
        sorted_bp = sorted(breakpoints, key=lambda x: x["t"])
        if sorted_bp[0]["t"] > 0:
            raise ValueError("First breakpoint must be at t=0")
        self._times = np.array([bp["t"] for bp in sorted_bp], dtype=np.float64)
        self._values = np.array([bp[value_key] for bp in sorted_bp], dtype=np.float64)
        self._cache_idx = 0

    def value_at(self, t):
        # Advance cache forward (monotone access typical)
        while (self._cache_idx + 1 < len(self._times)
               and self._times[self._cache_idx + 1] <= t):
            self._cache_idx += 1
        # Handle backward access (rare, e.g. reset)
        while self._cache_idx > 0 and self._times[self._cache_idx] > t:
            self._cache_idx -= 1
        return float(self._values[self._cache_idx])

    def reset_cache(self):
        self._cache_idx = 0

    @property
    def max_value(self):
        return float(np.max(self._values))

    @property
    def min_value(self):
        return float(np.min(self._values))


def _breakdown_sampler(dist_cfg, rng):
    """Sampler for breakdown TTF/TTR distributions. Returns callable()->float."""
    dist = dist_cfg["distribution"].lower()
    if dist == "exponential":
        mu = float(dist_cfg["mean"])
        return lambda: max(rng.exponential(mu), 1e-6)
    elif dist == "lognormal":
        mean = float(dist_cfg["mean"])
        std = float(dist_cfg["std"])
        # Convert mean/std of the lognormal to mu/sigma of the underlying normal
        var = std ** 2
        sigma = np.sqrt(np.log(1.0 + var / (mean ** 2)))
        mu = np.log(mean) - 0.5 * sigma ** 2
        return lambda: max(rng.lognormal(mu, sigma), 1e-6)
    elif dist == "weibull":
        shape = float(dist_cfg["shape"])
        scale = float(dist_cfg["scale"])
        return lambda: max(scale * rng.weibull(shape), 1e-6)
    elif dist == "normal":
        mu = float(dist_cfg["mean"])
        sigma = float(dist_cfg["std"])
        return lambda: max(rng.normal(mu, sigma), 1e-6)
    else:
        raise ValueError(f"Unknown breakdown distribution: {dist}")


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

class FlexFlowSimEnv(gym.Env):
    """
    N-stage, M-server heterogeneous flow shop with MORL reward.

    Parameters
    ----------
    config : dict or str or Path
        Environment configuration. If str/Path, loaded from JSON file.
    weights : tuple of float
        (w_cost, w_throughput, w_leadtime) for reward scalarisation.
    reward_scale : float
        Multiplicative scale for the scalarised reward (default 10.0).
    seed : int, optional
        Random seed.
    """

    metadata = {"render_modes": []}

    def __init__(self, config, weights=(0.33, 0.33, 0.34),
                 reward_scale=10.0, seed=None):
        super().__init__()

        # Load config
        if isinstance(config, (str, Path)):
            self.cfg = load_config(config)
        else:
            self.cfg = config

        self.weights = np.array(weights, dtype=np.float32)
        self.reward_scale = reward_scale

        # Parse stage/server structure
        self._stages = self.cfg["stages"]
        self._n_stages = len(self._stages)
        self._servers_per_stage = [len(s["servers"]) for s in self._stages]
        self._total_servers = sum(self._servers_per_stage)

        # Build action map: action_index -> tuple of server indices per stage
        self._action_tuples = list(
            cartesian_product(*(range(m) for m in self._servers_per_stage))
        )
        self._n_actions = len(self._action_tuples)

        # Flat server index mapping: (stage_idx, local_srv_idx) -> flat_idx
        self._flat_idx = {}
        idx = 0
        for si, n_srv in enumerate(self._servers_per_stage):
            for sj in range(n_srv):
                self._flat_idx[(si, sj)] = idx
                idx += 1

        # Spaces
        self.action_space = spaces.Discrete(self._n_actions)
        obs_dim = 2 * self._total_servers  # queue_lens + in_service flags
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # System params
        self._arrival_cfg = self.cfg["arrival"]
        self._waiting_cost = float(self.cfg["waiting_cost"])
        self._max_time = float(self.cfg["max_time"])
        self._dt = float(self.cfg["dt"])
        self._max_queue = float(self.cfg["max_queue"])
        self._norm = [float(n) for n in self.cfg["norm_constants"]]

        # Per-server cost arrays (flat)
        self._processing_cost = np.zeros(self._total_servers)
        self._idle_cost = np.zeros(self._total_servers)
        for si, stage in enumerate(self._stages):
            for sj, srv in enumerate(stage["servers"]):
                fi = self._flat_idx[(si, sj)]
                self._processing_cost[fi] = float(srv["processing_cost"])
                self._idle_cost[fi] = float(srv.get("idle_cost", 1.0))

        # RNG
        self._rng = np.random.default_rng(seed)
        self._current_action = 0

        # ──────────────────────────────────────────────────────────
        # Non-stationarity configuration (Paper 5)
        # All blocks are optional; absence preserves Paper 3/4 behaviour.
        # ──────────────────────────────────────────────────────────
        self._seed = seed  # retained so reset() can derive the breakdown RNG

        # Arrival schedule: time-varying lambda(t)
        self._arrival_schedule_cfg = self._arrival_cfg.get("schedule", None)
        self._arrival_is_nonstationary = self._arrival_schedule_cfg is not None

        # Cost schedule: piecewise-constant multiplier on processing cost
        cs_cfg = self.cfg.get("cost_schedule", {})
        self._cost_schedule_enabled = bool(cs_cfg.get("enabled", False))
        self._cost_schedule_cfg = cs_cfg if self._cost_schedule_enabled else None

        # Breakdowns: per-server TTF/TTR processes
        bd_cfg = self.cfg.get("breakdowns", {})
        self._breakdowns_enabled = bool(bd_cfg.get("enabled", False))
        self._breakdowns_cfg = bd_cfg if self._breakdowns_enabled else None

        # CPU-aligned reward (Paper 5 fix for reward-metric misalignment).
        # Opt-in via cfg["cpu_aligned_reward"]=True. Default False preserves
        # Paper 3/4 byte-identical behaviour.
        self._cpu_aligned = bool(self.cfg.get("cpu_aligned_reward", False))

        # NS-aware observation features (Paper 5)
        ns_feat_cfg = self.cfg.get("ns_features", {})
        self._ns_features_enabled = bool(ns_feat_cfg.get("enabled", False))
        self._ns_ema_window = int(ns_feat_cfg.get("ema_window", 30))
        # Baseline arrival rate for normalisation
        arrival_mean = float(self._arrival_cfg.get("mean", 10.0))
        self._baseline_arrival_rate = 1.0 / max(arrival_mean, 1e-9)

        # Recompute obs_dim if NS features are enabled
        if self._ns_features_enabled:
            # base (2*J) + availability (J) + arrival_rate_ema (1) + cost_mult (1)
            obs_dim = 3 * self._total_servers + 2
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    @property
    def n_stages(self):
        return self._n_stages

    @property
    def servers_per_stage(self):
        return list(self._servers_per_stage)

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def action_map(self):
        return list(self._action_tuples)

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._simpy_env = simpy.Environment()
        self._resources = [
            simpy.Resource(self._simpy_env, capacity=1)
            for _ in range(self._total_servers)
        ]

        # State tracking (flat arrays indexed by flat server index)
        self._queue_len = np.zeros(self._total_servers, dtype=np.float64)
        self._in_service = np.zeros(self._total_servers, dtype=np.float64)
        self._departures = 0
        self._current_action = 0

        # Cost accumulators
        self._acc_processing = 0.0
        self._acc_idle = 0.0
        self._acc_waiting = 0.0

        # Lead time tracking
        self._lt_sum = 0.0
        self._lt_count = 0

        # Utilisation tracking
        self._busy_time = np.zeros(self._total_servers, dtype=np.float64)

        # NS feature tracking
        if self._ns_features_enabled:
            alpha = 2.0 / (self._ns_ema_window + 1.0)
            self._ema_alpha = alpha
            self._arrival_rate_ema = self._baseline_arrival_rate
            self._step_arrival_count = 0

        # Build service time samplers (must be rebuilt each reset for RNG state)
        self._samplers = {}
        for si, stage in enumerate(self._stages):
            for sj, srv in enumerate(stage["servers"]):
                fi = self._flat_idx[(si, sj)]
                self._samplers[fi] = _make_sampler(srv["service_time"], self._rng)

        # Arrival sampler
        self._arrival_sampler = _make_sampler(self._arrival_cfg, self._rng)

        # ──────────────────────────────────────────────────────────
        # Non-stationarity state (Paper 5) — only built if enabled
        # ──────────────────────────────────────────────────────────
        # Arrival schedule
        if self._arrival_is_nonstationary:
            self._arrival_rate_schedule = _PiecewiseSchedule(
                self._arrival_schedule_cfg, value_key="rate"
            )
        else:
            self._arrival_rate_schedule = None

        # Cost multiplier schedule
        if self._cost_schedule_enabled:
            self._cost_multiplier_schedule = _PiecewiseSchedule(
                self._cost_schedule_cfg["processing_multiplier"],
                value_key="multiplier",
            )
        else:
            self._cost_multiplier_schedule = None

        # Breakdowns
        if self._breakdowns_enabled:
            # Separate RNG stream so breakdowns are independent of arrivals/services
            bd_seed = (self._seed ^ 0xB7EA0) if self._seed is not None else None
            self._breakdown_rng = np.random.default_rng(bd_seed)
            default_cfg = self._breakdowns_cfg.get("default", {})
            overrides = self._breakdowns_cfg.get("overrides", [])
            override_map = {}
            for ov in overrides:
                key = (int(ov["stage"]), int(ov["server"]))
                override_map[key] = ov

            self._available = np.ones(self._total_servers, dtype=bool)
            self._available_events = [None] * self._total_servers
            self._service_process = [None] * self._total_servers
            self._breakdown_count = np.zeros(self._total_servers, dtype=np.int64)
            self._breakdown_time = np.zeros(self._total_servers, dtype=np.float64)
            self._ttf_samplers = {}
            self._ttr_samplers = {}
            for si in range(self._n_stages):
                for sj in range(self._servers_per_stage[si]):
                    fi = self._flat_idx[(si, sj)]
                    ov = override_map.get((si, sj), {})
                    ttf_cfg = ov.get("ttf", default_cfg.get("ttf"))
                    ttr_cfg = ov.get("ttr", default_cfg.get("ttr"))
                    if ttf_cfg is None or ttr_cfg is None:
                        raise ValueError(
                            f"Breakdowns enabled but no ttf/ttr for stage {si} server {sj}"
                        )
                    self._ttf_samplers[fi] = _breakdown_sampler(
                        ttf_cfg, self._breakdown_rng
                    )
                    self._ttr_samplers[fi] = _breakdown_sampler(
                        ttr_cfg, self._breakdown_rng
                    )
                    self._available_events[fi] = self._simpy_env.event()
        else:
            self._available = None
            self._available_events = None

        # Start simulation
        self._simpy_env.process(self._arrival_process())
        if self._breakdowns_enabled:
            for fi in range(self._total_servers):
                self._simpy_env.process(self._breakdown_process(fi))
        self._simpy_env.run(until=self._dt)
        self._accumulate_costs(self._dt)

        return self._get_obs(), self._get_info()

    # ─────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
        self._current_action = int(action)
        t_start = self._simpy_env.now
        t_end = t_start + self._dt
        dep_before = self._departures

        self._simpy_env.run(until=t_end)
        dt_actual = max(self._simpy_env.now - t_start, 1e-9)

        self._accumulate_costs(dt_actual)
        new_departures = self._departures - dep_before
        reward = self._compute_reward(dt_actual, new_departures)

        # Update arrival rate EMA
        if self._ns_features_enabled:
            step_rate = self._step_arrival_count / max(dt_actual, 1e-9)
            self._arrival_rate_ema = (
                self._ema_alpha * step_rate
                + (1.0 - self._ema_alpha) * self._arrival_rate_ema
            )
            self._step_arrival_count = 0

        truncated = self._simpy_env.now >= self._max_time
        obs = self._get_obs()
        info = self._get_info()
        info["new_departures"] = new_departures

        return obs, reward, False, truncated, info

    # ─────────────────────────────────────────────────────────────
    # SIMPY PROCESSES
    # ─────────────────────────────────────────────────────────────
    def _arrival_process(self):
        if not self._arrival_is_nonstationary:
            # Original stationary path — byte-identical to Paper 3/4
            while True:
                iat = self._arrival_sampler()
                yield self._simpy_env.timeout(iat)
                server_route = self._action_tuples[self._current_action]
                self._simpy_env.process(self._entity_process(server_route))
                if self._ns_features_enabled:
                    self._step_arrival_count += 1
        else:
            # Non-homogeneous Poisson via thinning (Lewis & Shedler, 1979)
            # schedule values are rates (arrivals per time unit)
            lambda_max = self._arrival_rate_schedule.max_value
            while True:
                iat = self._rng.exponential(1.0 / max(lambda_max, 1e-9))
                yield self._simpy_env.timeout(iat)
                current_rate = self._arrival_rate_schedule.value_at(
                    self._simpy_env.now
                )
                if self._rng.uniform() <= current_rate / max(lambda_max, 1e-9):
                    server_route = self._action_tuples[self._current_action]
                    self._simpy_env.process(self._entity_process(server_route))
                    if self._ns_features_enabled:
                        self._step_arrival_count += 1

    def _breakdown_process(self, fi):
        """Alternates failure and repair cycles for a single server.
        On failure, interrupts any in-service entity so it pauses."""
        while True:
            ttf = self._ttf_samplers[fi]()
            yield self._simpy_env.timeout(ttf)
            # Failure
            self._available[fi] = False
            self._breakdown_count[fi] += 1
            # Interrupt in-service entity so its service clock pauses
            sp = self._service_process[fi]
            if sp is not None and sp.is_alive:
                try:
                    sp.interrupt()
                except RuntimeError:
                    pass  # process already completed at this exact tick
            start = self._simpy_env.now
            ttr = self._ttr_samplers[fi]()
            yield self._simpy_env.timeout(ttr)
            # Repair
            self._breakdown_time[fi] += self._simpy_env.now - start
            self._available[fi] = True
            ev = self._available_events[fi]
            self._available_events[fi] = self._simpy_env.event()
            ev.succeed()

    def _entity_process(self, server_route):
        """Entity flows through stages sequentially, using the pre-assigned server at each stage."""
        arrival_time = self._simpy_env.now

        for stage_idx, local_srv_idx in enumerate(server_route):
            fi = self._flat_idx[(stage_idx, local_srv_idx)]

            self._queue_len[fi] += 1

            if not self._breakdowns_enabled:
                # Original path — byte-identical to Paper 3/4
                with self._resources[fi].request() as req:
                    yield req
                    self._queue_len[fi] -= 1
                    self._in_service[fi] += 1
                    svc_time = self._samplers[fi]()
                    yield self._simpy_env.timeout(svc_time)
                    self._in_service[fi] -= 1
            else:
                # Breakdown-aware path: gate on availability, pause-and-resume
                # if the server breaks down mid-service (standard Pinedo / Banks
                # et al. convention; see Paper 5 methodology).
                started = False
                while not started:
                    while not self._available[fi]:
                        yield self._available_events[fi]
                    req = self._resources[fi].request()
                    yield req
                    if not self._available[fi]:
                        self._resources[fi].release(req)
                        continue
                    # Acquired and available: begin service
                    self._queue_len[fi] -= 1
                    svc_time = self._samplers[fi]()
                    remaining = svc_time
                    self._service_process[fi] = self._simpy_env.active_process
                    try:
                        while remaining > 0:
                            # Wait for server to come back online if broken
                            while not self._available[fi]:
                                yield self._available_events[fi]
                            self._in_service[fi] += 1
                            start_tick = self._simpy_env.now
                            try:
                                yield self._simpy_env.timeout(remaining)
                                remaining = 0.0
                                self._in_service[fi] -= 1
                            except simpy.Interrupt:
                                elapsed = self._simpy_env.now - start_tick
                                remaining = max(remaining - elapsed, 0.0)
                                self._in_service[fi] -= 1
                    finally:
                        self._service_process[fi] = None
                    self._resources[fi].release(req)
                    started = True

        # Departure (after last stage)
        self._departures += 1
        self._lt_sum += self._simpy_env.now - arrival_time
        self._lt_count += 1

    # ─────────────────────────────────────────────────────────────
    # COST & UTILISATION
    # ─────────────────────────────────────────────────────────────
    def _accumulate_costs(self, dt):
        if self._cost_multiplier_schedule is not None:
            mult = self._cost_multiplier_schedule.value_at(self._simpy_env.now)
        else:
            mult = 1.0
        for i in range(self._total_servers):
            busy = min(self._in_service[i], 1.0)
            self._acc_processing += busy * self._processing_cost[i] * mult * dt
            self._acc_idle += (1.0 - busy) * self._idle_cost[i] * dt
            self._busy_time[i] += busy * dt
        self._acc_waiting += self._waiting_cost * np.sum(self._queue_len) * dt

    # ─────────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self, dt, new_departures):
        norm = self._norm

        # Current cost multiplier (1.0 if no schedule)
        if self._cost_multiplier_schedule is not None:
            mult = self._cost_multiplier_schedule.value_at(self._simpy_env.now)
        else:
            mult = 1.0

        # Cost rate
        cost_rate = 0.0
        for i in range(self._total_servers):
            busy = min(self._in_service[i], 1.0)
            cost_rate += busy * self._processing_cost[i] * mult
            cost_rate += (1.0 - busy) * self._idle_cost[i]
        cost_rate += self._waiting_cost * np.sum(self._queue_len)

        # CPU-aligned reward path (opt-in via cfg["cpu_aligned_reward"]=True)
        # Design: every step gives credit for (new departures) minus
        # (step cost normalised). At terminal step, give a large bonus/penalty
        # equal to -CPU (normalised). This aligns the step-wise gradient with
        # what matters over the episode: maximise throughput per unit cost.
        if getattr(self, "_cpu_aligned", False):
            step_cost = cost_rate * dt
            # Dense component: reward departures, penalise cost.
            # Scale so throughput dominates idleness preference.
            r_tp_dense = new_departures
            r_cost_dense = -step_cost / max(norm[0], 1e-9)
            # Terminal CPU bonus (only at end of episode)
            terminal_bonus = 0.0
            if self._simpy_env.now >= self._max_time - self._dt:
                total_cost_so_far = (self._acc_processing + self._acc_idle
                                     + self._acc_waiting)
                deps = max(self._departures, 1)
                cpu_observed = total_cost_so_far / deps
                # Normalise against a reference CPU (norm[0] / norm[1] ≈ per-unit scale)
                cpu_ref = max(norm[0], 1.0) / max(norm[1], 1.0)
                # Large negative reward proportional to how bad CPU is
                terminal_bonus = -5.0 * cpu_observed / cpu_ref
            wip = (np.sum(self._queue_len)
                   + np.sum(np.minimum(self._in_service, 1.0)))
            r_lt = -(wip * dt) / max(norm[2], 1e-9)
            w = self.weights
            dense = w[0] * r_cost_dense + w[1] * r_tp_dense + w[2] * r_lt
            return float(self.reward_scale * dense + terminal_bonus)

        # Legacy Paper 3/4 reward (default; preserved byte-identical)
        t = max(self._simpy_env.now, 0.01)
        departure_rate = self._departures / t
        wip = np.sum(self._queue_len) + np.sum(np.minimum(self._in_service, 1.0))
        r_cost = -(cost_rate * dt) / max(norm[0], 1e-9)
        r_tp = (departure_rate * dt) / max(norm[1], 1e-9)
        r_lt = -(wip * dt) / max(norm[2], 1e-9)
        w = self.weights
        return float(self.reward_scale * (w[0] * r_cost + w[1] * r_tp + w[2] * r_lt))

    # ─────────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self):
        mq = self._max_queue
        if not self._ns_features_enabled:
            # Original path — byte-identical to Paper 3/4
            obs = np.zeros(2 * self._total_servers, dtype=np.float32)
            for i in range(self._total_servers):
                obs[i] = np.clip(self._queue_len[i] / mq, 0, 1)
                obs[self._total_servers + i] = np.clip(self._in_service[i], 0, 1)
            return obs
        else:
            # Extended observation: base + availability + arrival_ema + cost_mult
            J = self._total_servers
            obs = np.zeros(3 * J + 2, dtype=np.float32)

            # [0..J-1] normalised queue lengths
            for i in range(J):
                obs[i] = np.clip(self._queue_len[i] / mq, 0, 1)

            # [J..2J-1] in-service flags
            for i in range(J):
                obs[J + i] = np.clip(self._in_service[i], 0, 1)

            # [2J..3J-1] per-server availability (1=up, 0=broken)
            if self._available is not None:
                for i in range(J):
                    obs[2 * J + i] = 1.0 if self._available[i] else 0.0
            else:
                obs[2 * J: 3 * J] = 1.0  # all available if no breakdowns

            # [3J] arrival rate EMA, normalised by baseline rate
            obs[3 * J] = np.clip(
                self._arrival_rate_ema / max(self._baseline_arrival_rate, 1e-9),
                0.0, 5.0
            )

            # [3J+1] current cost multiplier
            if self._cost_multiplier_schedule is not None:
                obs[3 * J + 1] = self._cost_multiplier_schedule.value_at(
                    self._simpy_env.now
                )
            else:
                obs[3 * J + 1] = 1.0

            return obs

    # ─────────────────────────────────────────────────────────────
    # INFO
    # ─────────────────────────────────────────────────────────────
    def _get_info(self):
        total_cost = self._acc_processing + self._acc_idle + self._acc_waiting
        avg_lt = self._lt_sum / self._lt_count if self._lt_count > 0 else 0.0
        t = max(self._simpy_env.now, 1e-9)

        # Per-server utilisation
        util = [self._busy_time[i] / t for i in range(self._total_servers)]

        # Per-stage utilisation (average across servers in that stage)
        stage_util = []
        for si in range(self._n_stages):
            stage_utils = [util[self._flat_idx[(si, sj)]]
                           for sj in range(self._servers_per_stage[si])]
            stage_util.append(float(np.mean(stage_utils)))

        info = {
            "total_cost": total_cost,
            "processing_cost": self._acc_processing,
            "idle_cost": self._acc_idle,
            "waiting_cost": self._acc_waiting,
            "total_departed": self._departures,
            "avg_lead_time": avg_lt,
            "lt_count": self._lt_count,
            "sim_time": self._simpy_env.now,
            "utilisation": util,
            "stage_utilisation": stage_util,
        }

        if self._breakdowns_enabled:
            info["breakdown_count"] = self._breakdown_count.tolist()
            info["breakdown_time"] = self._breakdown_time.tolist()
            info["server_available"] = self._available.tolist()

        return info

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────
    def describe(self):
        """Print a human-readable summary of the environment configuration."""
        lines = [
            f"FlexFlowSim Environment",
            f"  Stages: {self._n_stages}",
            f"  Servers per stage: {self._servers_per_stage}",
            f"  Total servers: {self._total_servers}",
            f"  Action space: Discrete({self._n_actions})",
            f"  Observation space: Box(shape=({2 * self._total_servers},))",
            f"  Arrival: {self._arrival_cfg}",
            f"  Max time: {self._max_time}, dt: {self._dt}",
            f"  Norm constants: {self._norm}",
            f"  Weights: {self.weights.tolist()}",
        ]
        for si, stage in enumerate(self._stages):
            name = stage.get("name", f"Stage {si+1}")
            lines.append(f"\n  {name}:")
            for sj, srv in enumerate(stage["servers"]):
                sname = srv.get("name", f"Server {sj+1}")
                dist = srv["service_time"]
                pcost = srv["processing_cost"]
                lines.append(f"    {sname}: {dist}, proc_cost={pcost}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY WRAPPER
# ═══════════════════════════════════════════════════════════════════

class MultiServerMORLEnv(FlexFlowSimEnv):
    """
    Drop-in replacement for the Paper 2 environment.

    Accepts the old-style params dict and converts to FlexFlowSim config.
    """

    _LEGACY_DEFAULTS = {
        "interarrival_mu": 1.8,
        "service_mu": [2.0, 5.0, 2.0, 5.0],
        "processing_cost": [15.0, 3.0, 15.0, 3.0],
        "idle_cost": [1.0, 1.0, 1.0, 1.0],
        "waiting_cost": 0.1,
        "norm_constants": [25000.0, 330.0, 56000.0],
        "dt": 0.2,
        "max_time": 1000.0,
        "max_queue": 50.0,
    }

    def __init__(self, weights=(0.33, 0.33, 0.34), params=None, seed=None):
        p = {**self._LEGACY_DEFAULTS, **(params or {})}
        config = self._convert_legacy(p)
        super().__init__(config=config, weights=weights, seed=seed)

    @staticmethod
    def _convert_legacy(p):
        """Convert Paper 2 flat params to FlexFlowSim config."""
        smu = p["service_mu"]
        pcost = p["processing_cost"]
        icost = p["idle_cost"]

        def _srv(idx):
            return {
                "name": f"Server {idx + 1}",
                "service_time": {"distribution": "exponential", "mean": smu[idx]},
                "processing_cost": pcost[idx],
                "idle_cost": icost[idx],
            }

        return {
            "stages": [
                {"name": "Stage 1", "servers": [_srv(0), _srv(1)]},
                {"name": "Stage 2", "servers": [_srv(2), _srv(3)]},
            ],
            "arrival": {"distribution": "exponential", "mean": p["interarrival_mu"]},
            "waiting_cost": p["waiting_cost"],
            "max_time": p["max_time"],
            "dt": p["dt"],
            "max_queue": p["max_queue"],
            "norm_constants": p["norm_constants"],
        }


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Backward Compatibility (Paper 2 style) ===")
    env = MultiServerMORLEnv(weights=(0.33, 0.33, 0.34), seed=42)
    print(env.describe())
    obs, info = env.reset()
    total_reward = 0
    for step in range(100):
        obs, reward, _, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if truncated:
            break
    print(f"\nSteps: {step+1}, Reward: {total_reward:.4f}, Dep: {info['total_departed']}")
    print(f"Utilisation: {[f'{u:.3f}' for u in info['utilisation']]}")
    print("Legacy smoke test PASSED\n")

    # Test with JSON config if available
    import os
    bakery_cfg = os.path.join(os.path.dirname(__file__), "configs", "bakery_bk50.json")
    if os.path.exists(bakery_cfg):
        print("=== Bakery BK50 Config ===")
        env2 = FlexFlowSimEnv(config=bakery_cfg, weights=(0.33, 0.33, 0.34), seed=42)
        print(env2.describe())
        obs, info = env2.reset()
        total_reward = 0
        for step in range(100):
            obs, reward, _, truncated, info = env2.step(env2.action_space.sample())
            total_reward += reward
            if truncated:
                break
        print(f"\nSteps: {step+1}, Reward: {total_reward:.4f}, Dep: {info['total_departed']}")
        print(f"Utilisation: {[f'{u:.3f}' for u in info['utilisation']]}")
        print("Bakery smoke test PASSED")
