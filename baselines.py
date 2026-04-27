"""
FlexFlowSim — Dispatching Rule Baselines
==========================================

Generalised for N-stage, M-server environments.

Policies:
  RoundRobin         — cycles through all actions
  Random             — uniform random action selection
  ShortestQueue      — routes to servers with lowest combined load
  FastServerFirst    — always picks action 0 (first server at each stage)
  SPT                — Shortest Processing Time: picks servers with lowest mean service time
  LPT                — Longest Processing Time: picks servers with highest mean service time
  CostMinimising     — picks servers with lowest processing cost per stage
  LeastUtilised      — routes to servers with lowest estimated utilisation
"""

import numpy as np


class BasePolicy:
    """Base class for benchmark policies."""
    name = "Base"

    def __init__(self, env=None):
        self.env = env

    def reset(self):
        pass

    def predict(self, obs):
        raise NotImplementedError


class RoundRobinPolicy(BasePolicy):
    """Cycles through actions 0, 1, ..., n_actions-1, 0, ..."""
    name = "Round Robin"

    def __init__(self, env=None):
        super().__init__(env)
        self._counter = 0

    def reset(self):
        self._counter = 0

    def predict(self, obs):
        n = self.env.n_actions if self.env else 4
        action = self._counter % n
        self._counter += 1
        return action


class RandomPolicy(BasePolicy):
    """Uniform random action selection."""
    name = "Random"

    def __init__(self, env=None, seed=None):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)

    def predict(self, obs):
        n = self.env.n_actions if self.env else 4
        return self._rng.integers(0, n)


class ShortestQueuePolicy(BasePolicy):
    """Routes to the action whose servers have the lowest combined load.

    Load = queue_length + in_service for each server in the action's route.
    """
    name = "Shortest Queue"

    def predict(self, obs):
        env = self.env
        n_total = env._total_servers
        q = obs[:n_total]
        busy = obs[n_total:]
        load = q + busy

        best_action = 0
        best_load = np.inf
        for a, route in enumerate(env._action_tuples):
            total_load = sum(load[env._flat_idx[(si, sj)]]
                             for si, sj in enumerate(route))
            if total_load < best_load:
                best_load = total_load
                best_action = a
        return best_action


class FastServerFirstPolicy(BasePolicy):
    """Always picks action 0 (first server at each stage)."""
    name = "Fast Server First"

    def predict(self, obs):
        return 0


class SPTPolicy(BasePolicy):
    """Shortest Processing Time: picks the action whose servers have the
    lowest total mean service time across stages.
    """
    name = "SPT"

    def __init__(self, env=None):
        super().__init__(env)
        self._best_action = None

    def reset(self):
        if self.env is None:
            return
        best_action = 0
        best_total = np.inf
        for a, route in enumerate(self.env._action_tuples):
            total_mean = 0.0
            for si, sj in enumerate(route):
                srv_cfg = self.env._stages[si]["servers"][sj]
                dist = srv_cfg["service_time"]
                total_mean += float(dist.get("mean", 1.0))
            if total_mean < best_total:
                best_total = total_mean
                best_action = a
        self._best_action = best_action

    def predict(self, obs):
        return self._best_action if self._best_action is not None else 0


class LPTPolicy(BasePolicy):
    """Longest Processing Time: picks the action whose servers have the
    highest total mean service time across stages.
    """
    name = "LPT"

    def __init__(self, env=None):
        super().__init__(env)
        self._best_action = None

    def reset(self):
        if self.env is None:
            return
        best_action = 0
        best_total = -np.inf
        for a, route in enumerate(self.env._action_tuples):
            total_mean = 0.0
            for si, sj in enumerate(route):
                srv_cfg = self.env._stages[si]["servers"][sj]
                dist = srv_cfg["service_time"]
                total_mean += float(dist.get("mean", 1.0))
            if total_mean > best_total:
                best_total = total_mean
                best_action = a
        self._best_action = best_action

    def predict(self, obs):
        return self._best_action if self._best_action is not None else 0


class CostMinimisingPolicy(BasePolicy):
    """Picks the action whose servers have the lowest total processing cost."""
    name = "Cost Minimising"

    def __init__(self, env=None):
        super().__init__(env)
        self._best_action = None

    def reset(self):
        if self.env is None:
            return
        best_action = 0
        best_cost = np.inf
        for a, route in enumerate(self.env._action_tuples):
            total_cost = sum(
                self.env._processing_cost[self.env._flat_idx[(si, sj)]]
                for si, sj in enumerate(route)
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = a
        self._best_action = best_action

    def predict(self, obs):
        return self._best_action if self._best_action is not None else 0


class LeastUtilisedPolicy(BasePolicy):
    """Routes to the action whose servers have the lowest estimated
    utilisation, approximated by current load (queue + in_service).

    Similar to ShortestQueue but weighted by mean service time to estimate
    the actual time commitment.
    """
    name = "Least Utilised"

    def predict(self, obs):
        env = self.env
        n_total = env._total_servers
        q = obs[:n_total]
        busy = obs[n_total:]

        best_action = 0
        best_score = np.inf
        for a, route in enumerate(env._action_tuples):
            score = 0.0
            for si, sj in enumerate(route):
                fi = env._flat_idx[(si, sj)]
                srv_cfg = env._stages[si]["servers"][sj]
                mu = float(srv_cfg["service_time"].get("mean", 1.0))
                score += (q[fi] + busy[fi]) * mu
            if score < best_score:
                best_score = score
                best_action = a
        return best_action


# ═══════════════════════════════════════════════════════════════════
# BANDITS (Paper 5)
# ═══════════════════════════════════════════════════════════════════
#
# Bandits use a different control flow from the dispatching rules above:
# they need a per-stage update() call after each routing decision, so they
# do not slot into the BASELINE_POLICIES registry. They have the same
# .predict(obs) interface for action selection but require their own run
# loop (see paper5/run_sweep.py for the canonical use). The implementations
# below are the ones used to produce baseline_sweep.csv in the Paper 5
# benchmark.

class VanillaThompsonSampling:
    """Per-stage Gaussian Thompson Sampling.

    Each (stage, server) arm maintains a posterior mean ``mu`` and variance
    ``var`` over the per-step processing-cost reward signal. At each routing
    decision, one sample is drawn per arm and the arg-max is selected
    independently per stage. Updates use an incremental running mean and a
    multiplicative variance decay (var *= 0.99 with a floor at 1.0).

    This is a heuristic, not a strict conjugate-Bayesian update — the decay
    is what gives the bandit the ability to track non-stationary reward
    means. Prior: ``mu_0 = 0``, ``var_0 = 100``.
    """
    def __init__(self, env, seed=None):
        self.env = env
        self._rng = np.random.default_rng(seed)
        self._n_stages = env.n_stages
        self._sps = env.servers_per_stage
        self.reset()

    def reset(self):
        self._mu = {}
        self._var = {}
        self._n = {}
        for si in range(self._n_stages):
            for sj in range(self._sps[si]):
                self._mu[(si, sj)] = 0.0
                self._var[(si, sj)] = 100.0
                self._n[(si, sj)] = 0

    def predict(self, obs):
        route = []
        for si in range(self._n_stages):
            best_sj, best = 0, -np.inf
            for sj in range(self._sps[si]):
                s = self._rng.normal(self._mu[(si, sj)],
                                     np.sqrt(self._var[(si, sj)]))
                if s > best:
                    best, best_sj = s, sj
            route.append(best_sj)
        # Encode route as flat action index (last-stage-fastest convention)
        action, mult = 0, 1
        for si in reversed(range(self._n_stages)):
            action += route[si] * mult
            mult *= self._sps[si]
        return action

    def update(self, si, sj, cost):
        """Update arm (si, sj) with reward = cost (which is negative for losses)."""
        k = (si, sj)
        n = self._n[k] + 1
        self._mu[k] += (cost - self._mu[k]) / n
        self._var[k] = max(self._var[k] * 0.99, 1.0)
        self._n[k] = n


class HybridBanditQueue:
    """Availability-gated Thompson Sampling with load-balancing bias (HBQ).

    Two extensions over Vanilla TS:
      1. Servers currently broken are excluded from the per-stage candidate
         set. If every server in a stage is broken, the routing decision
         falls back to ShortestQueue over those servers.
      2. The per-arm Thompson sample is penalised by a load term:
            score = ts_sample - load_weight * (queue_len + in_service)
         The default ``load_weight = 20`` was set so that a one-job
         difference in queue is worth roughly the same as a single
         time-step's reward signal.

    Variance decay is slightly slower (0.995 vs 0.99) than Vanilla TS,
    reflecting the additional structural information and trading exploration
    for stability.
    """
    def __init__(self, env, seed=None, load_weight=20.0):
        self.env = env
        self._rng = np.random.default_rng(seed)
        self._n_stages = env.n_stages
        self._sps = env.servers_per_stage
        self._lw = load_weight
        self.reset()

    def reset(self):
        self._mu = {}
        self._var = {}
        self._n = {}
        for si in range(self._n_stages):
            for sj in range(self._sps[si]):
                self._mu[(si, sj)] = 0.0
                self._var[(si, sj)] = 100.0
                self._n[(si, sj)] = 0

    def predict(self, obs):
        env = self.env
        J = env._total_servers
        q = obs[:J]
        busy = obs[J:2 * J]
        avail = (env._available if env._available is not None
                 else np.ones(J, dtype=bool))
        route = []
        for si in range(self._n_stages):
            best_sj, best_score, any_up = 0, -np.inf, False
            for sj in range(self._sps[si]):
                fi = env._flat_idx[(si, sj)]
                if not avail[fi]:
                    continue
                any_up = True
                ts = self._rng.normal(self._mu[(si, sj)],
                                      np.sqrt(self._var[(si, sj)]))
                score = ts - self._lw * (q[fi] + busy[fi])
                if score > best_score:
                    best_score, best_sj = score, sj
            if not any_up:
                # Fallback: every server in this stage is down. Pick the one
                # with the shortest queue.
                best_q = np.inf
                for sj in range(self._sps[si]):
                    fi = env._flat_idx[(si, sj)]
                    if q[fi] + busy[fi] < best_q:
                        best_q, best_sj = q[fi] + busy[fi], sj
            route.append(best_sj)
        action, mult = 0, 1
        for si in reversed(range(self._n_stages)):
            action += route[si] * mult
            mult *= self._sps[si]
        return action

    def update(self, si, sj, cost):
        k = (si, sj)
        n = self._n[k] + 1
        self._mu[k] += (cost - self._mu[k]) / n
        self._var[k] = max(self._var[k] * 0.995, 1.0)
        self._n[k] = n


# Registry of bandit algorithms (separate from BASELINE_POLICIES because the
# update() interface differs).
BANDIT_POLICIES = {
    "VanillaTS": VanillaThompsonSampling,
    "HBQ": HybridBanditQueue,
}


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════

BASELINE_POLICIES = {
    "RoundRobin": RoundRobinPolicy,
    "Random": RandomPolicy,
    "ShortestQueue": ShortestQueuePolicy,
    "FastServerFirst": FastServerFirstPolicy,
    "SPT": SPTPolicy,
    "LPT": LPTPolicy,
    "CostMinimising": CostMinimisingPolicy,
    "LeastUtilised": LeastUtilisedPolicy,
}


def run_episode(policy, env, seed):
    """Run a single episode with a baseline policy. Returns metrics dict."""
    policy.reset()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    dep = info["total_departed"]
    result = {
        "totalCost": info["total_cost"],
        "totalDeparted": dep,
        "costPerUnit": info["total_cost"] / max(dep, 1),
        "avgLeadTime": info["avg_lead_time"],
        "processingCost": info["processing_cost"],
        "idleCost": info["idle_cost"],
        "waitingCost": info["waiting_cost"],
        "totalReward": total_reward,
    }
    # Add per-server utilisation
    for i, u in enumerate(info["utilisation"]):
        result[f"util_{i}"] = u
    return result


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from env import FlexFlowSimEnv
    import os

    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "bakery_bk50.json")
    if not os.path.exists(cfg_path):
        print("Bakery config not found, using legacy env")
        from env import MultiServerMORLEnv
        env = MultiServerMORLEnv(weights=(0.33, 0.33, 0.34), seed=42)
    else:
        env = FlexFlowSimEnv(config=cfg_path, weights=(0.33, 0.33, 0.34), seed=42)

    print(f"Environment: {env.n_stages} stages, {env.servers_per_stage} servers, "
          f"{env.n_actions} actions\n")

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=3)

    for name, PolicyClass in BASELINE_POLICIES.items():
        policy = PolicyClass(env=env)
        results = [run_episode(policy, env, int(s)) for s in seeds]
        avg_cost = np.mean([r["totalCost"] for r in results])
        avg_dep = np.mean([r["totalDeparted"] for r in results])
        avg_lt = np.mean([r["avgLeadTime"] for r in results])
        print(f"  {name:20s}  Cost={avg_cost:8.1f}  Dep={avg_dep:5.1f}  LT={avg_lt:6.1f}")

    print("\nBaseline smoke test PASSED")
