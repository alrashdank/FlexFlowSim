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
