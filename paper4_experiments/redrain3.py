"""Final drain protocol: dedicated arrival RNG (policy-invariant by construction)."""
import json
import numpy as np
from stable_baselines3 import PPO
import drain_experiment as D
from env import load_config, _make_sampler

class SplitRNGDrainEnv(D.DrainEnv):
    def reset(self, seed=None, options=None):
        r = super().reset(seed=seed, options=options)
        arr_rng = np.random.default_rng(10_000_019 * (seed or 0) + 7)
        self._arrival_sampler = _make_sampler(self._arrival_cfg, arr_rng)
        return r

D.DrainEnv = SplitRNGDrainEnv
cfg = load_config("configs/bakery_bk50.json")
out = json.load(open("drain_results_v3.json"))
for rule in ["RoundRobin", "Random", "ShortestQueue", "LeastUtilised"]:
    pol, fn = D.rule_fn(rule, cfg)
    out["rules"][rule]["drain"] = D.evaluate(fn, cfg, D.COSTFOCUS, drain=True, policy=pol)
    json.dump(out, open("drain_results_v4.json", "w"))
    print(f"rule {rule} drained", flush=True)
for seed in [42, 123, 256, 512, 1024]:
    m = PPO.load(f"model_wc010_s{seed}.zip")
    fn = lambda o: int(m.predict(o, deterministic=True)[0])
    out["ppo"][str(seed)]["drain"] = D.evaluate(fn, cfg, D.COSTFOCUS, drain=True)
    json.dump(out, open("drain_results_v4.json", "w"))
    print(f"seed {seed} drained", flush=True)
sq = [e["throughput"] for e in out["rules"]["ShortestQueue"]["drain"]]
ident = all(all(e["throughput"] == sq[i] for i, e in enumerate(v["drain"]))
            for v in out["ppo"].values()) and \
        all(all(e["throughput"] == sq[i] for i, e in enumerate(out["rules"][r]["drain"]))
            for r in out["rules"])
print(f"ARRIVAL IDENTITY (all methods, 250 eps + rules): {ident}")
print("REDRAIN3 COMPLETE")
