"""Record expert laps as a dataset of transitions

    uv run collect_expert.py --laps 10 --noise 0.2 --out data/expert_abyss.pt

At each step, the EXECUTED action is the expert's action plus Gaussian noise on
the steering (noise = 0 gives identical laps, since the expert and the
simulator are deterministic). We store:
- action : the executed action
- expert_action : what the expert wanted to do (the label for imitation)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from stk_env import SoloTrackEnv, make_expert


def collect(track: str, n_laps: int, noise: float, seed: int) -> dict:
    env = SoloTrackEnv(track=track)
    rng = np.random.default_rng(seed)
    data = {k: [] for k in ("obs", "action", "expert_action", "reward", "next_obs", "terminated", "truncated")}
    lap_steps, finished = [], []

    for lap in range(n_laps):
        expert = make_expert(env)
        obs, info = env.reset(seed=seed + lap)
        done = False
        while not done:
            expert_action = expert(info["raw_obs"])
            action = expert_action.copy()
            # noise on the steering only
            action[1] += noise * rng.standard_normal()
            action = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action)

            data["obs"].append(obs)
            data["action"].append(action)
            data["expert_action"].append(expert_action)
            data["reward"].append(reward)
            data["next_obs"].append(next_obs)
            data["terminated"].append(terminated)
            data["truncated"].append(truncated)

            obs, done = next_obs, terminated or truncated
        lap_steps.append(info["lap_steps"])
        finished.append(terminated)
        print(f"lap {lap + 1}/{n_laps}: finished={terminated} in {info['lap_steps']} steps", flush=True)
    env.close()

    out = {k: torch.as_tensor(np.array(v)) for k, v in data.items()}
    out["meta"] = {"track": track, "n_laps": n_laps, "noise": noise, "seed": seed,
                   "lap_steps": lap_steps, "finished": finished, "expert": "team4"}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="abyss")
    p.add_argument("--laps", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/expert_abyss.pt"))
    a = p.parse_args()

    dataset = collect(a.track, a.laps, a.noise, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, a.out)
    n = len(dataset["reward"])
    print(f"\nSaved {n} transitions ({a.laps} laps) to {a.out}")
    print(f"lap steps: mean {np.mean(dataset['meta']['lap_steps']):.1f}, all finished: {all(dataset['meta']['finished'])}")
    print({k: tuple(v.shape) for k, v in dataset.items() if k != "meta"})
