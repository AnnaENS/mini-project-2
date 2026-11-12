"""DAgger on SuperTuxKart (Ross et al., 2011; course slide p23)

    python dagger.py --seed 0                    # DAgger, same online budget as rlpd.py
    python dagger.py --seed 0 --iterations 0     # behavioural cloning only (iteration 0)

Algorithm (beta_1 = 1, beta_i = 0 for i > 1):
    D <- the expert laps (collect_expert.py): states and EXPERT actions    (beta_1 = 1)
    repeat until the online budget is used:
        pi_i <- train a policy on D (supervised regression, no reward)
        drive one lap with pi_i (the LEARNER drives: beta_i = 0)
        at each visited state, ask the expert what it would do -> add to D

The simulator and the policy are deterministic, so the lap driven with pi_i to
collect data is exactly the lap that evaluate() in rlpd.py would drive (same
reset): we use it as the evaluation of pi_i, without driving a second lap.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_mind.nn import build_mlp

from stk_env import SoloTrackEnv, make_expert

# Hyper-parameters

@dataclass(frozen=True)
class Config:
    seed: int = 0
    dataset: str = "data/expert_abyss.pt"
    max_steps: int = 30_000        # online environment steps (same budget as rlpd.py)
    max_iterations: int = 1_000    # DAgger iterations (0 = behavioural cloning only)
    hidden: tuple[int, ...] = (256, 256)   # same as the RLPD actor
    lr: float = 1e-3
    batch_size: int = 256
    grad_steps: int = 3_000        # gradient steps per training (from scratch at each iteration)

# Policy

class Policy(nn.Module):
    """Deterministic policy: normalized observation -> MLP -> tanh -> action in [-1, 1]^2.
    Same architecture and normalization as the actor of rlpd.py."""

    def __init__(self, obs_dim, action_dim, hidden, obs_mean, obs_std):
        super().__init__()
        self.obs_mean, self.obs_std = obs_mean, obs_std
        self.model = build_mlp([obs_dim, *hidden, action_dim], output_activation=nn.Tanh())

    def act(self, obs):
        return self.model((obs - self.obs_mean) / self.obs_std)


def train_policy(obs, actions, cfg: Config, obs_mean, obs_std) -> Policy:
    """Behavioural cloning on the whole dataset D: minimize ||pi(s) - a_expert||^2.
    The policy is trained from scratch at each iteration (plain DAgger)."""
    policy = Policy(obs.shape[1], actions.shape[1], cfg.hidden, obs_mean, obs_std)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    for _ in range(cfg.grad_steps):
        idx = torch.randint(len(obs), (cfg.batch_size,))
        loss = F.mse_loss(policy.act(obs[idx]), actions[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy

# One lap driven by the learner, labelled by the expert

def drive_and_label(policy, env) -> tuple[np.ndarray, np.ndarray, dict]:
    """The learner drives; the expert only gives its action at each visited state.
    The expert remembers past observations (to detect when it is stuck), so a
    new expert is created for each lap and sees every step of it."""
    expert = make_expert(env)
    obs, info = env.reset(seed=0)            # same reset as evaluate() in rlpd.py
    states, labels = [], []
    total_reward, done = 0.0, False
    while not done:
        states.append(obs)
        labels.append(expert(info["raw_obs"]))           # what the expert WOULD do
        with torch.no_grad():
            action = policy.act(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        obs, reward, terminated, truncated, info = env.step(action)   # what the LEARNER does
        total_reward += reward
        done = terminated or truncated
    labels = np.array(labels)
    result = {"finished": bool(terminated), "lap_steps": info["lap_steps"], "return": total_reward,
              # share of labels where the expert does not floor the throttle
              # (it always does on its own laps; < full throttle = rescue mode when stuck)
              "slow_labels": float(np.mean(labels[:, 0] < 0.99))}
    return np.array(states), labels, result

# Training

def run_dagger(cfg: Config) -> list[dict]:
    torch.manual_seed(cfg.seed)
    env = SoloTrackEnv()

    # D <- the expert laps, with the actions the expert wanted (not the noisy executed ones)
    d = torch.load(cfg.dataset, weights_only=False)
    obs, actions = d["obs"].float(), d["expert_action"].float()
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-3      # same statistics as rlpd.py

    history, steps, iteration, start = [], 0, 0, time.time()
    while True:
        policy = train_policy(obs, actions, cfg, obs_mean, obs_std)
        new_obs, new_actions, result = drive_and_label(policy, env)
        # "step": online steps used to train this policy (iteration 0 = behavioural cloning)
        history.append({"step": steps, "iteration": iteration, "dataset_size": len(obs), **result})
        print(f"iter {iteration:3d}  step {steps:6d}  D {len(obs):6d}  finished {result['finished']}  "
              f"lap {result['lap_steps']:4d} steps  return {result['return']:7.1f}  "
              f"({time.time() - start:.0f}s)", flush=True)

        if steps >= cfg.max_steps or iteration >= cfg.max_iterations:
            break
        # Aggregate: D <- D U {(s, expert(s))} for the states visited by the learner
        obs = torch.cat([obs, torch.as_tensor(new_obs).float()])
        actions = torch.cat([actions, torch.as_tensor(new_actions).float()])
        steps += len(new_obs)
        iteration += 1

    env.close()
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=Config.max_steps)
    parser.add_argument("--iterations", type=int, default=Config.max_iterations)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    torch.set_num_threads(1)   # one core per run, so that several runs can run in parallel
    cfg = Config(seed=args.seed, max_steps=args.steps, max_iterations=args.iterations)
    name = f"{'dagger' if cfg.max_iterations > 0 else 'bc'}-s{cfg.seed}"

    history = run_dagger(cfg)

    path = Path(args.outdir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "evaluations": history}, indent=1))
    print(f"Results saved in {path}")
