"""RLPD on SuperTuxKart: SAC + symmetric sampling (+ LayerNorm in the critics)

    python rlpd.py --seed 0                      # RLPD
    python rlpd.py --seed 0 --offline-ratio 0    # plain SAC (no expert data), for comparison
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from rl_mind.collectors import TransitionCollector
from rl_mind.core import Actor, StochasticAction
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.env import VecEnv
from rl_mind.nn import build_mlp, soft_update

import stk_env
from stk_env import SoloTrackEnv

# Hyper-parameters

@dataclass(frozen=True)
class Config:
    seed: int = 0
    dataset: str = "data/expert_abyss.pt"
    max_steps: int = 30_000        # online environment steps (1 gradient step per env step)
    learning_starts: int = 1_000   # online steps before the first update
    offline_ratio: float = 0.5     # part of each batch taken from the expert data (0 = plain SAC)
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden: tuple[int, ...] = (256, 256)
    init_alpha: float = 0.2        # initial entropy coefficient (then tuned automatically)
    eval_interval: int = 2_500     # steps between two evaluations (one lap each)

# Networks

class Critic(nn.Module):
    """Q(s, a): our ContinuousQNetwork from mini-project 1, with LayerNorm always on.
    Each hidden layer is Linear -> LayerNorm -> ReLU."""

    def __init__(self, obs_dim, action_dim, hidden, obs_mean, obs_std):
        super().__init__()
        self.obs_mean, self.obs_std = obs_mean, obs_std
        layers, n_in = [], obs_dim + action_dim
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU()]
            n_in = n_out
        layers.append(nn.Linear(n_in, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs, action):
        obs = (obs - self.obs_mean) / self.obs_std      # normalize the observation
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class SquashedGaussianActor(Actor[StochasticAction]):
    """SAC's actor a Gaussian whose sample goes through tanh,
    so that actions are in [-1, 1]."""

    def __init__(self, obs_dim, action_dim, hidden, obs_mean, obs_std):
        super().__init__()
        self.obs_mean, self.obs_std = obs_mean, obs_std
        self.backbone = build_mlp([obs_dim, *hidden], output_activation=nn.ReLU())
        self.mean_layer = nn.Linear(hidden[-1], action_dim)
        self.std_layer = nn.Linear(hidden[-1], action_dim)
        self.tanh = TanhTransform(cache_size=1)

    def dist(self, obs):
        features = self.backbone((obs - self.obs_mean) / self.obs_std)
        mean = self.mean_layer(features)
        std = F.softplus(self.std_layer(features)) + 1e-4
        return TransformedDistribution(Independent(Normal(mean, std), 1), [self.tanh])

    def forward(self, obs):
        """Training: a random action, and its log-probability"""
        dist = self.dist(obs)
        action = dist.rsample()     # reparametrization trick: gradients flow through the sample
        log_prob = dist.log_prob(action)
        self.tanh._cached_x_y = [None, None]
        return StochasticAction(value=action, log_prob=log_prob)

    def act(self, obs):
        """Evaluation: the deterministic action (mean of the Gaussian, through tanh)"""
        features = self.backbone((obs - self.obs_mean) / self.obs_std)
        return torch.tanh(self.mean_layer(features))

# Expert data

def load_expert_buffer(path) -> ReplayBuffer:
    """The expert laps (collect_expert.py) in a replay buffer.
    SAC stores actions with their log-probability; the expert has none, so we
    store 0 (the SAC losses never use the stored log-probability)."""
    d = torch.load(path, weights_only=False)
    transitions = Transitions(
        obs=d["obs"].float(),
        action=StochasticAction(value=d["action"].float(), log_prob=torch.zeros(len(d["action"]))),
        reward=d["reward"].float(),
        next_obs=d["next_obs"].float(),
        terminated=d["terminated"].bool(),
    )
    buffer = ReplayBuffer(len(transitions))
    buffer.add(transitions)
    return buffer

# Evaluation: one lap with the deterministic actor

def evaluate(actor, env) -> dict:
    obs, info = env.reset(seed=0)
    total_reward, done = 0.0, False
    while not done:
        with torch.no_grad():
            action = actor.act(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return {"finished": bool(terminated), "lap_steps": info["lap_steps"], "return": total_reward}

# Training

def run_rlpd(cfg: Config) -> list[dict]:
    torch.manual_seed(cfg.seed)
    env = VecEnv("stk-solo-v0", 1, seed=cfg.seed)   # training environment
    eval_env = SoloTrackEnv()                         # evaluation environment

    # Expert data, and the normalization statistics of the observations
    expert_buffer = load_expert_buffer(cfg.dataset)
    expert_obs = expert_buffer.sample(len(expert_buffer)).obs
    obs_mean, obs_std = expert_obs.mean(0), expert_obs.std(0) + 1e-3

    # Actor, two critics and their targets (as in TD3)
    dims = (env.observation_dim, env.action_dim, cfg.hidden, obs_mean, obs_std)
    actor = SquashedGaussianActor(*dims)
    critic_1, critic_2 = Critic(*dims), Critic(*dims)
    target_critic_1, target_critic_2 = copy.deepcopy(critic_1), copy.deepcopy(critic_2)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(
        list(critic_1.parameters()) + list(critic_2.parameters()), lr=cfg.lr)

    # Entropy coefficient alpha, tuned so that the policy keeps some randomness
    log_alpha = torch.tensor([np.log(cfg.init_alpha)], requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.lr)
    target_entropy = -float(env.action_dim)

    collector = TransitionCollector(env, actor)    # SAC: no added noise, the actor is random
    online_buffer = ReplayBuffer(1_000_000)
    n_expert = int(cfg.offline_ratio * cfg.batch_size)
    n_online = cfg.batch_size - n_expert

    history, start = [], time.time()
    while collector.steps < cfg.max_steps:
        # Evaluate from time to time
        if collector.steps % cfg.eval_interval == 0:
            result = {"step": collector.steps, **evaluate(actor, eval_env)}
            history.append(result)
            print(f"step {result['step']:6d}  finished {result['finished']}  "
                  f"lap {result['lap_steps']:4d} steps  return {result['return']:7.1f}  "
                  f"({time.time() - start:.0f}s)", flush=True)

        # Drive one step and store it
        online_buffer.add(collector.collect(1))
        if len(online_buffer) < cfg.learning_starts:
            continue

        # 1. SYMMETRIC SAMPLING: half online data, half expert data
        if n_expert > 0:
            batch = Transitions.cat([online_buffer.sample(n_online), expert_buffer.sample(n_expert)])
        else:
            batch = online_buffer.sample(cfg.batch_size)
        alpha = log_alpha.exp().item()

        # 2. Critic update: target = r + gamma * (min of target critics - alpha * log pi)
        with torch.no_grad():
            next_action = actor(batch.next_obs)          # sampled from the current actor
            next_q = torch.min(target_critic_1(batch.next_obs, next_action.value),
                               target_critic_2(batch.next_obs, next_action.value))
            next_q = next_q - alpha * next_action.log_prob
            target = batch.reward + cfg.gamma * (~batch.terminated).float() * next_q
        critic_loss = F.mse_loss(critic_1(batch.obs, batch.action.value), target) \
                    + F.mse_loss(critic_2(batch.obs, batch.action.value), target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 3. Actor update: maximize Q(s, a) - alpha * log pi(a|s), with a ~ pi(.|s)
        action = actor(batch.obs)
        q = (critic_1(batch.obs, action.value) + critic_2(batch.obs, action.value)) / 2
        actor_loss = (alpha * action.log_prob - q).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 4. Entropy coefficient update
        alpha_loss = -(log_alpha * (action.log_prob + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # 5. Target critics follow the critics slowly
        soft_update(critic_1, target_critic_1, cfg.tau)
        soft_update(critic_2, target_critic_2, cfg.tau)

    history.append({"step": collector.steps, **evaluate(actor, eval_env)})
    print(f"final: {history[-1]}", flush=True)
    eval_env.close()
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=Config.max_steps)
    parser.add_argument("--offline-ratio", type=float, default=Config.offline_ratio)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    torch.set_num_threads(1)   # one core per run, so that several runs can run in parallel
    cfg = Config(seed=args.seed, max_steps=args.steps, offline_ratio=args.offline_ratio)
    name = f"{'rlpd' if cfg.offline_ratio > 0 else 'sac'}-s{cfg.seed}"

    history = run_rlpd(cfg)

    path = Path(args.outdir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "evaluations": history}, indent=1))
    print(f"Results saved in {path}")