"""SuperTuxKart environment for the mini-project (DAgger vs symmetric sampling).

One kart alone on one track for one lap, with:
- observation: a fixed-size vector of track features (see `flatten_obs`);
- action: [acceleration, steer] in [-1, 1]^2 (see `to_stk` / `from_stk`);
- reward: the built-in reward of pystk2_gymnasium (progress / 10 - 0.1 per step,
  +10 at the finish line);
- the raw observation dict in `info["raw_obs"]`, for the expert.

Registered as "stk-solo-v0":  rl_mind.env.VecEnv("stk-solo-v0", n_envs, track="abyss")
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pystk2_gymnasium
from gymnasium import spaces
from pystk2_gymnasium import AgentSpec

# The expert agents live in the course repo, cloned into external/
COURSE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "pystk_project")
sys.path.append(os.path.join(COURSE_REPO, "src"))

N_PATHS = 5  # number of track segments ahead kept in the observation
OBS_DIM = 3 + 1 + 3 + 1 + 3 * N_PATHS + 3 * N_PATHS + N_PATHS  # = 43
ACTION_DIM = 2


# Observation (step 3)

def pad(rows, n, width):
    # Keep the first n rows of a list of vectors, fill with zeros if fewer
    out = np.zeros((n, width), dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, width)[:n]
    out[: len(rows)] = rows
    return out


def flatten_obs(obs: dict) -> np.ndarray:
    # Observation dict -> fixed-size vector (track features only)
    parts = [
        np.ravel(obs["velocity"]),                      # 3: speed vector (kart frame)
        np.ravel(obs["center_path_distance"]),          # 1: signed distance to the center line
        np.ravel(obs["center_path"]),                   # 3: vector to the center line
        np.ravel(obs["max_steer_angle"]),               # 1: steering limit at this speed
        pad(obs["paths_start"], N_PATHS, 3).ravel(),    # 15: start points of next segments
        pad(obs["paths_end"], N_PATHS, 3).ravel(),      # 15: end points of next segments
        pad(obs["paths_width"], N_PATHS, 1).ravel(),    # 5: widths of next segments
    ]
    return np.concatenate(parts).astype(np.float32)


# Actions (step 4)


def to_stk(a: np.ndarray) -> dict:
    # Our action [acceleration, steer] in [-1, 1]^2  ->  STK action dict
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return {
        "acceleration": np.array([(a[0] + 1.0) / 2.0], dtype=np.float32),  # [-1, 1] -> [0, 1]
        "steer": np.array([a[1]], dtype=np.float32),
        "brake": 0, "drift": 0, "fire": 0, "nitro": 0, "rescue": 0,      # disabled
    }


def from_stk(action: dict) -> np.ndarray:
    # STK action dict (e.g. from the expert)  ->  our action in [-1, 1]^2
    acceleration = float(np.ravel(action["acceleration"])[0])
    steer = float(np.ravel(action["steer"])[0])
    return np.clip(np.array([2.0 * acceleration - 1.0, steer], dtype=np.float32), -1.0, 1.0)


# The environment (step 5)


class SoloTrackEnv(gym.Env):
    # One kart, alone, one lap; vector observation, 2D continuous action

    def __init__(self, track: str = "abyss", max_steps: int = 1500):
        super().__init__()
        self.stk = gym.make(
            "supertuxkart/full-v0",
            agent=AgentSpec(use_ai=False, rank_start=0),
            track=track, num_kart=1, laps=1,
            max_episode_steps=max_steps,
            disable_env_checker=True,
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw, info = self.stk.reset(seed=seed, options=options)
        self.steps = 0
        return flatten_obs(raw), {**info, "raw_obs": raw}

    def step(self, action):
        raw, reward, terminated, truncated, info = self.stk.step(to_stk(action))
        self.steps += 1
        info = {**info, "raw_obs": raw, "lap_steps": self.steps}
        return flatten_obs(raw), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.stk.close()


def make_expert(env: SoloTrackEnv):
    # The team4 expert, as a function: raw observation dict -> our 2D action
    from agents.team4.agent4 import Agent4

    agent = Agent4(env.stk.unwrapped, path_lookahead=3)
    return lambda raw_obs: from_stk(agent.choose_action(raw_obs))


gym.register("stk-solo-v0", entry_point=SoloTrackEnv)