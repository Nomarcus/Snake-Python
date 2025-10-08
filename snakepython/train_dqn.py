"""DQN training entry point that mirrors Marcus Petersson's HTML Snake-ML loop.

Key features carried over from the original browser implementation:
* Identical reward shaping via :class:`snake_env.SnakeEnv`.
* Real-time pygame rendering for the first vectorised environment while the
  remaining environments run silently in the background.
* Periodic logging of reward, episode length and fruits collected.
* Optional TensorBoard summaries at ``./tb_snake`` to match the JavaScript UI.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.util import make_vec_env

from snake_env import SnakeEnv

ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


class EpisodeTracker(BaseCallback):
    """Collect metrics and print Marcus-style coloured console updates."""

    def __init__(self, log_interval: int = 10_000):
        super().__init__(verbose=1)
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_fruits: List[int] = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_fruits.append(info.get("fruits", 0))
                self.logger.record("rollout/fruits", info.get("fruits", 0))
                coloured = f"{ANSI_GREEN if info['episode']['r'] > 0 else ANSI_YELLOW}"  # noqa: E501
                print(
                    f"{coloured}Episode | Reward: {info['episode']['r']:.2f} | "
                    f"Length: {info['episode']['l']} | Fruits: {info.get('fruits', 0)}{ANSI_RESET}"
                )

        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            if self.episode_rewards:
                mean_r = float(np.mean(self.episode_rewards[-20:]))
                mean_l = float(np.mean(self.episode_lengths[-20:]))
                mean_f = float(np.mean(self.episode_fruits[-20:]))
                self.logger.record("snake/avg_reward_20", mean_r)
                self.logger.record("snake/avg_length_20", mean_l)
                self.logger.record("snake/avg_fruits_20", mean_f)
                print(
                    f"{ANSI_GREEN}Step {self.num_timesteps:,} | Avg Reward (20 ep): {mean_r:.2f} | "
                    f"Avg Len: {mean_l:.1f} | Avg Fruits: {mean_f:.2f}{ANSI_RESET}"
                )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN Snake agent")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--grid-size", type=int, default=15, help="Snake grid size")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=8,
        help="Number of simultaneous environments to train on",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom run identifier used for saved models and logs")
    parser.add_argument("--headless", action="store_true", help="Disable pygame rendering (useful for batch jobs)")
    return parser.parse_args()


def build_vector_env(grid_size: int, n_envs: int, seed: int, render_first_env: bool) -> VecEnv:
    """Create an 8-env vector with optional pygame rendering for env[0]."""

    env_kwargs = {"grid_size": grid_size, "render_mode": None, "show_window": False}
    vec_env = make_vec_env(
        lambda: SnakeEnv(**env_kwargs),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=None,
        vec_env_cls=DummyVecEnv,
    )

    # Activate rendering for the first environment only to avoid pygame clashes.
    if render_first_env:
        vec_env.env_method("set_rendering", render_mode="human", show_window=True, indices=[0])
    return vec_env


def main() -> None:
    args = parse_args()

    models_dir = Path("snakepython") / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, int(args.parallel_envs))
    vec_env = build_vector_env(
        args.grid_size,
        n_envs=n_envs,
        seed=args.seed,
        render_first_env=not args.headless,
    )

    policy_kwargs = dict(net_arch=[256, 256])
    if args.tensorboard:
        if args.run_name:
            tb_path = Path("snakepython") / "tb_snake" / args.run_name
        else:
            tb_path = Path("snakepython") / "tb_snake"
        tb_path.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(tb_path)
    else:
        tensorboard_log = None

    model = DQN(
        "CnnPolicy",
        vec_env,
        verbose=1,
        learning_starts=10_000,
        buffer_size=100_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        batch_size=256,
        target_update_interval=1_000,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    callback = EpisodeTracker()

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=1)
    finally:
        if args.run_name:
            filename = f"dqn_snake_{args.run_name}.zip"
        else:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dqn_snake_{timestamp}.zip"
        model_path = models_dir / filename
        model.save(model_path)
        print(f"Saved model to {model_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
