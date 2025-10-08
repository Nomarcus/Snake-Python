"""PPO training script offering an alternative to Marcus Petersson's DQN agent."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.util import make_vec_env

from snake_env import SnakeEnv

ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


class EpisodeTracker(BaseCallback):
    """Shared callback for PPO mirroring console telemetry from the web UI."""

    def __init__(self, log_interval: int = 10_000):
        super().__init__(verbose=1)
        self.log_interval = log_interval
        self.rewards: list[float] = []
        self.lengths: list[int] = []
        self.fruits: list[int] = []
        self.last_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.lengths.append(info["episode"]["l"])
                self.fruits.append(info.get("fruits", 0))
                self.logger.record("rollout/fruits", info.get("fruits", 0))
                colour = ANSI_GREEN if info["episode"]["r"] > 0 else ANSI_YELLOW
                print(
                    f"{colour}Episode | Reward: {info['episode']['r']:.2f} | "
                    f"Length: {info['episode']['l']} | Fruits: {info.get('fruits', 0)}{ANSI_RESET}"
                )

        if self.num_timesteps - self.last_log >= self.log_interval:
            self.last_log = self.num_timesteps
            if self.rewards:
                avg_r = float(np.mean(self.rewards[-20:]))
                avg_l = float(np.mean(self.lengths[-20:]))
                avg_f = float(np.mean(self.fruits[-20:]))
                self.logger.record("snake/avg_reward_20", avg_r)
                self.logger.record("snake/avg_length_20", avg_l)
                self.logger.record("snake/avg_fruits_20", avg_f)
                print(
                    f"{ANSI_GREEN}Step {self.num_timesteps:,} | Avg Reward (20 ep): {avg_r:.2f} | "
                    f"Avg Len: {avg_l:.1f} | Avg Fruits: {avg_f:.2f}{ANSI_RESET}"
                )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO Snake agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=8,
        help="Number of simultaneous environments to train on",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def build_vector_env(grid_size: int, n_envs: int, seed: int, render_first_env: bool) -> VecEnv:
    env_kwargs = {"grid_size": grid_size, "render_mode": None, "show_window": False}
    vec_env = make_vec_env(
        lambda: SnakeEnv(**env_kwargs),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=None,
        vec_env_cls=DummyVecEnv,
    )
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

    if args.tensorboard:
        if args.run_name:
            tb_path = Path("snakepython") / "tb_snake" / args.run_name
        else:
            tb_path = Path("snakepython") / "tb_snake"
        tb_path.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(tb_path)
    else:
        tensorboard_log = None

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        gamma=0.975,
        gae_lambda=0.92,
        clip_range=0.2,
        n_steps=2048,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tensorboard_log,
    )

    callback = EpisodeTracker()

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    finally:
        if args.run_name:
            filename = f"ppo_snake_{args.run_name}.zip"
        else:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ppo_snake_{timestamp}.zip"
        model_path = models_dir / filename
        model.save(model_path)
        print(f"Saved PPO model to {model_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
