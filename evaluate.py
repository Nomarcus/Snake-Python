"""Evaluate a trained Snake agent and stream the game via pygame."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO

from snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake model")
    parser.add_argument("model", type=Path, help="Path to the saved SB3 .zip model")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=15)
    return parser.parse_args()


def load_model(path: Path, algo: str):
    if algo == "dqn":
        return DQN.load(path)
    if algo == "ppo":
        return PPO.load(path)
    raise ValueError(f"Unsupported algorithm: {algo}")


def main() -> None:
    args = parse_args()
    model = load_model(args.model, args.algo)

    env = SnakeEnv(grid_size=args.grid_size, render_mode="human", show_window=True)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        fruits = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            fruits = info.get("fruits", fruits)
            steps += 1
            done = terminated or truncated
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Fruits: {fruits} | Steps: {steps}")
    env.close()


if __name__ == "__main__":
    main()
