"""Utility to launch multiple Snake training jobs in parallel processes."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def stream_output(process: subprocess.Popen, prefix: str) -> None:
    """Forward child stdout to the console with a helpful prefix."""

    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}")
    process.stdout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple Snake training runs")
    parser.add_argument("--runs", type=int, default=4, help="Number of parallel runs")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn", help="Algorithm to train")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Timesteps per run")
    parser.add_argument("--grid-size", type=int, default=15, help="Environment grid size")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--base-seed", type=int, default=100, help="Seed offset for reproducibility")
    parser.add_argument("--headless", action="store_true", help="Disable pygame windows in all runs")
    return parser.parse_args()


def build_command(args: argparse.Namespace, run_index: int) -> List[str]:
    script = ROOT / f"train_{args.algo}.py"
    run_name = f"run{run_index}"
    seed = args.base_seed + run_index

    command = [
        sys.executable,
        str(script),
        "--timesteps",
        str(args.timesteps),
        "--grid-size",
        str(args.grid_size),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]

    if args.tensorboard:
        command.append("--tensorboard")
    if args.headless:
        command.append("--headless")

    return command


def main() -> None:
    args = parse_args()
    processes: List[subprocess.Popen] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for run in range(1, args.runs + 1):
        command = build_command(args, run)
        print(f"Starting run {run}: {' '.join(command)}")
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        processes.append(proc)
        Thread(target=stream_output, args=(proc, f"{args.algo.upper()}-{run}"), daemon=True).start()

    exit_code = 0
    for run, proc in enumerate(processes, start=1):
        ret = proc.wait()
        if ret != 0:
            print(f"Run {run} exited with code {ret}")
            exit_code = ret
        else:
            print(f"Run {run} completed successfully")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
