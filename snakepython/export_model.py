"""Export Stable-Baselines3 Snake agents to ONNX + JSON for the web replay UI.

Web integration (add to Marcus' ``Watch`` screen)::

    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <button id="loadPythonModelBtn">🧠 Load Trained Model (Python)</button>

    const session = await ort.InferenceSession.create('export/snake_agent.onnx');
    const input = new ort.Tensor('float32', gridData, [1, 3, gridSize, gridSize]);
    const output = await session.run({ input });
    const action = output.action.data[0];

Persist the selected agent source in ``localStorage`` so that Marcus can toggle
between "Browser Agent" and "Python Model (ONNX)" at runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import onnx
import torch

try:
    from stable_baselines3 import DQN, PPO
except ImportError as exc:  # pragma: no cover - safety net for missing deps
    raise SystemExit("stable-baselines3 must be installed to export models") from exc


class DQNOnnxWrapper(torch.nn.Module):
    """Torch module that mimics the TensorFlow.js action selection logic."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q_values = self.policy.q_net(obs)
        action = torch.argmax(q_values, dim=1, keepdim=True)
        return action.to(torch.int64)


class PPOOnnxWrapper(torch.nn.Module):
    """Torch module returning greedy actions from the PPO actor network."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        actions, _, _ = self.policy.forward(obs, deterministic=True)
        if actions.dtype != torch.int64:
            actions = torch.argmax(actions, dim=1, keepdim=True)
        return actions.to(torch.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SB3 Snake models to ONNX")
    parser.add_argument("model", type=Path, help="Path to the trained .zip model")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("snakepython/export"))
    return parser.parse_args()


def load_model(path: Path, algo: str):
    if algo == "dqn":
        return DQN.load(path)
    if algo == "ppo":
        return PPO.load(path)
    raise ValueError(f"Unsupported algorithm: {algo}")


def export_to_onnx(model, algo: str, grid_size: int, output_path: Path) -> None:
    dummy = torch.zeros((1, 3, grid_size, grid_size), dtype=torch.float32)

    if algo == "dqn":
        module = DQNOnnxWrapper(model.policy)
    else:
        module = PPOOnnxWrapper(model.policy)

    module.eval()
    module.to(torch.device("cpu"))
    dummy = dummy.to(torch.device("cpu"))

    torch.onnx.export(
        module,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "action": {0: "batch"}},
    )

    model_proto = onnx.load(output_path)
    meta_entries = {
        "grid_size": str(grid_size),
        "policy_type": model.policy.__class__.__name__,
        "agent_type": algo.upper(),
    }
    model_proto.metadata_props.clear()
    for key, value in meta_entries.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model_proto, output_path)


def export_to_json(model, algo: str, grid_size: int, json_path: Path) -> None:
    state_dict = model.policy.state_dict()
    serialised: Dict[str, list] = {name: tensor.cpu().numpy().tolist() for name, tensor in state_dict.items()}
    payload = {
        "meta": {
            "grid_size": grid_size,
            "policy_type": model.policy.__class__.__name__,
            "agent_type": algo.upper(),
        },
        "state_dict": serialised,
    }
    json_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    model = load_model(args.model, args.algo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = args.output_dir / "snake_agent.onnx"
    json_path = args.output_dir / "snake_agent.json"

    export_to_onnx(model, args.algo, args.grid_size, onnx_path)
    export_to_json(model, args.algo, args.grid_size, json_path)

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Exported JSON weights to {json_path}")
    print(
        "Web integration hint: include onnxruntime-web and load 'export/snake_agent.onnx' "
        "from Marcus' Watch screen using the provided button callback."
    )


if __name__ == "__main__":
    main()
