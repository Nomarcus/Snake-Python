"""Standalone Snake game with optional Double DQN training for Python's IDLE."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

try:  # NumPy krävs för Double DQN-implementationen
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - tydligare fel i IDLE
    raise SystemExit(
        "Det här skriptet kräver NumPy. Installera det via 'pip install numpy' "
        "innan du tränar eller använder autopiloten."
    ) from exc

import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------
GRID_SIZE = 15  # Width and height in tiles
CELL_SIZE = 32  # Pixels per tile
STEP_DELAY = 120  # Milliseconds between snake moves
START_LENGTH = 3

# Reward configuration mirroring the web project defaults


@dataclass
class RewardConfig:
    step_penalty: float = 0.01
    turn_penalty: float = 0.001
    approach_bonus: float = 0.03
    retreat_penalty: float = 0.03
    loop_penalty: float = 0.5
    tight_loop_penalty: float = 1.2
    revisit_penalty: float = 0.05
    dead_end_penalty: float = 0.5
    wall_penalty: float = 10.0
    self_penalty: float = 25.5
    timeout_penalty: float = 5.0
    fruit_reward: float = 10.0
    growth_bonus: float = 1.0
    compact_weight: float = 0.0
    compact_bonus: float = 0.25
    trap_penalty: float = 1.2
    space_gain_bonus: float = 0.05


REWARD_BREAKDOWN_KEYS: Tuple[str, ...] = (
    "stepPenalty",
    "turnPenalty",
    "approachBonus",
    "retreatPenalty",
    "loopPenalty",
    "tightLoopPenalty",
    "revisitPenalty",
    "deadEndPenalty",
    "trapPenalty",
    "spaceGainBonus",
    "fruitReward",
    "growthBonus",
    "compactness",
    "wallPenalty",
    "selfPenalty",
    "timeoutPenalty",
)

VISIT_DECAY = 0.995
LOOP_PATTERNS = {(1, 2, 1, 2), (2, 1, 2, 1)}

Direction = Tuple[int, int]
Point = Tuple[int, int]

DIRECTIONS: Dict[str, Direction] = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}
ACTION_VECTORS: Tuple[Direction, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
OPPOSITE: Dict[Direction, Direction] = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
}
DIRECTION_TO_INDEX: Dict[Direction, int] = {vec: idx for idx, vec in enumerate(ACTION_VECTORS)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def grid_size_state_size(grid_size: int) -> int:
    return grid_size * grid_size * 2 + len(ACTION_VECTORS)


def build_state_vector(
    snake: Iterable[Point],
    fruit: Point,
    direction_index: int,
    grid_size: int,
) -> np.ndarray:
    """Flattened observation used both for training and the Tk autopilot."""

    snake_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    fruit_grid = np.zeros_like(snake_grid)
    for x, y in snake:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            snake_grid[y, x] = 1.0
    fx, fy = fruit
    if 0 <= fx < grid_size and 0 <= fy < grid_size:
        fruit_grid[fy, fx] = 1.0
    direction_one_hot = np.zeros(len(ACTION_VECTORS), dtype=np.float32)
    direction_one_hot[direction_index] = 1.0
    return np.concatenate((snake_grid.flatten(), fruit_grid.flatten(), direction_one_hot))


# ---------------------------------------------------------------------------
# Double DQN training components
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Basic replay buffer for off-policy training."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)
        self.rng = rng

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = self.rng.sample(self.memory, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)


class MLP:
    """Small fully connected network implemented with NumPy."""

    def __init__(self, layer_sizes: List[int], seed: Optional[int] = None) -> None:
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (in_size + out_size))
            weight = self.rng.uniform(-limit, limit, size=(in_size, out_size)).astype(np.float32)
            bias = np.zeros(out_size, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, inputs: np.ndarray, return_cache: bool = False):
        x = inputs.astype(np.float32)
        activations = [x]
        pre_activations = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            pre_activations.append(x)
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        if return_cache:
            return x, (activations, pre_activations)
        return x

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs, return_cache=False)

    def backward(self, grad_output: np.ndarray, cache, learning_rate: float) -> None:
        activations, pre_activations = cache
        grad = grad_output
        for layer in reversed(range(len(self.weights))):
            a_prev = activations[layer]
            weight = self.weights[layer]
            grad_w = a_prev.T @ grad
            grad_b = grad.sum(axis=0)
            if layer > 0:
                grad = grad @ weight.T
                grad = grad * (pre_activations[layer - 1] > 0)
            self.weights[layer] -= learning_rate * grad_w
            self.biases[layer] -= learning_rate * grad_b

    def copy(self) -> "MLP":
        clone = MLP(self.layer_sizes)
        clone.weights = [weight.copy() for weight in self.weights]
        clone.biases = [bias.copy() for bias in self.biases]
        return clone


class DoubleDQNAgent:
    """Double DQN agent implemented specifically for the IDLE snake grid."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_sync_interval: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers or [128, 128]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.train_steps = 0

        rng_seed = seed if seed is not None else random.randrange(1_000_000)
        self.random = random.Random(rng_seed)
        self.layer_sizes = [state_size, *self.hidden_layers, action_size]
        self.online = MLP(self.layer_sizes, seed=rng_seed)
        self.target = self.online.copy()
        self.buffer = ReplayBuffer(buffer_capacity, self.random)

    # ------------------------------------------------------------------
    # Persistence utilities
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        arrays = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int64),
            "learning_rate": np.array([self.learning_rate], dtype=np.float32),
            "gamma": np.array([self.gamma], dtype=np.float32),
            "epsilon": np.array([self.epsilon], dtype=np.float32),
            "epsilon_min": np.array([self.epsilon_min], dtype=np.float32),
            "epsilon_decay": np.array([self.epsilon_decay], dtype=np.float32),
            "batch_size": np.array([self.batch_size], dtype=np.int64),
            "target_sync": np.array([self.target_sync_interval], dtype=np.int64),
        }
        for idx, weight in enumerate(self.online.weights):
            arrays[f"W{idx}"] = weight
        for idx, bias in enumerate(self.online.biases):
            arrays[f"b{idx}"] = bias
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DoubleDQNAgent":
        data = np.load(Path(path), allow_pickle=False)
        layer_sizes = data["layer_sizes"].astype(np.int64).tolist()
        state_size = int(layer_sizes[0])
        action_size = int(layer_sizes[-1])
        hidden_layers = [int(x) for x in layer_sizes[1:-1]]
        agent = cls(
            state_size,
            action_size,
            hidden_layers=hidden_layers,
            learning_rate=float(data["learning_rate"][0]),
            gamma=float(data["gamma"][0]),
            epsilon_start=float(data["epsilon"][0]),
            epsilon_end=float(data["epsilon_min"][0]),
            epsilon_decay=float(data["epsilon_decay"][0]),
            batch_size=int(data["batch_size"][0]),
            target_sync_interval=int(data["target_sync"][0]),
        )
        for idx in range(len(agent.online.weights)):
            agent.online.weights[idx] = data[f"W{idx}"]
            agent.online.biases[idx] = data[f"b{idx}"]
        agent.target = agent.online.copy()
        return agent

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if state.ndim == 1:
            state = state[None, :]
        if not greedy and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)
        q_values = self.online.predict(state)
        if q_values.ndim == 2:
            q_values = q_values[0]
        return int(np.argmax(q_values))

    def push(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target = self.online.copy()

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states = batch.state
        actions = batch.action
        rewards = batch.reward
        next_states = batch.next_state
        dones = batch.done

        q_values, cache = self.online.forward(states, return_cache=True)
        q_next_online = self.online.predict(next_states)
        q_next_target = self.target.predict(next_states)
        next_actions = np.argmax(q_next_online, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * q_next_target[np.arange(self.batch_size), next_actions]

        diff = q_values[np.arange(self.batch_size), actions] - targets
        loss = float(np.mean(diff ** 2) * 0.5)
        grad_output = np.zeros_like(q_values)
        grad_output[np.arange(self.batch_size), actions] = diff / self.batch_size
        self.online.backward(grad_output, cache, self.learning_rate)

        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self.update_target()
        return loss


class IdleSnakeEnv:
    """Lightweight snake environment for headless Double DQN training."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        seed: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.grid_size = grid_size
        self.random = random.Random(seed)
        self.reward_config = reward_config or RewardConfig()
        self.state = np.zeros(grid_size_state_size(grid_size), dtype=np.float32)
        self.total_cells = grid_size * grid_size
        self.snake: List[Point] = []
        self.snake_set: Set[Point] = set()
        self.direction_index: int = 1
        self.fruit: Point = (0, 0)
        self.pending_growth = 0
        self.steps_since_fruit = 0
        self.total_reward = 0.0
        self.max_length = START_LENGTH
        self.prev_slack = 0.0
        self.last_slack_delta = 0.0
        self.last_free_space_ratio = 1.0
        self.relative_history: Deque[int] = deque(maxlen=6)
        self.head_history: Deque[Point] = deque(maxlen=12)
        self.freedom_history: Deque[float] = deque(maxlen=20)
        self.visit_map: List[List[float]] = []
        self.episode_breakdown = self._make_reward_breakdown()
        self.steps_taken = 0
        self.fruits_eaten = 0
        self.reward_breakdown = self._make_reward_breakdown()

    @property
    def state_size(self) -> int:
        return grid_size_state_size(self.grid_size)

    def _make_reward_breakdown(self) -> Dict[str, float]:
        breakdown = {key: 0.0 for key in REWARD_BREAKDOWN_KEYS}
        breakdown["total"] = 0.0
        return breakdown

    def _decay_visits(self) -> None:
        if not self.visit_map:
            return
        for y in range(self.grid_size):
            row = self.visit_map[y]
            for x in range(self.grid_size):
                row[x] *= VISIT_DECAY

    def _relative_action(self, previous: int, current: int) -> int:
        if current == previous:
            return 0
        if current == (previous - 1) % len(ACTION_VECTORS):
            return 1  # left turn
        if current == (previous + 1) % len(ACTION_VECTORS):
            return 2  # right turn
        return 0

    def _free_space_from(self, start: Point, tail_will_move: bool) -> int:
        blocked = set(self.snake_set)
        blocked.discard(start)
        if tail_will_move and self.snake:
            blocked.discard(self.snake[-1])
        seen: Set[Point] = set()
        queue: List[Point] = [start]
        while queue:
            x, y = queue.pop()
            if (x, y) in seen or (x, y) in blocked:
                continue
            seen.add((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    queue.append((nx, ny))
            if len(seen) > self.total_cells:
                break
        return len(seen)

    def _compute_slack(self) -> float:
        if not self.snake:
            return 0.0
        xs = [segment[0] for segment in self.snake]
        ys = [segment[1] for segment in self.snake]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        area = width * height
        return max(0.0, float(area - len(self.snake)))

    def reset(self) -> np.ndarray:
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.snake = [(start_x - i, start_y) for i in range(START_LENGTH)]
        self.snake_set = set(self.snake)
        self.direction_index = 1
        self.pending_growth = 0
        self.steps_since_fruit = 0
        self.total_reward = 0.0
        self.steps_taken = 0
        self.fruits_eaten = 0
        self.max_length = len(self.snake)
        self.prev_slack = self._compute_slack()
        self.last_slack_delta = 0.0
        self.last_free_space_ratio = 1.0
        self.relative_history.clear()
        self.head_history.clear()
        self.freedom_history.clear()
        self.visit_map = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.episode_breakdown = self._make_reward_breakdown()
        self.reward_breakdown = self._make_reward_breakdown()
        self._spawn_fruit()
        self.state = build_state_vector(self.snake, self.fruit, self.direction_index, self.grid_size)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if action < 0 or action >= len(ACTION_VECTORS):
            raise ValueError("Action out of range")

        current_direction = ACTION_VECTORS[self.direction_index]
        chosen_direction = ACTION_VECTORS[action]
        if chosen_direction == OPPOSITE[current_direction]:
            chosen_direction = current_direction
            action = self.direction_index
        new_direction_index = DIRECTION_TO_INDEX[chosen_direction]
        relative_action = self._relative_action(self.direction_index, new_direction_index)
        self.direction_index = new_direction_index

        head_x, head_y = self.snake[0]
        dx, dy = chosen_direction
        new_head = (head_x + dx, head_y + dy)

        fx, fy = self.fruit
        prev_distance = abs(head_x - fx) + abs(head_y - fy)
        will_grow = new_head == self.fruit or self.pending_growth > 0
        hits_wall = not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)
        body_to_check = self.snake if will_grow else self.snake[:-1]
        hits_self = new_head in body_to_check
        space_before = self._free_space_from(self.snake[0], tail_will_move=self.pending_growth == 0)

        reward = 0.0
        done = False
        cause = "step"
        step_breakdown = self._make_reward_breakdown()

        if hits_wall or hits_self:
            penalty = self.reward_config.wall_penalty if hits_wall else self.reward_config.self_penalty
            reward -= penalty
            key = "wallPenalty" if hits_wall else "selfPenalty"
            step_breakdown[key] -= penalty
            cause = "wall" if hits_wall else "self"
            done = True
        else:
            self._decay_visits()
            self.snake.insert(0, new_head)
            self.snake_set.add(new_head)
            ate_fruit = new_head == self.fruit
            if ate_fruit:
                self.pending_growth += 1
                self.fruits_eaten += 1
                reward += self.reward_config.fruit_reward
                step_breakdown["fruitReward"] += self.reward_config.fruit_reward
                self.steps_since_fruit = 0
                self._spawn_fruit()
                cause = "fruit"
            else:
                self.steps_since_fruit += 1
                if self.pending_growth > 0:
                    self.pending_growth -= 1
                else:
                    tail = self.snake.pop()
                    self.snake_set.discard(tail)

            nx, ny = new_head
            revisit_penalty = self.visit_map[ny][nx] * self.reward_config.revisit_penalty
            if revisit_penalty:
                reward -= revisit_penalty
                step_breakdown["revisitPenalty"] -= revisit_penalty
            self.visit_map[ny][nx] = min(1.0, self.visit_map[ny][nx] + 0.3)

            reward -= self.reward_config.step_penalty
            step_breakdown["stepPenalty"] -= self.reward_config.step_penalty

            if relative_action in (1, 2):
                reward -= self.reward_config.turn_penalty
                step_breakdown["turnPenalty"] -= self.reward_config.turn_penalty

            new_distance = abs(new_head[0] - fx) + abs(new_head[1] - fy)
            if new_distance < prev_distance:
                reward += self.reward_config.approach_bonus
                step_breakdown["approachBonus"] += self.reward_config.approach_bonus
            elif new_distance > prev_distance:
                reward -= self.reward_config.retreat_penalty
                step_breakdown["retreatPenalty"] -= self.reward_config.retreat_penalty

            space_after = self._free_space_from(new_head, tail_will_move=self.pending_growth == 0)
            need = len(self.snake) + 2
            denom = max(1, need)

            if space_after < need:
                penalty = self.reward_config.trap_penalty * (1.0 + (need - space_after) / denom)
                reward -= penalty
                step_breakdown["trapPenalty"] -= penalty
            elif self.reward_config.space_gain_bonus > 0 and space_after > space_before:
                bonus = self.reward_config.space_gain_bonus * min(1.0, (space_after - space_before) / denom)
                reward += bonus
                step_breakdown["spaceGainBonus"] += bonus

            margin = 5
            min_reachable = len(self.snake) + (1 if self.pending_growth > 0 else 0) + margin
            if min_reachable > 0:
                ratio = space_after / max(1, min_reachable)
                base = -0.5 * (1.0 - ratio)
                if base:
                    dead_end_reward = base * self.reward_config.dead_end_penalty
                    reward += dead_end_reward
                    step_breakdown["deadEndPenalty"] += dead_end_reward

            freedom_ratio = max(0.0, min(1.0, space_after / max(1, self.total_cells)))
            self.freedom_history.append(freedom_ratio)
            if len(self.freedom_history) > self.freedom_history.maxlen:
                self.freedom_history.popleft()
            avg_freedom = sum(self.freedom_history) / len(self.freedom_history)
            trend = freedom_ratio - (self.freedom_history[0] if self.freedom_history else freedom_ratio)

            if trend < -0.05 and avg_freedom < 0.15:
                long_term_penalty = -self.reward_config.trap_penalty * abs(trend) * 4.0
                reward += long_term_penalty
                step_breakdown["trapPenalty"] += long_term_penalty
            if trend > 0.03 and avg_freedom > 0.2:
                long_term_bonus = self.reward_config.compact_bonus * trend * 5.0
                reward += long_term_bonus
                step_breakdown["compactness"] += long_term_bonus

            prev_ratio = self.last_free_space_ratio
            drop = max(0.0, prev_ratio - freedom_ratio)
            penalty_factor = 0.0
            if len(self.snake) > 4 and new_head in self.head_history:
                penalty_factor += 1.0
            if drop > 0.0:
                penalty_factor += min(1.5, drop * 12.0)
            if penalty_factor > 0.0 and self.reward_config.tight_loop_penalty != 0.0:
                loop_penalty = self.reward_config.tight_loop_penalty * penalty_factor
                reward -= loop_penalty
                step_breakdown["tightLoopPenalty"] -= loop_penalty
            self.last_free_space_ratio = freedom_ratio

            self.head_history.append(new_head)
            if len(self.head_history) > self.head_history.maxlen:
                self.head_history.popleft()

            self.relative_history.append(relative_action)
            if len(self.relative_history) > self.relative_history.maxlen:
                self.relative_history.popleft()
            if len(self.relative_history) >= 4:
                last_four = tuple(list(self.relative_history)[-4:])
                if last_four in LOOP_PATTERNS:
                    reward -= self.reward_config.loop_penalty
                    step_breakdown["loopPenalty"] -= self.reward_config.loop_penalty

            slack = self._compute_slack()
            slack_delta = self.prev_slack - slack
            if self.reward_config.compact_weight != 0.0 and slack_delta != 0.0:
                compact_reward = slack_delta * self.reward_config.compact_weight
                reward += compact_reward
                step_breakdown["compactness"] += compact_reward
            self.last_slack_delta = slack_delta
            self.prev_slack = slack

            if len(self.snake) > self.max_length:
                gain = len(self.snake) - self.max_length
                growth_bonus = self.reward_config.growth_bonus * gain
                reward += growth_bonus
                step_breakdown["growthBonus"] += growth_bonus
                self.max_length = len(self.snake)

            if self.steps_since_fruit > self.total_cells * 2:
                reward -= self.reward_config.timeout_penalty
                step_breakdown["timeoutPenalty"] -= self.reward_config.timeout_penalty
                done = True
                cause = "timeout"

        step_breakdown["total"] = reward
        self.total_reward += reward
        self.steps_taken += 1

        for key, value in step_breakdown.items():
            if key == "total":
                continue
            self.episode_breakdown[key] += value
        self.episode_breakdown["total"] += reward
        self.reward_breakdown = {key: (self.episode_breakdown[key] if key != "total" else self.episode_breakdown["total"]) for key in self.episode_breakdown}

        if not done:
            next_state = build_state_vector(self.snake, self.fruit, self.direction_index, self.grid_size)
            self.state = next_state.copy()
        else:
            next_state = self.state.copy()

        info: Dict[str, object] = {
            "cause": cause,
            "breakdown": step_breakdown,
            "fruits": self.fruits_eaten,
            "steps": self.steps_taken,
        }
        return next_state, reward, done, info

    def _spawn_fruit(self) -> None:
        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake_set
        ]
        self.fruit = self.random.choice(free_cells) if free_cells else self.snake[0]

    def snapshot(self) -> Dict[str, object]:
        """Return a shallow copy of the current game state for visualisering."""

        return {
            "snake": list(self.snake),
            "fruit": self.fruit,
            "direction_index": self.direction_index,
            "pending_growth": self.pending_growth,
        }


# ---------------------------------------------------------------------------
# Tkinter score tracking and rendering
# ---------------------------------------------------------------------------


@dataclass
class Score:
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    fruits: int = 0
    steps: int = 0
    reward: float = 0.0

    def reset(self) -> None:
        self.fruits = 0
        self.steps = 0
        self.reward = 0.0

    def apply_step(self, outcome: str) -> None:
        self.steps += 1
        self.reward -= self.reward_config.step_penalty
        if outcome == "fruit":
            self.fruits += 1
            self.reward += self.reward_config.fruit_reward
        elif outcome == "wall":
            self.reward -= self.reward_config.wall_penalty
        elif outcome == "self":
            self.reward -= self.reward_config.self_penalty


class SnakeCanvas(tk.Canvas):
    """Canvas widget that hosts the actual snake game."""

    def __init__(self, master: tk.Misc, status: tk.Label, agent: Optional[DoubleDQNAgent] = None, autopilot: bool = False) -> None:
        pixel_size = GRID_SIZE * CELL_SIZE
        super().__init__(
            master,
            width=pixel_size,
            height=pixel_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.pack()
        self.status_label = status
        self.score = Score()
        self._running = False
        self._after_id: Optional[str] = None
        self.agent = agent
        self.autopilot = autopilot and agent is not None
        self.reset()
        self.focus_set()
        self.bind("<KeyPress>", self.on_key_press)

    def reset(self) -> None:
        self.delete("all")
        start_x = GRID_SIZE // 2
        start_y = GRID_SIZE // 2
        self.snake = [(start_x - i, start_y) for i in range(START_LENGTH)]
        self.direction = (1, 0)
        self.score.reset()
        self._running = True
        self.spawn_fruit()
        self.draw_frame()
        self.update_status("start")

    def spawn_fruit(self) -> None:
        free_cells = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in self.snake
        ]
        self.fruit = random.choice(free_cells) if free_cells else self.snake[0]

    # Input handling -----------------------------------------------------
    def on_key_press(self, event: tk.Event[tk.Misc]) -> None:
        if event.keysym == "space":
            self.reset()
            return
        if event.keysym == "Escape":
            self.quit_game()
            return
        if event.keysym.lower() == "a":
            self.toggle_autopilot()
            return
        if event.keysym not in DIRECTIONS:
            return
        new_dir = DIRECTIONS[event.keysym]
        if OPPOSITE.get(self.direction) == new_dir:
            return
        self.direction = new_dir

    def toggle_autopilot(self) -> None:
        if self.agent is None:
            print("Ingen tränad agent laddad. Använd --load-model för att aktivera autopilot.")
            return
        self.autopilot = not self.autopilot
        mode = "på" if self.autopilot else "av"
        print(f"Autopilot är nu {mode}.")
        self.update_status("step")

    def quit_game(self) -> None:
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self.master.destroy()

    # Rendering helpers --------------------------------------------------
    def draw_frame(self) -> None:
        self.delete("all")
        self.draw_grid()
        self.draw_snake()
        self.draw_fruit()

    def draw_grid(self) -> None:
        pixel_size = GRID_SIZE * CELL_SIZE
        for row in range(GRID_SIZE):
            color = "#1f1f1f" if row % 2 == 0 else "#232323"
            self.create_rectangle(
                0,
                row * CELL_SIZE,
                pixel_size,
                (row + 1) * CELL_SIZE,
                fill=color,
                outline=color,
            )

    def draw_snake(self) -> None:
        for index, (x, y) in enumerate(self.snake):
            fill = "#32c85c" if index == 0 else "#58e279"
            self._draw_cell(x, y, fill)

    def draw_fruit(self) -> None:
        fx, fy = self.fruit
        self._draw_cell(fx, fy, "#ff5c5c")

    def _draw_cell(self, x: int, y: int, color: str) -> None:
        self.create_rectangle(
            x * CELL_SIZE + 2,
            y * CELL_SIZE + 2,
            (x + 1) * CELL_SIZE - 2,
            (y + 1) * CELL_SIZE - 2,
            fill=color,
            outline="",
        )

    # Game loop ----------------------------------------------------------
    def start(self) -> None:
        self.after(300, self.tick)

    def tick(self) -> None:
        if not self._running:
            return
        if self.autopilot and self.agent is not None:
            direction_idx = DIRECTION_TO_INDEX[self.direction]
            state = build_state_vector(self.snake, self.fruit, direction_idx, GRID_SIZE)
            action = self.agent.select_action(state, greedy=True)
            chosen_direction = ACTION_VECTORS[action]
            if OPPOSITE[self.direction] != chosen_direction:
                self.direction = chosen_direction
        outcome = self.advance_snake()
        self.draw_frame()
        self.score.apply_step(outcome)
        self.update_status(outcome)
        if outcome in {"wall", "self"}:
            self._running = False
            self.update_status(outcome, finished=True)
            return
        self._after_id = self.after(STEP_DELAY, self.tick)

    def update_status(self, outcome: str, finished: bool = False) -> None:
        if outcome == "fruit":
            prefix = "🍎 Frukt!"
        elif outcome == "wall":
            prefix = "💥 Krockade med väggen."
        elif outcome == "self":
            prefix = "💥 Åt sig själv."
        elif outcome == "start":
            prefix = "🐍 Nystart."
        else:
            prefix = "🐍"
        suffix = " Tryck Space för att börja om." if finished else ""
        autopilot_text = " 🤖" if self.autopilot and self.agent is not None else ""
        self.status_label.configure(
            text=(
                f"{prefix}{autopilot_text} Poäng: {self.score.reward:.0f}  "
                f"Frukter: {self.score.fruits}  Steg: {self.score.steps}.{suffix}"
            )
        )

    def advance_snake(self) -> str:
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            return "wall"
        if new_head in self.snake:
            return "self"

        self.snake.insert(0, new_head)
        if new_head == self.fruit:
            self.spawn_fruit()
            return "fruit"
        self.snake.pop()
        return "step"


class TrainingViewer:
    """Lightweight Tkinter-visualisering av träningsmiljön."""

    def __init__(
        self,
        *,
        grid_size: int,
        total_episodes: int,
        steps_per_episode: int,
        cell_size: int = CELL_SIZE,
        delay_ms: int = STEP_DELAY,
    ) -> None:
        self.grid_size = grid_size
        self.total_episodes = total_episodes
        self.steps_per_episode = steps_per_episode
        self.cell_size = cell_size
        self.delay = max(delay_ms / 1000.0, 0.0)
        self._last_draw = 0.0
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Snake-ML – Träningsvisualisering")
        self.root.configure(bg="#101010")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        pixel_size = self.grid_size * self.cell_size
        margin = 16

        self.status = tk.Label(
            self.root,
            text="Initierar träning…",
            font=("Segoe UI", 12),
            anchor="w",
            padx=12,
            pady=8,
            width=80,
            bg="#101010",
            fg="#f5f5f5",
        )
        self.status.pack(fill="x")

        self.canvas = tk.Canvas(
            self.root,
            width=pixel_size,
            height=pixel_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.canvas.pack(padx=margin, pady=margin)

        view_size = pixel_size + 2 * margin
        self.root.update_idletasks()
        status_height = self.status.winfo_height()
        total_height = view_size + status_height
        geometry = f"{view_size}x{total_height}"
        self.root.geometry(geometry)
        self.root.minsize(view_size, total_height)
        self.root.maxsize(view_size, total_height)

        self._draw_background()
        self.root.update_idletasks()
        self.root.update()

    def update(
        self,
        *,
        snapshot: Dict[str, object],
        episode: int,
        step: int,
        epsilon: float,
        total_reward: float,
        last_reward: float,
        cause: str,
        loss: float,
    ) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        if now - self._last_draw < self.delay:
            self.root.update_idletasks()
            self.root.update()
            return
        self._last_draw = now

        snake = list(snapshot.get("snake", []))
        fruit = snapshot.get("fruit")

        self.canvas.delete("all")
        self._draw_background()
        self._draw_fruit(fruit)
        self._draw_snake(snake)

        status_text = (
            f"Episod {episode}/{self.total_episodes} | Steg {step} | "
            f"Reward {total_reward:6.1f} | Δ {last_reward:5.2f} | ε={epsilon:.3f} | "
            f"Senaste: {self._format_cause(cause)} | Förlust {loss:7.4f}"
        )
        self.status.config(text=status_text)

        self.root.update_idletasks()
        self.root.update()

    def episode_done(self) -> None:
        if self.closed:
            return
        current = self.status.cget("text")
        if "| Slut" not in current:
            self.status.config(text=f"{current} | Slut på episod")
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _draw_background(self) -> None:
        pixel_size = self.grid_size * self.cell_size
        for row in range(self.grid_size):
            color = "#1f1f1f" if row % 2 == 0 else "#232323"
            self.canvas.create_rectangle(
                0,
                row * self.cell_size,
                pixel_size,
                (row + 1) * self.cell_size,
                fill=color,
                width=0,
            )

    def _draw_snake(self, snake: List[Point]) -> None:
        for index, (x, y) in enumerate(snake):
            color = "#8bc34a" if index == 0 else "#4caf50"
            self.canvas.create_rectangle(
                x * self.cell_size,
                y * self.cell_size,
                (x + 1) * self.cell_size,
                (y + 1) * self.cell_size,
                fill=color,
                outline="#1b5e20",
                width=1,
            )

    def _draw_fruit(self, fruit: Optional[Point]) -> None:
        if fruit is None:
            return
        x, y = fruit
        self.canvas.create_oval(
            x * self.cell_size + 4,
            y * self.cell_size + 4,
            (x + 1) * self.cell_size - 4,
            (y + 1) * self.cell_size - 4,
            fill="#ff9800",
            outline="#ef6c00",
            width=1,
        )

    def _format_cause(self, cause: str) -> str:
        translations = {
            "fruit": "frukt",
            "wall": "vägg",
            "self": "kollision",
            "step": "steg",
        }
        return translations.get(cause, cause)


# ---------------------------------------------------------------------------
# Interaktiv träningskontroll för IDLE
# ---------------------------------------------------------------------------


@dataclass
class TrainingStats:
    episodes: int = 0
    total_reward: float = 0.0
    total_length: float = 0.0
    best_length: int = 0
    last_episode_reward: float = 0.0
    last_episode_length: float = 0.0

    def reset(self) -> None:
        self.episodes = 0
        self.total_reward = 0.0
        self.total_length = 0.0
        self.best_length = 0
        self.last_episode_reward = 0.0
        self.last_episode_length = 0.0

    def record(self, reward: float, length: float) -> None:
        self.episodes += 1
        self.total_reward += reward
        self.total_length += length
        self.best_length = max(self.best_length, int(length))
        self.last_episode_reward = reward
        self.last_episode_length = length

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.episodes if self.episodes else 0.0

    @property
    def average_length(self) -> float:
        return self.total_length / self.episodes if self.episodes else 0.0


class IdleTrainerApp:
    """Tkinter-gränssnitt för att styra träning och visa statistik live."""

    def __init__(self, *, load_path: Optional[Path | str] = None) -> None:
        self.env = IdleSnakeEnv()
        initial_state = self.env.reset()
        self.stats = TrainingStats()
        self.training_active = False
        self.evaluation_active = False
        self.last_reward: float = 0.0
        self.last_loss: Optional[float] = None
        self.last_evaluation_reward: float = 0.0
        self.evaluation_episodes: int = 0
        self.current_state: Optional[np.ndarray] = None
        self.current_episode_reward: float = 0.0
        self.current_episode_steps: int = 0
        self._syncing_params = False
        self._syncing_rewards = False
        self._pending_message: Optional[str] = None
        self._last_mode: str = "idle"

        self.training_envs: List[IdleSnakeEnv] = [self.env]
        self.env_states: List[np.ndarray] = [initial_state.copy()]
        self.env_episode_rewards: List[float] = [0.0]
        self.env_episode_steps: List[int] = [0]
        self.max_parallel_envs = 16
        self._syncing_env_count = False

        self.agent = self._load_initial_agent(load_path)

        self.root = tk.Tk()
        self.root.title("Snake-ML – Träningskontroll")
        self.root.configure(bg="#101010")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._sync_param_entries_from_agent()
        self._sync_reward_sliders_from_env()
        self._reset_episode()
        if self._pending_message:
            self._log_message(self._pending_message)
        else:
            self._log_message("Klar att starta träning.")
        self.root.update_idletasks()
        pixel_size = GRID_SIZE * CELL_SIZE
        min_width = pixel_size + 420
        min_height = pixel_size + 160
        requested_width = self.root.winfo_width()
        requested_height = self.root.winfo_height()
        target_width = max(requested_width, pixel_size + 640)
        target_height = max(requested_height, pixel_size + 220)
        self.root.geometry(f"{target_width}x{target_height}")
        self.root.minsize(min_width, min_height)

        _, initial_delay = self._compute_update_schedule()
        self.root.after(initial_delay, self._update_loop)

    def _load_initial_agent(self, load_path: Optional[Path | str]) -> DoubleDQNAgent:
        if load_path is not None:
            try:
                agent = DoubleDQNAgent.load(load_path)
            except Exception as exc:  # pragma: no cover - användarfeedback
                print(f"Kunde inte ladda modellen '{load_path}': {exc}", file=sys.stderr)
            else:
                if agent.state_size != self.env.state_size:
                    print(
                        "Den laddade modellen matchar inte rutnätets storlek och kan inte användas.",
                        file=sys.stderr,
                    )
                else:
                    name = Path(load_path).name if isinstance(load_path, (str, Path)) else "modell"
                    self._pending_message = f"Laddade modell från {name}."
                    return agent
            self._pending_message = "Misslyckades att ladda angiven modell. Startar med ny agent."
        return DoubleDQNAgent(state_size=self.env.state_size, action_size=len(ACTION_VECTORS))

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg="#101010")
        main.pack(fill="both", expand=True, padx=16, pady=16)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        pixel_size = GRID_SIZE * CELL_SIZE
        self.canvas = tk.Canvas(
            main,
            width=pixel_size,
            height=pixel_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nw")

        sidebar = tk.Frame(main, bg="#101010")
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        sidebar.grid_columnconfigure(0, weight=1)

        self.stats_var = tk.StringVar()
        stats_label = tk.Label(
            sidebar,
            textvariable=self.stats_var,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Segoe UI", 11),
            bg="#101010",
            fg="#f5f5f5",
        )
        stats_label.pack(fill="x")

        self.detail_var = tk.StringVar()
        detail_label = tk.Label(
            sidebar,
            textvariable=self.detail_var,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Segoe UI", 10),
            bg="#101010",
            fg="#cddc39",
        )
        detail_label.pack(fill="x", pady=(4, 8))

        self._create_buttons(sidebar)

        tuning_frame = tk.Frame(sidebar, bg="#101010")
        tuning_frame.pack(fill="both", expand=True, pady=(12, 8))
        tuning_frame.grid_columnconfigure(0, weight=1)
        tuning_frame.grid_columnconfigure(1, weight=1)
        tuning_frame.grid_rowconfigure(0, weight=1)

        params_container = tk.Frame(tuning_frame, bg="#101010")
        params_container.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        header = tk.Label(
            params_container,
            text="Hyperparametrar",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            bg="#101010",
            fg="#f5f5f5",
        )
        header.pack(fill="x", pady=(0, 6))

        self.param_controls: Dict[str, Dict[str, object]] = {}

        env_row = tk.Frame(params_container, bg="#101010")
        env_row.pack(fill="x", pady=2)
        tk.Label(
            env_row,
            text="Antal miljöer",
            width=18,
            anchor="w",
            bg="#101010",
            fg="#d0d0d0",
            font=("Segoe UI", 10),
        ).pack(side="left")
        self.parallel_envs_var = tk.IntVar(value=len(self.training_envs))
        env_spinbox = tk.Spinbox(
            env_row,
            from_=1,
            to=self.max_parallel_envs,
            textvariable=self.parallel_envs_var,
            width=6,
            justify="right",
            bg="#101010",
            fg="#90caf9",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#1c1c1c",
            highlightcolor="#90caf9",
            insertbackground="#90caf9",
        )
        env_spinbox.pack(side="right")
        self.parallel_envs_var.trace_add("write", self._on_parallel_envs_var_changed)

        param_specs = [
            ("learning_rate", "Lärhastighet", 0.0001, 0.01, 0.0001, float, "{:.4f}"),
            ("gamma", "Gamma", 0.8, 0.999, 0.001, float, "{:.3f}"),
            ("epsilon", "Epsilon", 0.0, 1.0, 0.01, float, "{:.2f}"),
            ("epsilon_min", "Min epsilon", 0.0, 0.5, 0.01, float, "{:.2f}"),
            ("epsilon_decay", "Epsilon-förfall", 0.8, 0.9999, 0.0001, float, "{:.4f}"),
            ("batch_size", "Batch-storlek", 8, 512, 8, int, "{:d}"),
            ("target_sync_interval", "Synkintervall", 100, 5000, 50, int, "{:d}"),
        ]
        for name, label_text, minimum, maximum, resolution, value_type, fmt in param_specs:
            row = tk.Frame(params_container, bg="#101010")
            row.pack(fill="x", pady=2)
            tk.Label(
                row,
                text=label_text,
                width=18,
                anchor="w",
                bg="#101010",
                fg="#d0d0d0",
                font=("Segoe UI", 10),
            ).pack(side="left")
            value_label = tk.Label(
                row,
                text="",
                width=8,
                anchor="e",
                bg="#101010",
                fg="#cddc39",
                font=("Consolas", 10),
            )
            value_label.pack(side="right")
            var_cls = tk.DoubleVar if value_type is float else tk.IntVar
            initial_value = getattr(self.agent, name)
            var = var_cls(value=initial_value)
            scale = tk.Scale(
                row,
                from_=minimum,
                to=maximum,
                resolution=resolution,
                orient="horizontal",
                variable=var,
                showvalue=False,
                length=220,
                bg="#101010",
                troughcolor="#1c1c1c",
                highlightthickness=0,
                sliderrelief="flat",
                command=partial(self._on_param_slider_change, name, value_type, fmt, value_label),
            )
            scale.pack(side="left", fill="x", expand=True, padx=(8, 0))
            display_value = initial_value if value_type is float else int(initial_value)
            value_label.config(text=fmt.format(display_value))
            self.param_controls[name] = {
                "var": var,
                "label": value_label,
                "format": fmt,
                "type": value_type,
                "scale": scale,
            }

        speed_frame = tk.Frame(params_container, bg="#101010")
        speed_frame.pack(fill="x", pady=(10, 0))
        tk.Label(
            speed_frame,
            text="Uppdateringar per sekund",
            anchor="w",
            bg="#101010",
            fg="#d0d0d0",
            font=("Segoe UI", 10),
        ).pack(side="left")
        self.updates_per_second_var = tk.DoubleVar(value=12.0)
        speed_value_label = tk.Label(
            speed_frame,
            text="",
            width=8,
            anchor="e",
            bg="#101010",
            fg="#90caf9",
            font=("Consolas", 10),
        )
        speed_value_label.pack(side="right")
        speed_slider = tk.Scale(
            speed_frame,
            from_=1,
            to=240,
            resolution=1,
            orient="horizontal",
            variable=self.updates_per_second_var,
            showvalue=False,
            length=220,
            bg="#101010",
            troughcolor="#1c1c1c",
            highlightthickness=0,
            sliderrelief="flat",
            command=partial(self._on_speed_change, speed_value_label),
        )
        speed_slider.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self._on_speed_change(speed_value_label, str(self.updates_per_second_var.get()))

        reward_container = tk.Frame(tuning_frame, bg="#101010")
        reward_container.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        reward_header = tk.Label(
            reward_container,
            text="Reward-vikter",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            bg="#101010",
            fg="#f5f5f5",
        )
        reward_header.pack(fill="x", pady=(0, 6))

        self.reward_controls: Dict[str, Dict[str, object]] = {}
        for reward_field in fields(RewardConfig):
            name = reward_field.name
            row = tk.Frame(reward_container, bg="#101010")
            row.pack(fill="x", pady=2)
            label_text = name.replace("_", " ").title()
            tk.Label(
                row,
                text=label_text,
                width=18,
                anchor="w",
                bg="#101010",
                fg="#d0d0d0",
                font=("Segoe UI", 10),
            ).pack(side="left")
            value_label = tk.Label(
                row,
                text="",
                width=8,
                anchor="e",
                bg="#101010",
                fg="#ffb74d",
                font=("Consolas", 10),
            )
            value_label.pack(side="right")
            default_value = getattr(self.env.reward_config, name)
            limit = max(5.0, abs(default_value) * 3.0)
            var = tk.DoubleVar(value=default_value)
            scale = tk.Scale(
                row,
                from_=-limit,
                to=limit,
                resolution=0.01,
                orient="horizontal",
                variable=var,
                showvalue=False,
                length=220,
                bg="#101010",
                troughcolor="#1c1c1c",
                highlightthickness=0,
                sliderrelief="flat",
                command=partial(self._on_reward_change, name, value_label),
            )
            scale.pack(side="left", fill="x", expand=True, padx=(8, 0))
            value_label.config(text=f"{default_value:+.2f}")
            self.reward_controls[name] = {
                "var": var,
                "label": value_label,
                "scale": scale,
                "format": "{:+.2f}",
            }

        self.message_var = tk.StringVar()
        message_label = tk.Label(
            sidebar,
            textvariable=self.message_var,
            justify="left",
            wraplength=360,
            anchor="w",
            font=("Segoe UI", 10),
            bg="#101010",
            fg="#90caf9",
        )
        message_label.pack(fill="x", pady=(8, 0))

        def _update_wraplength(event: tk.Event) -> None:
            wrap = max(event.width - 24, 240)
            stats_label.config(wraplength=wrap)
            detail_label.config(wraplength=wrap)
            message_label.config(wraplength=wrap)

        sidebar.bind("<Configure>", _update_wraplength)

    def _create_buttons(self, parent: tk.Misc) -> None:
        button_frame = tk.Frame(parent, bg="#101010")
        button_frame.pack(fill="x", pady=(0, 8))

        btn_specs = [
            ("Starta träning", self.start_training, "#2e7d32"),
            ("Pausa", self.pause_training, "#455a64"),
            ("Starta utvärdering", self.start_evaluation, "#1565c0"),
            ("Stoppa utvärdering", self.stop_evaluation, "#546e7a"),
            ("Spara modell", self.save_model, "#455a64"),
            ("Ladda modell", self.load_model, "#455a64"),
            ("Återställ statistik", self.reset_stats, "#455a64"),
        ]
        for text, command, color in btn_specs:
            button = tk.Button(
                button_frame,
                text=text,
                command=command,
                width=20,
                bg=color,
                fg="#f5f5f5",
                activebackground=color,
                activeforeground="#f5f5f5",
                relief="flat",
                pady=6,
                font=("Segoe UI", 10, "bold" if "Starta" in text else "normal"),
            )
            button.pack(fill="x", pady=3)

    def _reset_env(self, index: int, *, update_canvas: bool = True) -> None:
        env = self.training_envs[index]
        state = env.reset()
        if index >= len(self.env_states):
            self.env_states.append(state.copy())
            self.env_episode_rewards.append(0.0)
            self.env_episode_steps.append(0)
        else:
            self.env_states[index] = state.copy()
            self.env_episode_rewards[index] = 0.0
            self.env_episode_steps[index] = 0
        if index == 0:
            self.current_state = state.copy()
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
            self.last_reward = 0.0
            if update_canvas and hasattr(self, "canvas"):
                self._update_canvas(env.snapshot())

    def _set_parallel_envs(self, count: int) -> None:
        count = max(1, min(self.max_parallel_envs, count))
        current = len(self.training_envs)
        if count == current:
            return
        if count < current:
            self.training_envs = self.training_envs[:count]
            self.env_states = self.env_states[:count]
            self.env_episode_rewards = self.env_episode_rewards[:count]
            self.env_episode_steps = self.env_episode_steps[:count]
        else:
            for _ in range(count - current):
                new_env = IdleSnakeEnv(
                    grid_size=self.env.grid_size,
                    reward_config=replace(self.env.reward_config),
                )
                state = new_env.reset()
                self.training_envs.append(new_env)
                self.env_states.append(state.copy())
                self.env_episode_rewards.append(0.0)
                self.env_episode_steps.append(0)
        self.env = self.training_envs[0]
        if self.current_state is not None:
            self.env_states[0] = self.current_state.copy()
            self.env_episode_rewards[0] = self.current_episode_reward
            self.env_episode_steps[0] = self.current_episode_steps
        else:
            self._reset_env(0, update_canvas=False)
        self._update_additional_env_reward_configs()

    def _reset_episode(self) -> None:
        self._reset_env(0)
        self._update_stats_label()

    def _update_additional_env_reward_configs(self) -> None:
        if len(self.training_envs) <= 1:
            return
        source = self.env.reward_config
        for other_env in self.training_envs[1:]:
            for reward_field in fields(RewardConfig):
                setattr(other_env.reward_config, reward_field.name, getattr(source, reward_field.name))

    def _compute_update_schedule(self) -> Tuple[int, int]:
        try:
            updates = float(self.updates_per_second_var.get())
        except (TypeError, tk.TclError):
            updates = 12.0
        updates = max(1.0, updates)
        max_refresh = 60.0
        if updates <= max_refresh:
            delay = max(10, int(1000 / updates))
            steps = 1
        else:
            steps = int(math.ceil(updates / max_refresh))
            delay = max(10, int(1000 / max_refresh))
        return steps, delay

    def _update_loop(self) -> None:
        steps, delay = self._compute_update_schedule()
        if self.training_active and self.current_state is not None:
            for _ in range(steps):
                if not self.training_active:
                    break
                self._run_training_step()
        elif self.evaluation_active and self.current_state is not None:
            for _ in range(steps):
                if not self.evaluation_active:
                    break
                self._run_evaluation_step()
        self._update_canvas(self.env.snapshot())
        self._update_stats_label()
        self.root.after(delay, self._update_loop)

    def _run_training_step(self) -> None:
        for index, env in enumerate(self.training_envs):
            state = self.env_states[index]
            action = self.agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            transition = Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
            )
            self.agent.push(transition)
            loss = self.agent.learn()
            if loss is not None:
                self.last_loss = loss
            self.agent.decay_epsilon()

            self.env_states[index] = next_state.copy()
            self.env_episode_rewards[index] += reward
            steps = int(info.get("steps", self.env_episode_steps[index] + 1))
            self.env_episode_steps[index] = steps

            if index == 0:
                self.current_state = next_state.copy()
                self.current_episode_reward = self.env_episode_rewards[0]
                self.current_episode_steps = steps
                self.last_reward = reward
                self._last_mode = "training"

            if done:
                total_reward = self.env_episode_rewards[index]
                self.stats.record(total_reward, env.max_length)
                self._reset_env(index, update_canvas=index == 0)

    def _run_evaluation_step(self) -> None:
        assert self.current_state is not None
        action = self.agent.select_action(self.current_state, greedy=True)
        next_state, reward, done, info = self.env.step(action)

        self.current_state = next_state
        self.current_episode_reward += reward
        self.current_episode_steps = int(info.get("steps", self.current_episode_steps + 1))
        self.last_reward = reward
        self._last_mode = "evaluation"

        if done:
            self.last_evaluation_reward = self.current_episode_reward
            self.evaluation_episodes += 1
            self._reset_episode()

    def _update_canvas(self, snapshot: Dict[str, object]) -> None:
        self.canvas.delete("all")
        self._draw_background()
        self._draw_fruit(snapshot.get("fruit"))
        snake = snapshot.get("snake") or []
        self._draw_snake(list(snake))

    def _draw_background(self) -> None:
        pixel_size = GRID_SIZE * CELL_SIZE
        for row in range(GRID_SIZE):
            color = "#1f1f1f" if row % 2 == 0 else "#232323"
            self.canvas.create_rectangle(
                0,
                row * CELL_SIZE,
                pixel_size,
                (row + 1) * CELL_SIZE,
                fill=color,
                width=0,
            )

    def _draw_snake(self, snake: List[Point]) -> None:
        for index, (x, y) in enumerate(snake):
            color = "#8bc34a" if index == 0 else "#4caf50"
            self.canvas.create_rectangle(
                x * CELL_SIZE,
                y * CELL_SIZE,
                (x + 1) * CELL_SIZE,
                (y + 1) * CELL_SIZE,
                fill=color,
                outline="#1b5e20",
                width=1,
            )

    def _draw_fruit(self, fruit: Optional[Point]) -> None:
        if fruit is None:
            return
        x, y = fruit
        self.canvas.create_oval(
            x * CELL_SIZE + 4,
            y * CELL_SIZE + 4,
            (x + 1) * CELL_SIZE - 4,
            (y + 1) * CELL_SIZE - 4,
            fill="#ff9800",
            outline="#ef6c00",
            width=1,
        )

    def _update_stats_label(self) -> None:
        current_length = len(self.env.snake)
        fruits = self.env.fruits_eaten
        if self.evaluation_active or self._last_mode == "evaluation":
            status = "Utvärdering aktiv" if self.evaluation_active else "Utvärdering pausad"
            self.stats_var.set(
                (
                    f"{status} | Episoder: {self.evaluation_episodes} | "
                    f"Senaste episodreward: {self.last_evaluation_reward:.2f} | "
                    f"Pågående reward: {self.current_episode_reward:.2f} | Frukter: {fruits} | "
                    f"Längd: {current_length}"
                )
            )
            self.detail_var.set(
                (
                    f"Steg {self.current_episode_steps} | Senaste reward: {self.last_reward:+.2f} | "
                    f"Max längd: {self.env.max_length} | Total reward: {self.env.total_reward:.2f} | "
                    f"Miljöer: {len(self.training_envs)}"
                )
            )
            return

        avg_reward = self.stats.average_reward
        avg_length = self.stats.average_length
        best_length = self.stats.best_length
        self.stats_var.set(
            (
                f"Träningsstatistik – Episoder: {self.stats.episodes} | Medelpoäng: {avg_reward:.2f} | "
                f"Medellängd: {avg_length:.2f} | Längsta längd: {best_length} | "
                f"Senaste episodpoäng: {self.stats.last_episode_reward:.2f} | Senaste längd: {self.stats.last_episode_length:.2f}"
            )
        )
        loss_text = f"{self.last_loss:.4f}" if self.last_loss is not None else "–"
        mode = "Träning" if self.training_active else "Pausad"
        self.detail_var.set(
            (
                f"Läge: {mode} | Miljöer: {len(self.training_envs)} | Episod {self.stats.episodes + 1} | "
                f"Steg {self.current_episode_steps} | "
                f"Frukter: {fruits} | Nuvarande längd: {current_length} | Max längd: {self.env.max_length} | "
                f"Senaste reward: {self.last_reward:+.2f} | Episodreward: {self.current_episode_reward:.2f} | "
                f"Förlust: {loss_text} | ε={self.agent.epsilon:.3f}"
            )
        )

    def _log_message(self, message: str) -> None:
        self.message_var.set(message)

    def start_training(self) -> None:
        reset_needed = False
        if self.evaluation_active:
            self.evaluation_active = False
            reset_needed = True
        elif self._last_mode == "evaluation":
            reset_needed = True
        if reset_needed:
            self._reset_episode()
            for idx in range(1, len(self.training_envs)):
                self._reset_env(idx, update_canvas=False)
        if not self.training_active:
            self.training_active = True
            self._last_mode = "training"
            self._log_message("Träning pågår…")
        self._update_stats_label()

    def pause_training(self) -> None:
        if self.training_active or self.evaluation_active:
            was_training = self.training_active
            was_evaluation = self.evaluation_active
            self.training_active = False
            self.evaluation_active = False
            if was_training:
                self._last_mode = "training"
                self._log_message("Träningen pausades.")
            elif was_evaluation:
                self._last_mode = "evaluation"
                self._log_message("Utvärderingen pausades.")
            self._update_stats_label()

    def start_evaluation(self) -> None:
        reset_needed = False
        if self.training_active:
            self.training_active = False
            reset_needed = True
        if self._last_mode != "evaluation":
            reset_needed = True
        if reset_needed:
            self._reset_episode()
        if not self.evaluation_active:
            self.evaluation_active = True
            self._last_mode = "evaluation"
            self.last_loss = None
            self._log_message("Utvärdering pågår…")
        self._update_stats_label()

    def stop_evaluation(self) -> None:
        if self.evaluation_active:
            self.evaluation_active = False
            self._last_mode = "evaluation"
            self._log_message("Utvärderingen stoppades.")
        self._update_stats_label()

    def reset_stats(self) -> None:
        self.stats.reset()
        self.evaluation_episodes = 0
        self.last_evaluation_reward = 0.0
        self._log_message("Statistiken återställdes.")
        self._update_stats_label()

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Välj modell att ladda",
            filetypes=[("Double DQN-modeller", "*.npz"), ("Alla filer", "*.*")],
        )
        if not path:
            return
        try:
            agent = DoubleDQNAgent.load(path)
        except Exception as exc:  # pragma: no cover - användarfeedback
            messagebox.showerror("Fel", f"Kunde inte ladda modellen:\n{exc}")
            return
        if agent.state_size != self.env.state_size:
            messagebox.showerror(
                "Fel",
                "Modellen använder ett annat rutnätsformat och kan inte laddas i denna miljö.",
            )
            return
        self.pause_training()
        self.agent = agent
        self._sync_param_entries_from_agent()
        self._reset_episode()
        for idx in range(1, len(self.training_envs)):
            self._reset_env(idx, update_canvas=False)
        name = Path(path).name
        self._log_message(f"Laddade modell från {name}.")

    def save_model(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Spara Double DQN-modell",
            defaultextension=".npz",
            initialfile="idle_double_dqn.npz",
            filetypes=[("Double DQN-modeller", "*.npz"), ("Alla filer", "*.*")],
        )
        if not path:
            return
        try:
            saved_path = self.agent.save(path)
        except Exception as exc:  # pragma: no cover - användarfeedback
            messagebox.showerror("Fel", f"Kunde inte spara modellen:\n{exc}")
            return
        name = Path(saved_path).name
        self._log_message(f"Modellen sparades till {name}.")

    def _sync_param_entries_from_agent(self) -> None:
        if not hasattr(self, "param_controls"):
            return
        self._syncing_params = True
        for name, control in self.param_controls.items():
            value = getattr(self.agent, name)
            control["var"].set(value)
            display_value = int(value) if control["type"] is int else float(value)
            control["label"].config(text=control["format"].format(display_value))
        self._syncing_params = False

    def _sync_reward_sliders_from_env(self) -> None:
        if not hasattr(self, "reward_controls"):
            return
        self._syncing_rewards = True
        for name, control in self.reward_controls.items():
            value = getattr(self.env.reward_config, name)
            control["var"].set(value)
            control["label"].config(text=control["format"].format(value))
        self._syncing_rewards = False
        self._update_additional_env_reward_configs()

    def _on_parallel_envs_var_changed(self, *_: object) -> None:
        if self._syncing_env_count:
            return
        try:
            requested = int(self.parallel_envs_var.get())
        except (tk.TclError, ValueError):
            return
        self._apply_parallel_env_count(requested)

    def _apply_parallel_env_count(self, requested: int) -> None:
        requested = max(1, min(self.max_parallel_envs, requested))
        current = len(self.training_envs)
        if requested == current:
            return
        if self.training_active or self.evaluation_active:
            self._syncing_env_count = True
            try:
                self.parallel_envs_var.set(current)
            finally:
                self._syncing_env_count = False
            if self.training_active:
                self._log_message("Pausa träningen innan du ändrar antal miljöer.")
            else:
                self._log_message("Stoppa utvärderingen innan du ändrar antal miljöer.")
            return
        self._set_parallel_envs(requested)
        self._syncing_env_count = True
        try:
            self.parallel_envs_var.set(len(self.training_envs))
        finally:
            self._syncing_env_count = False
        self._log_message(f"Antal träningsmiljöer satt till {len(self.training_envs)}.")
        self._update_stats_label()

    def _on_param_slider_change(
        self,
        name: str,
        value_type: type,
        fmt: str,
        value_label: tk.Label,
        value: str,
    ) -> None:
        if self._syncing_params:
            return
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        control = self.param_controls[name]
        if value_type is int:
            numeric_value = int(round(numeric_value))
            self._syncing_params = True
            try:
                control["var"].set(numeric_value)
            finally:
                self._syncing_params = False
        setattr(self.agent, name, value_type(numeric_value) if value_type is float else numeric_value)
        display_value = numeric_value if value_type is float else int(numeric_value)
        value_label.config(text=fmt.format(display_value))

    def _on_speed_change(self, label: tk.Label, value: str) -> None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float(self.updates_per_second_var.get() or 0)
        numeric_value = max(1.0, numeric_value)
        label.config(text=f"{numeric_value:.0f}/s" if numeric_value >= 10 else f"{numeric_value:.1f}/s")

    def _on_reward_change(self, name: str, value_label: tk.Label, value: str) -> None:
        if self._syncing_rewards:
            return
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        control = self.reward_controls[name]
        self._syncing_rewards = True
        try:
            control["var"].set(numeric_value)
        finally:
            self._syncing_rewards = False
        setattr(self.env.reward_config, name, float(numeric_value))
        self._update_additional_env_reward_configs()
        value_label.config(text=control["format"].format(numeric_value))

    def _on_close(self) -> None:
        self.training_active = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        self.root.mainloop()


def start_idle_trainer_ui(load_path: Optional[Path | str] = None) -> None:
    """Starta det interaktiva Tk-gränssnittet för träning."""

    app = IdleTrainerApp(load_path=load_path)
    app.run()


# ---------------------------------------------------------------------------
# Training / evaluation CLI
# ---------------------------------------------------------------------------


def train_double_dqn(
    episodes: int,
    steps_per_episode: int,
    *,
    seed: Optional[int] = None,
    load_path: Optional[Path | str] = None,
    save_path: Optional[Path | str] = None,
    visualize: bool = False,
) -> DoubleDQNAgent:
    env = IdleSnakeEnv(seed=seed)
    if load_path is not None:
        agent = DoubleDQNAgent.load(load_path)
        print(f"Fortsätter träning från {load_path}.")
    else:
        agent = DoubleDQNAgent(state_size=env.state_size, action_size=len(ACTION_VECTORS), seed=seed)
        print("Startar ny Double DQN-träning.")

    rewards: List[float] = []
    start_time = time.time()
    viewer: Optional[TrainingViewer] = None
    if visualize:
        try:
            viewer = TrainingViewer(
                grid_size=env.grid_size,
                total_episodes=episodes,
                steps_per_episode=steps_per_episode,
            )
        except tk.TclError as exc:
            print(
                "Det gick inte att starta träningsvisualiseringen (Tkinter). Fortsätter utan grafiskt läge.",
                file=sys.stderr,
            )
            viewer = None

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses: List[float] = []
        for step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(Transition(state, action, reward, next_state, done))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            agent.decay_epsilon()
            state = next_state
            total_reward += reward
            if viewer is not None:
                try:
                    viewer.update(
                        snapshot=env.snapshot(),
                        episode=episode,
                        step=step + 1,
                        epsilon=agent.epsilon,
                        total_reward=total_reward,
                        last_reward=reward,
                        cause=info.get("cause", "step"),
                        loss=losses[-1] if losses else 0.0,
                    )
                except tk.TclError:
                    print("Träningsfönstret stängdes – fortsätter träningen utan visualisering.")
                    viewer = None
            if done:
                break
        rewards.append(total_reward)
        if episode % 10 == 0:
            mean_reward = sum(rewards[-10:]) / min(len(rewards), 10)
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            duration = time.time() - start_time
            print(
                f"Episod {episode:4d} | Snittbelöning (10): {mean_reward:6.2f} | "
                f"Senaste episod: {total_reward:6.2f} | Förlust: {mean_loss:8.4f} | "
                f"ε={agent.epsilon:.3f} | Tid: {duration:5.1f}s"
            )
        if viewer is not None:
            try:
                viewer.episode_done()
            except tk.TclError:
                print("Träningsfönstret stängdes – fortsätter träningen utan visualisering.")
                viewer = None

    if save_path is not None:
        path = agent.save(save_path)
        print(f"Sparade modellen till {path}.")
    if viewer is not None:
        viewer.close()
    return agent


def evaluate_agent(agent: DoubleDQNAgent, episodes: int, steps_per_episode: int, *, seed: Optional[int] = None) -> List[float]:
    env = IdleSnakeEnv(seed=seed)
    results: List[float] = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(steps_per_episode):
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        results.append(total_reward)
    return results


def start_game(agent: Optional[DoubleDQNAgent] = None, autopilot: bool = False) -> None:
    root = tk.Tk()
    root.title("Snake-ML – IDLE Edition")
    root.resizable(False, False)
    root.configure(bg="#101010")

    status = tk.Label(
        root,
        text="Tryck på piltangenterna för att styra. Space startar om, Escape avslutar.",
        font=("Segoe UI", 12),
        bg="#101010",
        fg="#f5f5f5",
        anchor="w",
        padx=12,
        pady=8,
    )
    status.pack(fill="x")

    canvas = SnakeCanvas(root, status, agent=agent, autopilot=autopilot)
    canvas.pack()
    canvas.start()

    root.mainloop()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake-ML IDLE Double DQN toolkit")
    parser.add_argument("--train", type=int, metavar="EP", help="Antal träningsepisoder att köra", default=0)
    parser.add_argument("--steps", type=int, metavar="N", help="Maxsteg per episod under träning", default=500)
    parser.add_argument("--seed", type=int, help="Slumpfrö", default=None)
    parser.add_argument("--load-model", type=str, help="Ladda en sparad agent för träning eller spel", default=None)
    parser.add_argument("--save-model", type=str, help="Var ska modellen sparas efter träning", default="idle_double_dqn.npz")
    parser.add_argument("--evaluate", type=int, metavar="EP", help="Kör utvärdering med angivet antal episoder", default=0)
    parser.add_argument("--play", action="store_true", help="Starta Tkinter-spelet")
    parser.add_argument("--autopilot", action="store_true", help="Aktivera autopilot när spelet startas")
    parser.add_argument(
        "--visualize-training",
        action="store_true",
        help="Visa träningsmiljön live i ett Tkinter-fönster",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    agent: Optional[DoubleDQNAgent] = None
    if args.train > 0:
        agent = train_double_dqn(
            args.train,
            args.steps,
            seed=args.seed,
            load_path=args.load_model,
            save_path=args.save_model,
            visualize=args.visualize_training,
        )
    if args.evaluate or args.autopilot:
        if agent is None:
            if args.load_model:
                agent = DoubleDQNAgent.load(args.load_model)
                print(f"Laddade agent från {args.load_model}.")
            else:
                raise SystemExit("Ingen agent att använda. Träna först eller ladda en modell.")

    if args.evaluate:
        scores = evaluate_agent(agent, args.evaluate, args.steps, seed=args.seed)
        avg = sum(scores) / len(scores)
        best = max(scores)
        worst = min(scores)
        print(
            f"Utvärdering över {args.evaluate} episoder | Snitt: {avg:.2f} | "
            f"Bäst: {best:.2f} | Sämst: {worst:.2f}"
        )

    if args.play or (args.autopilot and agent is not None):
        try:
            start_game(agent=agent, autopilot=args.autopilot)
        except tk.TclError as exc:  # pragma: no cover - headless safeguard
            print(
                "Det gick inte att starta Tkinter. Om du kör skriptet på en server utan skärm",
                "behöver du köra det lokalt i stället.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    elif args.train == 0 and args.evaluate == 0:
        # Default behaviour when running without flaggar: visa den nya träningskontrollen.
        try:
            start_idle_trainer_ui(load_path=args.load_model if agent is None else None)
        except tk.TclError as exc:  # pragma: no cover - headless safeguard
            print(
                "Det gick inte att starta Tkinter. Om du kör skriptet på en server utan skärm",
                "behöver du köra det lokalt i stället.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
