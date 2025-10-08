"""Gymnasium environment replicating Marcus Petersson's Snake-ML HTML logic.

This module mirrors the browser version by providing:
* Grid based snake game with identical reward shaping (fruit, step, death, loop,
  compact and wall penalties/bonuses).
* Multiple deterministic start patterns (line, cube, edge spiral, random) just
  like the original scripted setups Marcus used in TensorFlow.js.
* Loop detection that punishes repeating turning patterns such as 1,2,1,2 …
* Optional pygame rendering that shows the live training progress while
  respecting silent vectorised environments when training several agents.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.utils import seeding


Action = int
Coordinate = Tuple[int, int]


@dataclass
class RewardConfig:
    """Structured reward configuration to mirror the HTML constants."""

    fruit_reward: float = 10.0
    step_penalty: float = -0.01
    death_penalty: float = -10.0
    loop_penalty: float = -1.0
    compact_bonus: float = 0.05
    wall_penalty: float = -5.0


class SnakeEnv(Env):
    """Marcus Petersson's Snake-ML environment rewritten for Gymnasium.

    The environment keeps the same three channel observation encoding used in the
    browser version:
        * Channel 0 – Snake body/head occupancy (1 where the snake lives).
        * Channel 1 – Fruit position.
        * Channel 2 – Normalised Manhattan distance field towards the fruit to
          retain the compact guidance used by the original heuristics.

    Rendering with pygame is optional and only enabled for the first vectorised
    environment to avoid conflicts when running multiple training workers.
    """

    metadata = {"render_modes": ["human"], "render_fps": 8}

    ACTIONS: Tuple[Coordinate, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    START_PATTERNS: Sequence[str] = ("line", "cube", "spiral", "random")

    def __init__(
        self,
        grid_size: int = 15,
        render_mode: Optional[str] = None,
        show_window: bool = False,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert 10 <= grid_size <= 20, "grid_size must be between 10 and 20"
        self.grid_size = grid_size
        self.reward_cfg = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.show_window = show_window and render_mode == "human"

        obs_shape = (grid_size, grid_size, 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.clock: Optional[pygame.time.Clock] = None
        self.surface: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None

        self.rng, _ = seeding.np_random(seed)
        self.seed_value = seed

        self.snake: List[Coordinate] = []
        self.direction: Action = 1
        self.pending_growth: int = 0
        self.fruit: Coordinate = (0, 0)
        self.steps_since_reset: int = 0
        self.fruits_eaten: int = 0
        self.loop_history: Deque[Action] = deque(maxlen=12)
        self.episode_reward: float = 0.0

        self._init_pygame_if_needed()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        """Set deterministic seed while matching the Gymnasium protocol."""

        self.rng, _ = seeding.np_random(seed)
        self.seed_value = seed

    def set_rendering(self, render_mode: Optional[str], show_window: bool) -> None:
        """Utility used by the training scripts to toggle rendering at runtime."""

        self.render_mode = render_mode
        self.show_window = show_window and render_mode == "human"
        if self.show_window:
            self._init_pygame_if_needed()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.steps_since_reset = 0
        self.fruits_eaten = 0
        self.loop_history.clear()
        self.pending_growth = 0
        self.episode_reward = 0.0
        self._spawn_snake(options or {})
        self._spawn_fruit()
        observation = self._get_observation()
        info = {
            "pattern": self.start_pattern,
            "seed": self.seed_value,
        }
        return observation, info

    def step(self, action: Action):
        assert self.action_space.contains(action)
        reward = self.reward_cfg.step_penalty
        terminated = False
        truncated = False

        if self._is_opposite_direction(action):
            action = self.direction
        self.direction = action
        self.loop_history.append(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.ACTIONS[action]
        new_head = (head_x + dx, head_y + dy)

        # Wall detection – grant penalty and end episode just like HTML.
        if not self._within_bounds(new_head):
            reward += self.reward_cfg.wall_penalty
            reward += self.reward_cfg.death_penalty
            terminated = True
        else:
            if new_head in self.snake:
                reward += self.reward_cfg.death_penalty
                terminated = True
            else:
                self.snake.insert(0, new_head)
                if new_head == self.fruit:
                    self.pending_growth += 1
                    self.fruits_eaten += 1
                    reward += self.reward_cfg.fruit_reward
                    self._spawn_fruit()
                if self.pending_growth > 0:
                    self.pending_growth -= 1
                else:
                    self.snake.pop()

        self.steps_since_reset += 1

        if terminated:
            self.episode_reward += reward
            observation = self._get_observation()
            info = self._build_info(done=True)
            return observation, reward, terminated, truncated, info

        reward += self._loop_penalty()
        reward += self._compactness_bonus()

        self.episode_reward += reward
        observation = self._get_observation()
        info = self._build_info(done=False)

        if self.show_window and self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if not self.show_window or self.render_mode != "human":
            return

        if self.surface is None:
            self._init_pygame_if_needed()

        cell_size = 28
        margin = 20
        width = self.grid_size * cell_size + margin * 2
        height = self.grid_size * cell_size + margin * 2
        screen = self.surface
        screen.fill((15, 15, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    margin + x * cell_size,
                    margin + y * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(screen, (30, 30, 50), rect, 1)

        for i, (sx, sy) in enumerate(self.snake):
            color = (60, 200, 120) if i else (230, 240, 90)
            rect = pygame.Rect(
                margin + sx * cell_size,
                margin + sy * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(screen, color, rect)

        fx, fy = self.fruit
        fruit_rect = pygame.Rect(
            margin + fx * cell_size,
            margin + fy * cell_size,
            cell_size,
            cell_size,
        )
        pygame.draw.rect(screen, (220, 90, 110), fruit_rect)

        status = (
            f"Snake-ML | Pattern: {self.start_pattern} | "
            f"Fruits: {self.fruits_eaten} | Steps: {self.steps_since_reset}"
        )
        pygame.display.set_caption(status)

        if self.font:
            text_surf = self.font.render(status, True, (220, 220, 220))
            screen.blit(text_surf, (margin, 4))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.surface = None
            self.clock = None
            self.font = None
            self.show_window = False

    # ------------------------------------------------------------------
    # Internal helpers mirroring the JS version
    # ------------------------------------------------------------------
    def _init_pygame_if_needed(self) -> None:
        if not self.show_window:
            return
        if not pygame.get_init():
            pygame.init()
        self.clock = pygame.time.Clock()
        window_size = (self.grid_size * 28 + 40, self.grid_size * 28 + 40)
        self.surface = pygame.display.set_mode(window_size)
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 16)
        except Exception:
            self.font = None

    def _spawn_snake(self, options: dict) -> None:
        pattern = options.get("pattern") or str(self.rng.choice(np.array(self.START_PATTERNS)))
        self.start_pattern = pattern
        cx = self.grid_size // 2
        cy = self.grid_size // 2

        if pattern == "line":
            length = max(3, self.grid_size // 3)
            start_x = int(np.clip(cx - length // 2, 1, self.grid_size - length - 1))
            self.snake = [(start_x + i, cy) for i in range(length)]
            self.direction = 1
        elif pattern == "cube":
            half = max(2, self.grid_size // 4)
            points: List[Coordinate] = []
            for dy in range(-half, half):
                for dx in range(-half, half):
                    nx = int(np.clip(cx + dx, 1, self.grid_size - 2))
                    ny = int(np.clip(cy + dy, 1, self.grid_size - 2))
                    points.append((nx, ny))
            self.snake = list(dict.fromkeys(points))
            self.direction = 1
        elif pattern == "spiral":
            self.snake = self._generate_spiral(cx, cy)
            self.direction = 0
        else:  # random scatter
            length = max(4, self.grid_size // 2)
            self.snake = []
            used = set()
            while len(self.snake) < length:
                pos = (self.rng.integers(1, self.grid_size - 1), self.rng.integers(1, self.grid_size - 1))
                if pos not in used:
                    self.snake.append(pos)
                    used.add(pos)
            self.snake.sort()
            self.direction = int(self.rng.integers(0, 4))

        self.snake = [self._wrap_position(p) for p in self.snake]

    def _spawn_fruit(self) -> None:
        available = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))
        for segment in self.snake:
            available.discard(segment)
        if not available:
            self.fruit = self.snake[0]
            return
        options = list(available)
        idx = int(self.rng.integers(0, len(options)))
        self.fruit = options[idx]

    def _generate_spiral(self, cx: int, cy: int) -> List[Coordinate]:
        path: List[Coordinate] = []
        radius = min(cx, cy) - 1
        x, y = cx, cy
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        step_length = 1
        while radius > 0 and len(path) < self.grid_size * self.grid_size:
            for d in directions:
                for _ in range(step_length):
                    x += d[0]
                    y += d[1]
                    if not self._within_bounds((x, y)):
                        continue
                    path.append((x, y))
                step_length += 1
            radius -= 1
        if not path:
            path = [(cx, cy)]
        return path[: max(3, self.grid_size // 2)]

    def _get_observation(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            grid[y, x, 0] = 1.0 if i else 0.75
        fx, fy = self.fruit
        grid[fy, fx, 1] = 1.0
        self._encode_distance_field(grid)
        return grid

    def _encode_distance_field(self, grid: np.ndarray) -> None:
        fx, fy = self.fruit
        norm = self.grid_size * 2
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                dist = abs(fx - x) + abs(fy - y)
                grid[y, x, 2] = 1.0 - dist / norm

    def _within_bounds(self, pos: Coordinate) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_opposite_direction(self, action: Action) -> bool:
        return (self.direction + 2) % 4 == action

    def _wrap_position(self, pos: Coordinate) -> Coordinate:
        x, y = pos
        return (int(np.clip(x, 0, self.grid_size - 1)), int(np.clip(y, 0, self.grid_size - 1)))

    def _compactness_bonus(self) -> float:
        cx = cy = self.grid_size / 2
        head_x, head_y = self.snake[0]
        distance = math.hypot(head_x - cx, head_y - cy)
        max_distance = math.hypot(cx, cy)
        compact_score = 1.0 - (distance / max_distance)
        return self.reward_cfg.compact_bonus * compact_score

    def _loop_penalty(self) -> float:
        penalty = 0.0
        pattern = self._detect_loop_pattern(self.loop_history)
        if pattern:
            penalty += self.reward_cfg.loop_penalty
        return penalty

    def _detect_loop_pattern(self, history: Deque[Action]) -> Optional[Tuple[Action, ...]]:
        if len(history) < 6:
            return None
        counts = Counter(history)
        if len(counts) <= 2:
            seq = tuple(history)
            if all(seq[i] == seq[i % 2] for i in range(len(seq))):
                return seq[:2]
        if len(history) >= 8:
            seq = tuple(history)
            half = len(seq) // 2
            if seq[:half] == seq[half:]:
                return seq[:half]
        return None

    def _build_info(self, done: bool) -> dict:
        info = {
            "fruits": self.fruits_eaten,
            "steps": self.steps_since_reset,
            "pattern": self.start_pattern,
        }
        if done:
            info["episode"] = {
                "r": float(self.episode_reward),
                "l": self.steps_since_reset,
            }
        return info


__all__ = ["SnakeEnv", "RewardConfig"]
