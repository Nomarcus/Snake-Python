"""Utilities for aggregating and formatting reward component telemetry."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Sequence

import math
import numpy as np

WINDOW_SIZES: Sequence[int] = (1, 10, 100, 1000)


@dataclass
class RewardTelemetryTracker:
    """Track reward component trends across multiple rolling windows."""

    components: Sequence[str]
    window_sizes: Sequence[int] = WINDOW_SIZES
    component_windows: MutableMapping[str, Dict[int, deque]] = field(init=False)
    component_history: MutableMapping[str, List[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.component_windows = {
            component: {size: deque(maxlen=size) for size in self.window_sizes}
            for component in self.components
        }
        self.component_history = {component: [] for component in self.components}

    def update(self, breakdown: Mapping[str, float]) -> None:
        for component in self.components:
            value = float(breakdown.get(component, 0.0))
            for window_size in self.window_sizes:
                self.component_windows[component][window_size].append(value)
            self.component_history[component].append(value)

    def _mean(self, values: deque) -> float:
        if not values:
            return math.nan
        return float(np.mean(values))

    def _std(self, values: List[float]) -> float:
        if not values:
            return math.nan
        return float(np.std(values))

    def stats(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for component in self.components:
            comp_windows = self.component_windows[component]
            last = comp_windows[1][-1] if comp_windows[1] else math.nan
            summary[component] = {
                "last": last,
                **{f"avg_{size}": self._mean(comp_windows[size]) for size in self.window_sizes},
                "std": self._std(self.component_history[component]),
            }
        return summary

    def format_table(self) -> str:
        stats = self.stats()
        header = "Component        Last    Avg10    Avg100   Avg1000     Std"
        lines = ["Reward component trends:", header]
        for component in self.components:
            comp_stats = stats[component]
            line = (
                f"{component:15s}"
                f" {self._format_value(comp_stats['last']):>7}"
                f" {self._format_value(comp_stats['avg_10']):>8}"
                f" {self._format_value(comp_stats['avg_100']):>8}"
                f" {self._format_value(comp_stats['avg_1000']):>9}"
                f" {self._format_value(comp_stats['std']):>9}"
            )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_value(value: float) -> str:
        if math.isnan(value):
            return "   -"
        return f"{value:6.2f}"


__all__ = ["RewardTelemetryTracker", "WINDOW_SIZES"]
