"""Utilities for aggregating and formatting reward component telemetry."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import math
import numpy as np

WINDOW_SIZES: Sequence[int] = (1, 10, 100, 1000)

def compute_component_shares(
    components: Sequence[str],
    values: Mapping[str, float],
    *,
    total_key: str = "total",
) -> Dict[str, Optional[float]]:
    """Return percentage share for each component based on absolute magnitudes."""

    magnitudes = {
        component: abs(float(values.get(component, 0.0)))
        for component in components
        if component != total_key
    }
    denominator = sum(magnitudes.values())
    if denominator <= 0.0:
        return {component: None for component in components}

    shares: Dict[str, Optional[float]] = {}
    for component in components:
        if component == total_key:
            shares[component] = None
            continue
        shares[component] = (magnitudes.get(component, 0.0) / denominator) * 100.0
    return shares


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
        last_values: Dict[str, float] = {}
        cumulative_values: Dict[str, float] = {}
        for component in self.components:
            comp_windows = self.component_windows[component]
            last = comp_windows[1][-1] if comp_windows[1] else math.nan
            last_values[component] = 0.0 if math.isnan(last) else float(last)
            cumulative_values[component] = float(sum(self.component_history[component]))
            summary[component] = {
                "last": last,
                **{f"avg_{size}": self._mean(comp_windows[size]) for size in self.window_sizes},
                "std": self._std(self.component_history[component]),
            }
        last_shares = compute_component_shares(self.components, last_values)
        cumulative_shares = compute_component_shares(self.components, cumulative_values)
        for component in self.components:
            summary[component]["share_last"] = last_shares.get(component)
            summary[component]["share_total"] = cumulative_shares.get(component)
        return summary

    def format_table(self) -> str:
        stats = self.stats()
        header = "Component        Last    Avg10    Avg100   Avg1000     Std  Step%  Epis%"
        divider = "-" * len(header)
        lines = ["Reward component trends:", header, divider]
        for component in self.components:
            comp_stats = stats[component]
            line = (
                f"{self._format_component_label(component):15s}"
                f" {self._format_value(comp_stats['last']):>7}"
                f" {self._format_value(comp_stats['avg_10']):>8}"
                f" {self._format_value(comp_stats['avg_100']):>8}"
                f" {self._format_value(comp_stats['avg_1000']):>9}"
                f" {self._format_value(comp_stats['std']):>9}"
                f" {self._format_share(comp_stats['share_last']):>6}"
                f" {self._format_share(comp_stats['share_total']):>6}"
            )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_value(value: float) -> str:
        if math.isnan(value):
            return "   -"
        return f"{value:6.2f}"

    @staticmethod
    def _format_component_label(component: str) -> str:
        return component.replace("_", " ").title()

    @staticmethod
    def _format_share(value: Optional[float]) -> str:
        if value is None:
            return "   -"
        return f"{value:5.1f}%"


__all__ = ["RewardTelemetryTracker", "WINDOW_SIZES", "compute_component_shares"]
