"""
Symbol Drift Monitor - Track how symbol meanings change over time.

As agents evolve communication patterns across generations/checkpoints,
symbol meanings may shift. This module detects and quantifies drift.

Key metrics:
- Symbol→action mapping stability
- Per-symbol usage frequency change
- Overall semantic drift distance
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


@dataclass
class DriftReport:
    """Report comparing symbol usage between two snapshots."""
    snapshot_a: str
    snapshot_b: str
    num_symbols: int
    usage_overlap: float           # Jaccard similarity of active symbols
    meaning_stability: float       # avg stability of symbol→action mapping
    frequency_correlation: float   # correlation of symbol frequency ranks
    drifted_symbols: List[int]     # symbols whose meaning changed significantly
    summary: str = ""


class SymbolSnapshot:
    """Capture symbol usage patterns at a point in time."""

    def __init__(self, name: str, num_symbols: int = 32, num_actions: int = 10):
        self.name = name
        self.num_symbols = num_symbols
        self.num_actions = num_actions
        self.symbol_counts: Counter = Counter()
        self.symbol_action_map: Dict[int, Counter] = defaultdict(Counter)

    def record(self, symbols: torch.Tensor, action: int):
        """Record symbol usage with associated action."""
        for sym in symbols.tolist():
            self.symbol_counts[sym] += 1
            self.symbol_action_map[sym][action] += 1

    def active_symbols(self) -> set:
        return set(self.symbol_counts.keys())

    def action_distribution(self, symbol: int) -> List[float]:
        """Normalized action distribution for a symbol."""
        counts = self.symbol_action_map.get(symbol, Counter())
        total = sum(counts.values())
        if total == 0:
            return [0.0] * self.num_actions
        return [counts.get(a, 0) / total for a in range(self.num_actions)]

    def frequency_vector(self) -> np.ndarray:
        """Symbol frequency as a vector."""
        vec = np.zeros(self.num_symbols)
        total = sum(self.symbol_counts.values())
        if total > 0:
            for sym, count in self.symbol_counts.items():
                if sym < self.num_symbols:
                    vec[sym] = count / total
        return vec


class DriftMonitor:
    """
    Monitor symbol drift across snapshots (generations, checkpoints, etc.).

    Usage:
        monitor = DriftMonitor()
        # After generation 1:
        monitor.add_snapshot("gen_1", snapshot_1)
        # After generation 5:
        monitor.add_snapshot("gen_5", snapshot_5)
        # Compare:
        report = monitor.compare("gen_1", "gen_5")
    """

    def __init__(self, num_symbols: int = 32, num_actions: int = 10):
        self.num_symbols = num_symbols
        self.num_actions = num_actions
        self.snapshots: Dict[str, SymbolSnapshot] = {}

    def create_snapshot(self, name: str) -> SymbolSnapshot:
        """Create a new empty snapshot."""
        snap = SymbolSnapshot(name, self.num_symbols, self.num_actions)
        self.snapshots[name] = snap
        return snap

    def add_snapshot(self, name: str, snapshot: SymbolSnapshot):
        """Add a pre-built snapshot."""
        self.snapshots[name] = snapshot

    def compare(self, name_a: str, name_b: str, drift_threshold: float = 0.3) -> DriftReport:
        """
        Compare two snapshots and produce a drift report.

        Args:
            name_a: First snapshot name (earlier)
            name_b: Second snapshot name (later)
            drift_threshold: JS-divergence threshold for flagging a symbol as drifted
        """
        a = self.snapshots[name_a]
        b = self.snapshots[name_b]

        # 1. Usage overlap (Jaccard similarity)
        active_a = a.active_symbols()
        active_b = b.active_symbols()
        if active_a or active_b:
            overlap = len(active_a & active_b) / len(active_a | active_b)
        else:
            overlap = 1.0

        # 2. Meaning stability (per shared symbol)
        shared = active_a & active_b
        stabilities = []
        drifted = []

        for sym in shared:
            dist_a = np.array(a.action_distribution(sym))
            dist_b = np.array(b.action_distribution(sym))

            # Jensen-Shannon divergence
            m = (dist_a + dist_b) / 2
            # Clip to avoid log(0)
            eps = 1e-10
            kl_am = np.sum(dist_a * np.log((dist_a + eps) / (m + eps)))
            kl_bm = np.sum(dist_b * np.log((dist_b + eps) / (m + eps)))
            jsd = (kl_am + kl_bm) / 2

            stability = 1.0 - min(jsd, 1.0)
            stabilities.append(stability)

            if jsd > drift_threshold:
                drifted.append(sym)

        avg_stability = np.mean(stabilities) if stabilities else 1.0

        # 3. Frequency rank correlation
        freq_a = a.frequency_vector()
        freq_b = b.frequency_vector()
        if freq_a.sum() > 0 and freq_b.sum() > 0:
            # Spearman rank correlation
            rank_a = np.argsort(np.argsort(-freq_a))
            rank_b = np.argsort(np.argsort(-freq_b))
            n = len(rank_a)
            d_sq = np.sum((rank_a - rank_b) ** 2)
            freq_corr = 1 - (6 * d_sq) / (n * (n**2 - 1))
        else:
            freq_corr = 0.0

        # Summary
        summary_parts = []
        if overlap < 0.5:
            summary_parts.append(f"LOW symbol overlap ({overlap:.0%})")
        if avg_stability < 0.7:
            summary_parts.append(f"SIGNIFICANT meaning drift (stability={avg_stability:.2f})")
        if drifted:
            summary_parts.append(f"{len(drifted)} symbols drifted: {drifted[:10]}")
        if not summary_parts:
            summary_parts.append("Stable — no significant drift detected")

        return DriftReport(
            snapshot_a=name_a,
            snapshot_b=name_b,
            num_symbols=self.num_symbols,
            usage_overlap=round(overlap, 4),
            meaning_stability=round(float(avg_stability), 4),
            frequency_correlation=round(float(freq_corr), 4),
            drifted_symbols=sorted(drifted),
            summary="; ".join(summary_parts),
        )

    def trajectory(self, snapshot_names: Optional[List[str]] = None) -> List[DriftReport]:
        """Compare consecutive snapshots to see drift trajectory."""
        if snapshot_names is None:
            snapshot_names = list(self.snapshots.keys())
        reports = []
        for i in range(len(snapshot_names) - 1):
            reports.append(self.compare(snapshot_names[i], snapshot_names[i + 1]))
        return reports
