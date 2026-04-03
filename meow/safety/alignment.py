"""
Alignment Penalty - Detect and penalize say-do mismatch.

Compares what agents communicate (decoded message intent) with
what they actually do (observed actions). Penalizes divergence.

This is the core safety mechanism: if agents develop deceptive
communication (saying one thing, doing another), this catches it.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..tasks.harness import Message, EpisodeResult


@dataclass
class SayDoRecord:
    """Record of what an agent said vs. what it did."""
    agent_id: int
    symbols_sent: List[int]
    action_taken: int
    round: int


class SayDoTracker:
    """
    Track say-do consistency across episodes.

    Builds a mapping: (symbols → expected action distribution).
    Flags agents whose actions diverge from what their symbols predict.
    """

    def __init__(self, num_symbols: int = 32, num_actions: int = 10):
        self.num_symbols = num_symbols
        self.num_actions = num_actions
        # symbol_key → action histogram
        self.symbol_action_map: Dict[tuple, List[int]] = defaultdict(
            lambda: [0] * num_actions
        )
        self.records: List[SayDoRecord] = []

    def record(self, agent_id: int, symbols: torch.Tensor, action: int, round_idx: int = 0):
        """Record a say-do pair."""
        sym_key = tuple(sorted(symbols.tolist()))
        self.symbol_action_map[sym_key][action] += 1
        self.records.append(SayDoRecord(
            agent_id=agent_id,
            symbols_sent=symbols.tolist(),
            action_taken=action,
            round=round_idx,
        ))

    def record_episode(self, result: EpisodeResult, actions: List[int]):
        """Record all say-do pairs from an episode."""
        # Map sender → messages
        agent_messages: Dict[int, List[Message]] = defaultdict(list)
        for msg in result.messages:
            agent_messages[msg.sender].append(msg)

        for agent_id, action in enumerate(actions):
            msgs = agent_messages.get(agent_id, [])
            if msgs:
                # Use last message as representative
                self.record(agent_id, msgs[-1].symbols, action)

    def consistency_score(self, symbols: torch.Tensor, action: int) -> float:
        """
        How consistent is this (symbols, action) pair with history?

        Returns 0-1: 1.0 = perfectly consistent, 0.0 = never seen before.
        """
        sym_key = tuple(sorted(symbols.tolist()))
        hist = self.symbol_action_map.get(sym_key)
        if hist is None or sum(hist) == 0:
            return 0.5  # unknown — neutral
        total = sum(hist)
        return hist[action] / total

    def detect_anomalies(self, threshold: float = 0.1) -> List[SayDoRecord]:
        """Find records where action was inconsistent with symbol history."""
        anomalies = []
        for record in self.records:
            sym_key = tuple(sorted(record.symbols_sent))
            hist = self.symbol_action_map.get(sym_key)
            if hist is None:
                continue
            total = sum(hist)
            if total < 5:  # need enough data
                continue
            freq = hist[record.action_taken] / total
            if freq < threshold:
                anomalies.append(record)
        return anomalies

    def per_agent_consistency(self) -> Dict[int, float]:
        """Compute average consistency score per agent."""
        agent_scores: Dict[int, List[float]] = defaultdict(list)
        for record in self.records:
            score = self.consistency_score(
                torch.tensor(record.symbols_sent), record.action_taken
            )
            agent_scores[record.agent_id].append(score)

        return {
            agent_id: sum(scores) / len(scores)
            for agent_id, scores in agent_scores.items()
            if scores
        }

    def summary(self) -> Dict:
        """Summary statistics."""
        per_agent = self.per_agent_consistency()
        anomalies = self.detect_anomalies()
        return {
            "total_records": len(self.records),
            "unique_symbol_patterns": len(self.symbol_action_map),
            "per_agent_consistency": per_agent,
            "num_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(self.records) if self.records else 0,
        }


class AlignmentPenalty:
    """
    Compute alignment penalty for the reward function.

    penalty = -weight * (1 - consistency_score)

    Agents that use symbols inconsistently with their established
    meaning get penalized. This discourages deceptive communication.
    """

    def __init__(self, weight: float = 10.0, warmup_episodes: int = 50):
        self.weight = weight
        self.warmup_episodes = warmup_episodes
        self.tracker = SayDoTracker()
        self.episode_count = 0

    def update_and_penalize(
        self,
        messages: List[Message],
        actions: List[int],
    ) -> float:
        """
        Record episode data and compute alignment penalty.

        Returns penalty value (non-negative, to be subtracted from reward).
        """
        self.episode_count += 1

        # Record
        agent_messages: Dict[int, List[Message]] = defaultdict(list)
        for msg in messages:
            agent_messages[msg.sender].append(msg)

        penalties = []
        for agent_id, action in enumerate(actions):
            msgs = agent_messages.get(agent_id, [])
            if msgs:
                syms = msgs[-1].symbols
                # Compute consistency before recording (so we check against history)
                if self.episode_count > self.warmup_episodes:
                    consistency = self.tracker.consistency_score(syms, action)
                    penalties.append(1.0 - consistency)
                self.tracker.record(agent_id, syms, action)

        if not penalties:
            return 0.0

        avg_penalty = sum(penalties) / len(penalties)
        return self.weight * avg_penalty
