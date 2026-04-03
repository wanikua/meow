"""
Reward functions for Meow multi-agent tasks.

Design principle: communication is a COST, not a goal.
Agents minimize communication naturally while pursuing task success.
"""

from typing import Dict, List, Optional
from .harness import EpisodeResult


def task_reward(success: float, scale: float = 100.0) -> float:
    """Base task reward. Success is typically 0-1."""
    return success * scale


def communication_cost(
    num_symbols: int,
    cost_per_symbol: float = 0.1,
) -> float:
    """Cost proportional to total symbols sent."""
    return num_symbols * cost_per_symbol


def redundancy_penalty(
    messages: list,
    penalty_per_duplicate: float = 5.0,
) -> float:
    """Penalty for sending the same symbol sequence multiple times."""
    seen = set()
    duplicates = 0
    for msg in messages:
        key = tuple(msg.symbols.tolist())
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates * penalty_per_duplicate


def combined_reward(
    result: EpisodeResult,
    success_weight: float = 100.0,
    comm_weight: float = 0.1,
    redundancy_weight: float = 5.0,
) -> float:
    """
    Combined reward function.

    reward = task_success * 100 - communication_cost * 0.1 - redundancy * 5
    """
    r = task_reward(result.task_success, success_weight)
    c = communication_cost(result.info.get("total_symbols", 0), comm_weight)
    p = redundancy_penalty(result.messages, redundancy_weight)
    return r - c - p
