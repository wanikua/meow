"""
Meow Tasks - Multi-agent task framework for emergent communication.

Provides task environments, agent policies, communication channels,
and training infrastructure for studying how agents develop communication
patterns through the Meow codebook under task pressure.
"""

from .harness import MeowAgent, MeowChannel, TaskEnvironment, TaskRunner
from .rewards import task_reward, communication_cost, combined_reward
from .coding_task import CodingTask
from .logic_task import LogicTask
from .hypothesis_task import HypothesisTask

__all__ = [
    "MeowAgent",
    "MeowChannel",
    "TaskEnvironment",
    "TaskRunner",
    "task_reward",
    "communication_cost",
    "combined_reward",
    "CodingTask",
    "LogicTask",
    "HypothesisTask",
]
