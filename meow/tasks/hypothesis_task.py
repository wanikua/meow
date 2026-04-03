"""
Task 3: Parallel Hypothesis Exploration

Five agents each see a different partition of a dataset.
They must communicate to converge on a consistent explanation.

Abstracted as: the "dataset" is a feature vector, each agent sees
a different projection. The correct hypothesis is determined by
the full feature vector. Agents must pool information.
"""

import torch
from typing import Dict, List, Tuple

from .harness import TaskEnvironment


class HypothesisTask(TaskEnvironment):
    """
    Parallel hypothesis exploration.

    Setup:
    - 5 agents each see a different random projection of the full data
    - The correct hypothesis = f(full data)
    - Agents broadcast to all others (public channel)
    - Success = majority converges on the correct hypothesis

    Designed to test:
    - Efficient information sharing (broadcast)
    - Confidence signaling (strong vs weak evidence)
    - Consensus building
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 6,
        num_scenarios: int = 300,
        n_agents: int = 5,
        seed: int = 456,
    ):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._n_agents = n_agents
        self.num_scenarios = num_scenarios

        rng = torch.Generator().manual_seed(seed)

        # Full data dimension
        full_dim = obs_dim * n_agents

        # Per-agent random projection from full data to obs_dim
        # Each agent sees data through a different "lens"
        self.projections = [
            torch.randn(full_dim, obs_dim, generator=rng) / (full_dim ** 0.5)
            for _ in range(n_agents)
        ]

        # Mapping from full data to correct hypothesis
        self.answer_proj = torch.randn(full_dim, action_dim, generator=rng)

        self.scenarios = []
        for _ in range(num_scenarios):
            full_data = torch.randn(full_dim, generator=rng)
            target = (full_data @ self.answer_proj).argmax().item()

            views = [full_data @ proj for proj in self.projections]
            self.scenarios.append((views, target))

        self._current = None

    @property
    def num_agents(self) -> int:
        return self._n_agents

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, batch_size: int = 1) -> List[torch.Tensor]:
        idx = torch.randint(self.num_scenarios, (1,)).item()
        views, target = self.scenarios[idx]
        self._current = (views, target)
        return list(views)

    def evaluate(self, actions: List[torch.Tensor]) -> Tuple[float, Dict]:
        """
        Evaluate based on consensus and correctness.

        Rewards:
        - 1.0: all agents correct
        - 0.8: majority correct
        - 0.3: all agree but wrong
        - diversity_bonus: explored multiple hypotheses before converging
        """
        target = self._current[1]
        agent_actions = [a.item() for a in actions]

        correct = [a == target for a in agent_actions]
        num_correct = sum(correct)

        from collections import Counter
        vote_counts = Counter(agent_actions)
        majority_action, majority_count = vote_counts.most_common(1)[0]
        unique_actions = len(vote_counts)

        if num_correct == self._n_agents:
            success = 1.0
        elif num_correct >= (self._n_agents + 1) // 2:
            success = 0.8
        elif majority_count == self._n_agents:  # unanimous but wrong
            success = 0.3
        elif num_correct > 0:
            success = 0.1 * num_correct
        else:
            success = 0.0

        # Small diversity bonus for exploring multiple hypotheses
        diversity_bonus = min(unique_actions / self._action_dim, 0.5) * 0.1

        return success + diversity_bonus, {
            "target": target,
            "agent_actions": agent_actions,
            "num_correct": num_correct,
            "majority_action": majority_action,
            "majority_count": majority_count,
            "unique_actions": unique_actions,
            "diversity_bonus": diversity_bonus,
        }
