"""
Task 2: Distributed Logic Puzzle

Three agents each receive partial clues and must collaborate
to solve a constraint satisfaction puzzle.

Abstracted as: each agent sees a portion of the input features.
The correct answer depends on ALL features combined.
No single agent has enough information to solve it alone.
"""

import torch
from typing import Dict, List, Tuple

from .harness import TaskEnvironment


class LogicTask(TaskEnvironment):
    """
    Distributed logic puzzle.

    Setup:
    - A hidden state vector determines the answer
    - Each of 3 agents sees a different slice of the state
    - The correct action = f(all slices combined)
    - Agents must communicate their clues to solve the puzzle

    The mapping from combined state to answer is a learned function,
    making it impossible to solve without information exchange.
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 8,
        num_puzzles: int = 200,
        seed: int = 123,
    ):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.num_puzzles = num_puzzles

        rng = torch.Generator().manual_seed(seed)

        # Each puzzle: full_state → split into 3 parts → answer from full_state
        # Agent i sees state[i * slice : (i+1) * slice]
        self.slice_size = obs_dim  # each agent gets obs_dim features
        full_dim = obs_dim * 3

        # A deterministic mapping: full_state → answer
        # Use a fixed random projection + argmax
        self.projection = torch.randn(full_dim, action_dim, generator=rng)

        self.puzzles = []
        for _ in range(num_puzzles):
            full_state = torch.randn(full_dim, generator=rng)
            logits = full_state @ self.projection
            target = logits.argmax().item()

            # Split into 3 agent views
            views = [
                full_state[i * obs_dim : (i + 1) * obs_dim]
                for i in range(3)
            ]
            self.puzzles.append((views, target))

        self._current_puzzle = None

    @property
    def num_agents(self) -> int:
        return 3

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, batch_size: int = 1) -> List[torch.Tensor]:
        idx = torch.randint(self.num_puzzles, (1,)).item()
        views, target = self.puzzles[idx]
        self._current_puzzle = (views, target)
        return list(views)

    def evaluate(self, actions: List[torch.Tensor]) -> Tuple[float, Dict]:
        """
        All three agents must agree on the correct answer.

        Full credit (1.0) if majority correct.
        Partial credit for agreement.
        """
        target = self._current_puzzle[1]
        agent_actions = [a.item() for a in actions]

        correct = [a == target for a in agent_actions]
        num_correct = sum(correct)
        # Majority vote
        from collections import Counter
        vote_counts = Counter(agent_actions)
        majority_action, majority_count = vote_counts.most_common(1)[0]

        if num_correct == 3:
            success = 1.0
        elif num_correct >= 2:
            success = 0.7
        elif majority_count == 3:  # all agree but wrong
            success = 0.2
        elif num_correct == 1:
            success = 0.1
        else:
            success = 0.0

        return success, {
            "target": target,
            "agent_actions": agent_actions,
            "num_correct": num_correct,
            "majority_action": majority_action,
            "unanimous": majority_count == 3,
        }
