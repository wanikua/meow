"""
Task 1: Cooperative Code Refactoring

Two agents collaborate on code refactoring decisions.
- Architect: sees the requirements (what needs to change)
- Implementer: sees the current code state (what exists)
- Together they must agree on the correct refactoring action.

The task is abstracted as a coordination game over embeddings:
each agent sees a partial view and must communicate to select
the same target action from a shared action space.
"""

import torch
from typing import Dict, List, Tuple

from .harness import TaskEnvironment


class CodingTask(TaskEnvironment):
    """
    Cooperative coding task.

    Setup:
    - N refactoring scenarios, each with:
      - requirement_embedding (visible to architect)
      - code_embedding (visible to implementer)
      - target_action (the correct refactoring choice)
    - Agents must communicate to both select the target action.

    The embeddings are generated to have learnable structure:
    the correct action is determined by combining both views.
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 10,
        num_scenarios: int = 100,
        seed: int = 42,
    ):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.num_scenarios = num_scenarios

        rng = torch.Generator().manual_seed(seed)

        # Generate scenario data:
        # Each scenario has a requirement view, code view, and target action.
        # The target is determined by: hash(requirement_cluster, code_cluster) % action_dim
        # This forces agents to combine both views.

        num_req_types = action_dim
        num_code_types = action_dim

        # Cluster centers
        self.req_centers = torch.randn(num_req_types, obs_dim, generator=rng) * 2
        self.code_centers = torch.randn(num_code_types, obs_dim, generator=rng) * 2

        # Generate scenarios
        self.scenarios = []
        for _ in range(num_scenarios):
            req_type = torch.randint(num_req_types, (1,), generator=rng).item()
            code_type = torch.randint(num_code_types, (1,), generator=rng).item()
            target = (req_type * 3 + code_type * 7) % action_dim

            req_obs = self.req_centers[req_type] + torch.randn(obs_dim, generator=rng) * 0.3
            code_obs = self.code_centers[code_type] + torch.randn(obs_dim, generator=rng) * 0.3

            self.scenarios.append((req_obs, code_obs, target))

        self._current_scenario = None

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, batch_size: int = 1) -> List[torch.Tensor]:
        """Sample a random scenario and return observations."""
        idx = torch.randint(self.num_scenarios, (1,)).item()
        req_obs, code_obs, target = self.scenarios[idx]
        self._current_scenario = (req_obs, code_obs, target)
        return [req_obs, code_obs]

    def evaluate(self, actions: List[torch.Tensor]) -> Tuple[float, Dict]:
        """
        Evaluate: both agents must select the correct target action.

        Full success (1.0) only if both agents agree on the correct action.
        Partial credit (0.3) if they agree but are wrong.
        """
        target = self._current_scenario[2]
        architect_action = actions[0].item()
        implementer_action = actions[1].item()

        both_correct = (architect_action == target) and (implementer_action == target)
        agree = architect_action == implementer_action
        any_correct = (architect_action == target) or (implementer_action == target)

        if both_correct:
            success = 1.0
        elif agree and any_correct:
            success = 0.5
        elif agree:
            success = 0.3
        elif any_correct:
            success = 0.2
        else:
            success = 0.0

        return success, {
            "target": target,
            "architect_action": architect_action,
            "implementer_action": implementer_action,
            "both_correct": both_correct,
            "agree": agree,
        }
