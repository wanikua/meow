"""
Multi-agent task harness for Meow communication experiments.

Core abstractions:
- MeowAgent: Neural agent that observes, communicates via Meow symbols, and acts
- MeowChannel: Communication channel with budget constraints and noise
- TaskEnvironment: Base class for multi-agent tasks
- TaskRunner: Orchestrates episodes of agent interaction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Message:
    """A Meow message between agents."""
    sender: int
    symbols: torch.Tensor      # (num_symbols,) discrete indices
    log_prob: torch.Tensor      # log probability of this message (for REINFORCE)
    round: int = 0


@dataclass
class ChannelConfig:
    """Communication channel configuration."""
    max_symbols_per_message: int = 3
    max_messages_per_round: int = 1
    max_rounds: int = 10
    budget_per_agent: int = 30       # total symbols allowed across all rounds
    drop_rate: float = 0.0           # probability of dropping a message
    noise_std: float = 0.0           # noise added to symbol embeddings in transit


@dataclass
class EpisodeResult:
    """Result of one task episode."""
    task_success: float
    communication_cost: float
    total_reward: float
    num_rounds: int
    messages: List[Message] = field(default_factory=list)
    info: Dict = field(default_factory=dict)


class MeowAgent(nn.Module):
    """
    Neural agent that communicates via Meow symbols.

    Architecture:
        observation → obs_encoder → hidden state
        hidden state + received messages → message_head → discrete symbols
        hidden state + received messages → action_head → task action
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        num_symbols: int = 512,
        max_symbols_per_msg: int = 3,
        action_dim: int = 10,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.max_symbols_per_msg = max_symbols_per_msg

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message encoder (process received messages)
        self.msg_encoder = nn.Sequential(
            nn.Linear(num_symbols, hidden_dim),
            nn.ReLU(),
        )

        # Message generation head: produces symbol logits
        self.message_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_symbols * max_symbols_per_msg),
        )

        # Action head: produces task action logits
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to hidden state."""
        return self.obs_encoder(obs)

    def encode_messages(self, messages: List[Message]) -> torch.Tensor:
        """Encode received messages into a fixed-size representation."""
        device = next(self.parameters()).device
        agg = torch.zeros(1, self.num_symbols, device=device)
        for msg in messages:
            for sym in msg.symbols:
                if 0 <= sym.item() < self.num_symbols:
                    agg[0, sym.item()] += 1.0
        return self.msg_encoder(agg)

    def generate_message(
        self,
        hidden: torch.Tensor,
        msg_context: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a Meow message (discrete symbols).

        Returns:
            symbols: (max_symbols_per_msg,) symbol indices
            log_prob: scalar log probability of the message
        """
        combined = torch.cat([hidden, msg_context], dim=-1)
        logits = self.message_head(combined)
        logits = logits.view(-1, self.max_symbols_per_msg, self.num_symbols)

        # Sample symbols (Gumbel-Softmax for differentiability during training)
        logits = logits / temperature
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        symbols = dist.sample()
        log_prob = dist.log_prob(symbols).sum()

        return symbols, log_prob

    def select_action(
        self,
        hidden: torch.Tensor,
        msg_context: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select a task action.

        Returns:
            action: scalar action index
            log_prob: scalar log probability
        """
        combined = torch.cat([hidden, msg_context], dim=-1)
        logits = self.action_head(combined) / temperature
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def forward(
        self,
        obs: torch.Tensor,
        received_messages: List[Message],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: observe, compose message, select action.

        Returns:
            symbols, msg_log_prob, action, action_log_prob
        """
        hidden = self.encode_observation(obs)
        msg_ctx = self.encode_messages(received_messages)
        symbols, msg_lp = self.generate_message(hidden, msg_ctx, temperature)
        action, act_lp = self.select_action(hidden, msg_ctx, temperature)
        return symbols, msg_lp, action, act_lp


class MeowChannel:
    """
    Communication channel with budget constraints and optional noise.

    Tracks per-agent symbol budgets and applies message drops/noise.
    """

    def __init__(self, config: ChannelConfig, num_agents: int):
        self.config = config
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.budget_remaining = [self.config.budget_per_agent] * self.num_agents
        self.message_log: List[Message] = []
        self.current_round = 0

    def send(self, sender: int, symbols: torch.Tensor, log_prob: torch.Tensor) -> Optional[Message]:
        """
        Send a message through the channel.

        Returns Message if sent successfully, None if dropped or over budget.
        """
        num_symbols = len(symbols)

        # Budget check
        if self.budget_remaining[sender] < num_symbols:
            return None

        self.budget_remaining[sender] -= num_symbols

        # Random drop
        if self.config.drop_rate > 0 and torch.rand(1).item() < self.config.drop_rate:
            return None

        msg = Message(
            sender=sender,
            symbols=symbols.detach(),
            log_prob=log_prob,
            round=self.current_round,
        )
        self.message_log.append(msg)
        return msg

    def receive(self, receiver: int) -> List[Message]:
        """Get messages visible to a receiver (all messages not from self in current round)."""
        return [
            m for m in self.message_log
            if m.sender != receiver and m.round == self.current_round
        ]

    def advance_round(self):
        self.current_round += 1

    def total_symbols_sent(self) -> int:
        return sum(len(m.symbols) for m in self.message_log)

    @property
    def is_budget_exhausted(self) -> bool:
        return all(b <= 0 for b in self.budget_remaining)


class TaskEnvironment(ABC):
    """
    Base class for multi-agent tasks.

    Subclasses implement the specific task logic:
    - How observations are distributed to agents
    - How actions are evaluated
    - What constitutes task success
    """

    @abstractmethod
    def reset(self, batch_size: int = 1) -> List[torch.Tensor]:
        """
        Reset the environment and return initial observations for each agent.

        Returns:
            List of observation tensors, one per agent
        """
        ...

    @abstractmethod
    def evaluate(self, actions: List[torch.Tensor]) -> Tuple[float, Dict]:
        """
        Evaluate agent actions and return task success score.

        Args:
            actions: List of action tensors, one per agent

        Returns:
            (success_score, info_dict)
        """
        ...

    @property
    @abstractmethod
    def num_agents(self) -> int:
        ...

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...


class TaskRunner:
    """
    Orchestrates multi-agent task episodes.

    Runs the communication loop:
    1. Environment provides observations
    2. Agents exchange messages through the channel
    3. After communication rounds, agents take actions
    4. Environment evaluates and returns reward
    """

    def __init__(
        self,
        agents: List[MeowAgent],
        environment: TaskEnvironment,
        channel_config: Optional[ChannelConfig] = None,
        comm_cost_weight: float = 0.1,
    ):
        self.agents = agents
        self.env = environment
        self.channel_config = channel_config or ChannelConfig()
        self.comm_cost_weight = comm_cost_weight

        assert len(agents) == environment.num_agents

    def run_episode(self, temperature: float = 1.0) -> EpisodeResult:
        """Run one complete task episode."""
        channel = MeowChannel(self.channel_config, len(self.agents))
        observations = self.env.reset()

        all_log_probs: List[torch.Tensor] = []

        # Communication rounds
        for round_idx in range(self.channel_config.max_rounds):
            if channel.is_budget_exhausted:
                break

            for agent_idx, agent in enumerate(self.agents):
                received = channel.receive(agent_idx)
                symbols, msg_lp, _, _ = agent(
                    observations[agent_idx].unsqueeze(0),
                    received,
                    temperature=temperature,
                )
                msg = channel.send(agent_idx, symbols, msg_lp)
                if msg is not None:
                    all_log_probs.append(msg_lp)

            channel.advance_round()

        # Action phase: each agent selects final action
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            received = [m for m in channel.message_log if m.sender != agent_idx]
            hidden = agent.encode_observation(observations[agent_idx].unsqueeze(0))
            msg_ctx = agent.encode_messages(received)
            action, act_lp = agent.select_action(hidden, msg_ctx, temperature)
            actions.append(action)
            all_log_probs.append(act_lp)

        # Evaluate
        task_success, info = self.env.evaluate(actions)

        # Communication cost
        total_symbols = channel.total_symbols_sent()
        comm_cost = total_symbols * self.comm_cost_weight

        total_reward = task_success - comm_cost

        return EpisodeResult(
            task_success=task_success,
            communication_cost=comm_cost,
            total_reward=total_reward,
            num_rounds=channel.current_round,
            messages=channel.message_log,
            info={
                **info,
                "total_symbols": total_symbols,
                "log_probs": all_log_probs,
                "budget_remaining": channel.budget_remaining,
            },
        )

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        num_episodes: int = 16,
        temperature: float = 1.0,
        baseline: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        REINFORCE training step over multiple episodes.

        Returns average metrics.
        """
        for agent in self.agents:
            agent.train()

        total_reward = 0.0
        total_success = 0.0
        total_comm = 0.0
        total_symbols = 0

        optimizer.zero_grad()
        policy_loss = torch.tensor(0.0)

        for _ in range(num_episodes):
            result = self.run_episode(temperature=temperature)
            total_reward += result.total_reward
            total_success += result.task_success
            total_comm += result.communication_cost
            total_symbols += result.info["total_symbols"]

            # REINFORCE: loss = -reward * sum(log_probs)
            reward = result.total_reward
            if baseline is not None:
                reward = reward - baseline
            log_probs = result.info["log_probs"]
            if log_probs:
                episode_log_prob = torch.stack(log_probs).sum()
                policy_loss = policy_loss - reward * episode_log_prob

        policy_loss = policy_loss / num_episodes
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for a in self.agents for p in a.parameters()], max_norm=1.0
        )
        optimizer.step()

        return {
            "reward": total_reward / num_episodes,
            "task_success": total_success / num_episodes,
            "comm_cost": total_comm / num_episodes,
            "avg_symbols": total_symbols / num_episodes,
            "policy_loss": policy_loss.item(),
        }
