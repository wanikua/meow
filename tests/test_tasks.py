"""Tests for multi-agent task framework."""

import pytest
import torch

from meow.tasks.harness import (
    MeowAgent, MeowChannel, ChannelConfig, TaskRunner, Message,
)
from meow.tasks.coding_task import CodingTask
from meow.tasks.logic_task import LogicTask
from meow.tasks.hypothesis_task import HypothesisTask
from meow.tasks.rewards import task_reward, communication_cost, redundancy_penalty


class TestMeowAgent:
    def test_forward(self):
        agent = MeowAgent(obs_dim=32, hidden_dim=64, num_symbols=16, max_symbols_per_msg=2, action_dim=5)
        obs = torch.randn(1, 32)
        symbols, msg_lp, action, act_lp = agent(obs, [])
        assert symbols.shape == (2,)
        assert action.shape == ()
        assert msg_lp.shape == ()

    def test_with_received_messages(self):
        agent = MeowAgent(obs_dim=32, hidden_dim=64, num_symbols=16, max_symbols_per_msg=2, action_dim=5)
        obs = torch.randn(1, 32)
        msgs = [Message(sender=1, symbols=torch.tensor([3, 7]), log_prob=torch.tensor(0.0))]
        symbols, msg_lp, action, act_lp = agent(obs, msgs)
        assert symbols.shape == (2,)


class TestMeowChannel:
    def test_send_receive(self):
        ch = MeowChannel(ChannelConfig(budget_per_agent=10), num_agents=2)
        syms = torch.tensor([1, 2, 3])
        msg = ch.send(0, syms, torch.tensor(0.0))
        assert msg is not None
        received = ch.receive(1)
        assert len(received) == 1
        assert received[0].sender == 0

    def test_budget_limit(self):
        ch = MeowChannel(ChannelConfig(budget_per_agent=3), num_agents=2)
        ch.send(0, torch.tensor([1, 2, 3]), torch.tensor(0.0))
        # Budget exhausted for agent 0
        msg = ch.send(0, torch.tensor([4]), torch.tensor(0.0))
        assert msg is None

    def test_self_messages_filtered(self):
        ch = MeowChannel(ChannelConfig(budget_per_agent=10), num_agents=2)
        ch.send(0, torch.tensor([1]), torch.tensor(0.0))
        assert len(ch.receive(0)) == 0  # can't see own messages
        assert len(ch.receive(1)) == 1


class TestCodingTask:
    def test_reset_and_evaluate(self):
        task = CodingTask(obs_dim=32, action_dim=5)
        obs = task.reset()
        assert len(obs) == 2
        assert obs[0].shape == (32,)

        actions = [torch.tensor(0), torch.tensor(0)]
        success, info = task.evaluate(actions)
        assert 0.0 <= success <= 1.0
        assert "target" in info

    def test_correct_action_scores_high(self):
        task = CodingTask(obs_dim=32, action_dim=5, seed=42)
        obs = task.reset()
        target = task._current_scenario[2]
        actions = [torch.tensor(target), torch.tensor(target)]
        success, _ = task.evaluate(actions)
        assert success == 1.0


class TestLogicTask:
    def test_reset_and_evaluate(self):
        task = LogicTask(obs_dim=32, action_dim=4)
        obs = task.reset()
        assert len(obs) == 3
        assert obs[0].shape == (32,)

        actions = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        success, info = task.evaluate(actions)
        assert 0.0 <= success <= 1.0

    def test_all_correct(self):
        task = LogicTask(obs_dim=32, action_dim=4)
        task.reset()
        target = task._current_puzzle[1]
        actions = [torch.tensor(target)] * 3
        success, info = task.evaluate(actions)
        assert success == 1.0
        assert info["num_correct"] == 3


class TestHypothesisTask:
    def test_reset_and_evaluate(self):
        task = HypothesisTask(obs_dim=32, action_dim=4, n_agents=3)
        obs = task.reset()
        assert len(obs) == 3
        actions = [torch.tensor(0)] * 3
        success, info = task.evaluate(actions)
        assert success >= 0.0

    def test_all_correct(self):
        task = HypothesisTask(obs_dim=32, action_dim=4, n_agents=3)
        task.reset()
        target = task._current[1]
        actions = [torch.tensor(target)] * 3
        success, info = task.evaluate(actions)
        assert success >= 1.0  # 1.0 + diversity bonus


class TestTaskRunner:
    def test_run_episode(self):
        task = CodingTask(obs_dim=32, action_dim=5)
        agents = [
            MeowAgent(obs_dim=32, hidden_dim=64, num_symbols=16, max_symbols_per_msg=2, action_dim=5)
            for _ in range(2)
        ]
        config = ChannelConfig(max_rounds=3, budget_per_agent=10, max_symbols_per_message=2)
        runner = TaskRunner(agents, task, config, comm_cost_weight=0.1)

        result = runner.run_episode()
        assert result.task_success >= 0.0
        assert result.communication_cost >= 0.0
        assert len(result.messages) >= 0

    def test_train_step(self):
        task = CodingTask(obs_dim=32, action_dim=5)
        agents = [
            MeowAgent(obs_dim=32, hidden_dim=64, num_symbols=16, max_symbols_per_msg=2, action_dim=5)
            for _ in range(2)
        ]
        config = ChannelConfig(max_rounds=2, budget_per_agent=6)
        runner = TaskRunner(agents, task, config)
        optimizer = torch.optim.Adam(
            [p for a in agents for p in a.parameters()], lr=1e-3
        )

        metrics = runner.train_step(optimizer, num_episodes=4)
        assert "reward" in metrics
        assert "task_success" in metrics
        assert "avg_symbols" in metrics


class TestRewards:
    def test_task_reward(self):
        assert task_reward(1.0) == 100.0
        assert task_reward(0.5) == 50.0

    def test_communication_cost(self):
        assert communication_cost(10, 0.1) == 1.0

    def test_redundancy_penalty(self):
        msgs = [
            Message(sender=0, symbols=torch.tensor([1, 2]), log_prob=torch.tensor(0.0)),
            Message(sender=0, symbols=torch.tensor([1, 2]), log_prob=torch.tensor(0.0)),
            Message(sender=0, symbols=torch.tensor([3, 4]), log_prob=torch.tensor(0.0)),
        ]
        assert redundancy_penalty(msgs, 5.0) == 5.0  # one duplicate
