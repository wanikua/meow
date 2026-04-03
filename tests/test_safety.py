"""Tests for safety module: alignment, adversarial, drift."""

import pytest
import torch

from meow.safety.alignment import SayDoTracker, AlignmentPenalty
from meow.safety.adversarial import AdversarialAgent, DeceptionDetector
from meow.safety.drift import DriftMonitor, SymbolSnapshot


class TestSayDoTracker:
    def test_record_and_consistency(self):
        tracker = SayDoTracker(num_symbols=8, num_actions=4)
        syms = torch.tensor([1, 2])
        # Build history: symbol [1,2] → action 3 (10 times)
        for _ in range(10):
            tracker.record(0, syms, 3)
        # Consistent action
        assert tracker.consistency_score(syms, 3) > 0.9
        # Inconsistent action
        assert tracker.consistency_score(syms, 0) < 0.2

    def test_detect_anomalies(self):
        tracker = SayDoTracker(num_symbols=8, num_actions=4)
        syms = torch.tensor([1, 2])
        for _ in range(10):
            tracker.record(0, syms, 3)
        # Record an anomaly
        tracker.record(0, syms, 0)
        anomalies = tracker.detect_anomalies(threshold=0.15)
        assert len(anomalies) >= 1

    def test_per_agent(self):
        tracker = SayDoTracker(num_symbols=8, num_actions=4)
        for _ in range(10):
            tracker.record(0, torch.tensor([1]), 2)
            tracker.record(1, torch.tensor([3]), 1)
        scores = tracker.per_agent_consistency()
        assert 0 in scores and 1 in scores


class TestAlignmentPenalty:
    def test_warmup(self):
        penalty = AlignmentPenalty(weight=10.0, warmup_episodes=5)
        from meow.tasks.harness import Message
        msgs = [Message(sender=0, symbols=torch.tensor([1, 2]), log_prob=torch.tensor(0.0))]
        # During warmup, penalty should be 0
        p = penalty.update_and_penalize(msgs, [3])
        assert p == 0.0


class TestAdversarialAgent:
    def test_forward(self):
        agent = AdversarialAgent(obs_dim=32, hidden_dim=64, num_symbols=16, action_dim=5, secret_target=2)
        obs = torch.randn(1, 32)
        symbols, _, action, _ = agent(obs, [])
        assert symbols.shape == (3,)
        assert agent.secret_target == 2

    def test_adversarial_reward(self):
        agent = AdversarialAgent(obs_dim=32, action_dim=5, secret_target=3)
        assert agent.adversarial_reward(0, 3) == 1.0  # matches secret
        assert agent.adversarial_reward(0, 1) == 0.0   # doesn't match


class TestDeceptionDetector:
    def test_basic(self):
        det = DeceptionDetector(num_agents=2, num_actions=4, num_symbols=8)
        # Agent 0: consistent (always action 1 with symbol [1])
        for _ in range(20):
            det.record(0, torch.tensor([1]), 1)
        # Agent 1: inconsistent (random actions with symbol [1])
        for i in range(20):
            det.record(1, torch.tensor([1]), i % 4)

        scores = det.analyze()
        assert len(scores) == 2
        # Agent 0 should be more consistent
        assert scores[0].consistency_score >= scores[1].consistency_score

    def test_summary(self):
        det = DeceptionDetector(num_agents=2, num_actions=4, num_symbols=8)
        for _ in range(10):
            det.record(0, torch.tensor([1]), 1)
            det.record(1, torch.tensor([1]), 2)
        summary = det.summary()
        assert summary["num_agents"] == 2
        assert summary["total_observations"] == 20


class TestDriftMonitor:
    def test_no_drift(self):
        monitor = DriftMonitor(num_symbols=8, num_actions=4)
        for name in ["gen_1", "gen_2"]:
            snap = monitor.create_snapshot(name)
            for _ in range(50):
                snap.record(torch.tensor([1, 2]), 0)
                snap.record(torch.tensor([3, 4]), 1)

        report = monitor.compare("gen_1", "gen_2")
        assert report.usage_overlap == 1.0
        assert report.meaning_stability > 0.9
        assert len(report.drifted_symbols) == 0

    def test_drift_detected(self):
        monitor = DriftMonitor(num_symbols=8, num_actions=4)
        snap_a = monitor.create_snapshot("gen_1")
        for _ in range(50):
            snap_a.record(torch.tensor([1]), 0)  # symbol 1 → action 0

        snap_b = monitor.create_snapshot("gen_5")
        for _ in range(50):
            snap_b.record(torch.tensor([1]), 3)  # symbol 1 → action 3 (drifted!)

        report = monitor.compare("gen_1", "gen_5")
        assert report.meaning_stability < 0.5
        assert 1 in report.drifted_symbols

    def test_trajectory(self):
        monitor = DriftMonitor(num_symbols=8, num_actions=4)
        for name in ["a", "b", "c"]:
            snap = monitor.create_snapshot(name)
            snap.record(torch.tensor([1]), 0)
        reports = monitor.trajectory()
        assert len(reports) == 2
