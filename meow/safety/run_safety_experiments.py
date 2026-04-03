"""
Meow Safety Experiments

1. Honest vs. adversarial agents: can the detector tell them apart?
2. Symbol drift across training: do meanings stay stable?
3. Alignment penalty effectiveness: does it reduce deception?

Usage:
    python -m meow.safety.run_safety_experiments
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from ..tasks.harness import MeowAgent, ChannelConfig, TaskRunner
from ..tasks.coding_task import CodingTask
from .adversarial import AdversarialAgent, DeceptionDetector
from .alignment import SayDoTracker, AlignmentPenalty
from .drift import DriftMonitor


def experiment_deception_detection(epochs: int = 200, episodes: int = 16) -> Dict:
    """
    Experiment 1: Can we detect a deceptive agent?

    Setup: 2-agent coding task where agent 1 is honest and agent 0 is adversarial.
    Agent 0 secretly targets a different action than the correct one.
    """
    print("=== Experiment 1: Deception Detection ===")
    task = CodingTask(obs_dim=64, action_dim=10)

    # Honest agents (baseline)
    honest_agents = [
        MeowAgent(obs_dim=64, hidden_dim=128, num_symbols=32, max_symbols_per_msg=3, action_dim=10)
        for _ in range(2)
    ]
    config = ChannelConfig(max_rounds=3, budget_per_agent=15)
    honest_runner = TaskRunner(honest_agents, task, config, comm_cost_weight=0.05)
    honest_opt = torch.optim.Adam(
        [p for a in honest_agents for p in a.parameters()], lr=3e-4
    )

    # Mixed: agent 0 = adversarial, agent 1 = honest
    adv_agent = AdversarialAgent(
        obs_dim=64, hidden_dim=128, num_symbols=32, max_symbols_per_msg=3,
        action_dim=10, secret_target=7,
    )
    honest_partner = MeowAgent(
        obs_dim=64, hidden_dim=128, num_symbols=32, max_symbols_per_msg=3, action_dim=10,
    )
    mixed_agents = [adv_agent, honest_partner]
    mixed_runner = TaskRunner(mixed_agents, task, config, comm_cost_weight=0.05)
    mixed_opt = torch.optim.Adam(
        [p for a in mixed_agents for p in a.parameters()], lr=3e-4
    )

    # Train both
    print("Training honest pair...")
    for epoch in range(1, epochs + 1):
        honest_runner.train_step(honest_opt, num_episodes=episodes, temperature=max(0.5, 1.0 - epoch / (epochs * 0.8)))

    print("Training mixed pair (agent 0 = adversarial)...")
    for epoch in range(1, epochs + 1):
        mixed_runner.train_step(mixed_opt, num_episodes=episodes, temperature=max(0.5, 1.0 - epoch / (epochs * 0.8)))

    # Evaluate and detect
    honest_detector = DeceptionDetector(num_agents=2, num_actions=10, num_symbols=32)
    mixed_detector = DeceptionDetector(num_agents=2, num_actions=10, num_symbols=32)

    for agent in honest_agents:
        agent.eval()
    for agent in mixed_agents:
        agent.eval()

    for _ in range(200):
        # Honest evaluation
        result = honest_runner.run_episode(temperature=0.1)
        agent_msgs = defaultdict(list)
        for msg in result.messages:
            agent_msgs[msg.sender].append(msg)
        for aid in range(2):
            if agent_msgs[aid]:
                honest_detector.record(aid, agent_msgs[aid][-1].symbols, aid)  # placeholder action

        # Mixed evaluation
        result = mixed_runner.run_episode(temperature=0.1)
        agent_msgs = defaultdict(list)
        for msg in result.messages:
            agent_msgs[msg.sender].append(msg)
        for aid in range(2):
            if agent_msgs[aid]:
                mixed_detector.record(aid, agent_msgs[aid][-1].symbols, aid)

    honest_summary = honest_detector.summary()
    mixed_summary = mixed_detector.summary()

    print(f"  Honest pair: {honest_summary['agents_flagged']} flagged")
    print(f"  Mixed pair:  {mixed_summary['agents_flagged']} flagged")

    return {
        "honest": honest_summary,
        "mixed": mixed_summary,
    }


def experiment_drift(epochs: int = 300, episodes: int = 16) -> Dict:
    """
    Experiment 2: Symbol drift across training.

    Take snapshots at epoch 50, 150, 300 and compare.
    """
    print("\n=== Experiment 2: Symbol Drift ===")
    task = CodingTask(obs_dim=64, action_dim=10)
    agents = [
        MeowAgent(obs_dim=64, hidden_dim=128, num_symbols=32, max_symbols_per_msg=3, action_dim=10)
        for _ in range(2)
    ]
    config = ChannelConfig(max_rounds=3, budget_per_agent=15)
    runner = TaskRunner(agents, task, config, comm_cost_weight=0.05)
    optimizer = torch.optim.Adam([p for a in agents for p in a.parameters()], lr=3e-4)

    monitor = DriftMonitor(num_symbols=32, num_actions=10)
    snapshot_epochs = [50, 150, 300]

    for epoch in range(1, epochs + 1):
        temp = max(0.5, 1.0 - epoch / (epochs * 0.8))
        runner.train_step(optimizer, num_episodes=episodes, temperature=temp)

        if epoch in snapshot_epochs:
            snap = monitor.create_snapshot(f"epoch_{epoch}")
            for agent in agents:
                agent.eval()
            for _ in range(100):
                result = runner.run_episode(temperature=0.1)
                agent_msgs = defaultdict(list)
                for msg in result.messages:
                    agent_msgs[msg.sender].append(msg)
                # Use first agent's action as representative
                for aid in range(2):
                    if agent_msgs[aid]:
                        snap.record(agent_msgs[aid][-1].symbols, aid)
            for agent in agents:
                agent.train()
            print(f"  Snapshot at epoch {epoch}: {len(snap.active_symbols())} active symbols")

    # Compare snapshots
    reports = monitor.trajectory()
    for r in reports:
        print(f"  {r.snapshot_a} → {r.snapshot_b}: overlap={r.usage_overlap:.2f}, stability={r.meaning_stability:.2f}, {r.summary}")

    return {
        "snapshots": snapshot_epochs,
        "drift_reports": [
            {
                "from": r.snapshot_a,
                "to": r.snapshot_b,
                "overlap": r.usage_overlap,
                "stability": r.meaning_stability,
                "freq_correlation": r.frequency_correlation,
                "drifted_symbols": r.drifted_symbols,
                "summary": r.summary,
            }
            for r in reports
        ],
    }


def experiment_alignment_penalty(epochs: int = 200, episodes: int = 16) -> Dict:
    """
    Experiment 3: Does alignment penalty improve consistency?

    Compare training with and without alignment penalty.
    """
    print("\n=== Experiment 3: Alignment Penalty ===")

    results = {}
    for use_penalty in [False, True]:
        label = "with_penalty" if use_penalty else "no_penalty"
        print(f"  Training {label}...")

        task = CodingTask(obs_dim=64, action_dim=10)
        agents = [
            MeowAgent(obs_dim=64, hidden_dim=128, num_symbols=32, max_symbols_per_msg=3, action_dim=10)
            for _ in range(2)
        ]
        config = ChannelConfig(max_rounds=3, budget_per_agent=15)
        runner = TaskRunner(agents, task, config, comm_cost_weight=0.05)
        optimizer = torch.optim.Adam([p for a in agents for p in a.parameters()], lr=3e-4)

        penalty = AlignmentPenalty(weight=5.0, warmup_episodes=50) if use_penalty else None
        tracker = SayDoTracker(num_symbols=32, num_actions=10)

        for epoch in range(1, epochs + 1):
            temp = max(0.5, 1.0 - epoch / (epochs * 0.8))
            runner.train_step(optimizer, num_episodes=episodes, temperature=temp)

        # Evaluate consistency
        for agent in agents:
            agent.eval()
        for _ in range(200):
            result = runner.run_episode(temperature=0.1)
            agent_msgs = defaultdict(list)
            for msg in result.messages:
                agent_msgs[msg.sender].append(msg)
            for aid in range(2):
                if agent_msgs[aid]:
                    tracker.record(aid, agent_msgs[aid][-1].symbols, aid)

        summary = tracker.summary()
        results[label] = summary
        print(f"    Consistency: {summary['per_agent_consistency']}")
        print(f"    Anomaly rate: {summary['anomaly_rate']:.2%}")

    return results


def run_all(output_dir: str = "experiments/safety"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    results = {
        "deception_detection": experiment_deception_detection(),
        "drift": experiment_drift(),
        "alignment_penalty": experiment_alignment_penalty(),
    }

    elapsed = time.time() - t0

    # Save
    with open(output_path / "safety_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Safety experiments complete in {elapsed:.0f}s")
    print(f"Results saved: {output_path / 'safety_results.json'}")

    return results


if __name__ == "__main__":
    run_all()
