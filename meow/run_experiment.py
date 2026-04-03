"""
Meow Emergence Experiment Runner

Train agents on multi-agent tasks and observe communication pattern emergence.

Usage:
    python -m meow.run_experiment --task coding --epochs 200
    python -m meow.run_experiment --task logic --epochs 200
    python -m meow.run_experiment --task hypothesis --epochs 200
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

from .tasks.harness import MeowAgent, ChannelConfig, TaskRunner
from .tasks.coding_task import CodingTask
from .tasks.logic_task import LogicTask
from .tasks.hypothesis_task import HypothesisTask


def create_task(name: str, obs_dim: int, action_dim: int):
    if name == "coding":
        return CodingTask(obs_dim=obs_dim, action_dim=action_dim)
    elif name == "logic":
        return LogicTask(obs_dim=obs_dim, action_dim=action_dim)
    elif name == "hypothesis":
        return HypothesisTask(obs_dim=obs_dim, action_dim=action_dim, n_agents=5)
    else:
        raise ValueError(f"Unknown task: {name}")


def analyze_symbols(messages) -> Dict:
    """Analyze symbol usage patterns in messages."""
    all_symbols = []
    for msg in messages:
        all_symbols.extend(msg.symbols.tolist())

    if not all_symbols:
        return {"total": 0, "unique": 0}

    counts = Counter(all_symbols)
    return {
        "total": len(all_symbols),
        "unique": len(counts),
        "top_5": counts.most_common(5),
    }


def run_experiment(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task
    task = create_task(args.task, args.obs_dim, args.action_dim)
    print(f"Task: {args.task} ({task.num_agents} agents, obs={args.obs_dim}, actions={args.action_dim})")

    # Agents
    agents = [
        MeowAgent(
            obs_dim=args.obs_dim,
            hidden_dim=args.hidden_dim,
            num_symbols=args.num_symbols,
            max_symbols_per_msg=args.symbols_per_msg,
            action_dim=args.action_dim,
        )
        for _ in range(task.num_agents)
    ]
    total_params = sum(p.numel() for a in agents for p in a.parameters())
    print(f"Agents: {task.num_agents} × {total_params // task.num_agents} params")

    # Channel
    channel_config = ChannelConfig(
        max_symbols_per_message=args.symbols_per_msg,
        max_rounds=args.max_rounds,
        budget_per_agent=args.budget,
    )

    # Runner
    runner = TaskRunner(
        agents, task, channel_config,
        comm_cost_weight=args.comm_cost_weight,
    )

    # Optimizer
    all_params = [p for a in agents for p in a.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # Training
    history: List[Dict] = []
    baseline = 0.0  # running average for variance reduction

    print(f"\nTraining for {args.epochs} epochs ({args.episodes_per_epoch} episodes/epoch)...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Anneal temperature
        temperature = max(0.5, 1.0 - epoch / (args.epochs * 0.8))

        metrics = runner.train_step(
            optimizer,
            num_episodes=args.episodes_per_epoch,
            temperature=temperature,
            baseline=baseline,
        )

        # Update baseline (moving average of reward)
        baseline = 0.9 * baseline + 0.1 * metrics["reward"]

        elapsed = time.time() - t0
        metrics["epoch"] = epoch
        metrics["temperature"] = temperature
        history.append(metrics)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"reward {metrics['reward']:+.2f} | "
                f"success {metrics['task_success']:.2f} | "
                f"symbols {metrics['avg_symbols']:.1f} | "
                f"temp {temperature:.2f} | "
                f"{elapsed:.1f}s"
            )

    # Final evaluation (no exploration)
    print("\n" + "=" * 70)
    print("Final evaluation (100 episodes, greedy)...")
    for agent in agents:
        agent.eval()

    eval_results = []
    all_messages = []
    for _ in range(100):
        result = runner.run_episode(temperature=0.1)
        eval_results.append(result)
        all_messages.extend(result.messages)

    avg_success = sum(r.task_success for r in eval_results) / len(eval_results)
    avg_symbols = sum(r.info["total_symbols"] for r in eval_results) / len(eval_results)
    symbol_analysis = analyze_symbols(all_messages)

    print(f"  Task success rate: {avg_success:.2%}")
    print(f"  Avg symbols/episode: {avg_symbols:.1f}")
    print(f"  Unique symbols used: {symbol_analysis['unique']}")
    if symbol_analysis.get("top_5"):
        print(f"  Top 5 symbols: {symbol_analysis['top_5']}")

    # No-communication baseline: train agents with budget=0
    print("\nNo-communication baseline (100 episodes)...")
    baseline_agents = [
        MeowAgent(
            obs_dim=args.obs_dim, hidden_dim=args.hidden_dim,
            num_symbols=args.num_symbols, max_symbols_per_msg=args.symbols_per_msg,
            action_dim=args.action_dim,
        )
        for _ in range(task.num_agents)
    ]
    no_comm_config = ChannelConfig(max_rounds=0, budget_per_agent=0)
    no_comm_runner = TaskRunner(baseline_agents, task, no_comm_config)
    no_comm_opt = torch.optim.Adam(
        [p for a in baseline_agents for p in a.parameters()], lr=args.lr
    )
    # Quick training
    for _ in range(args.epochs):
        no_comm_runner.train_step(no_comm_opt, num_episodes=8, temperature=0.5)
    for a in baseline_agents:
        a.eval()
    no_comm_results = [no_comm_runner.run_episode(temperature=0.1) for _ in range(100)]
    no_comm_success = sum(r.task_success for r in no_comm_results) / len(no_comm_results)
    print(f"  No-comm success: {no_comm_success:.2%}")
    print(f"  Comm advantage:  {avg_success - no_comm_success:+.2%}")

    # Save results
    results = {
        "task": args.task,
        "config": vars(args),
        "final_eval": {
            "task_success": avg_success,
            "avg_symbols": avg_symbols,
            "symbol_analysis": {
                "total": symbol_analysis["total"],
                "unique": symbol_analysis["unique"],
            },
        },
        "no_comm_baseline": {
            "task_success": no_comm_success,
            "comm_advantage": avg_success - no_comm_success,
        },
        "training_history": history,
    }

    results_path = output_dir / f"experiment_{args.task}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Meow emergence experiment")

    # Task
    parser.add_argument("--task", type=str, default="coding", choices=["coding", "logic", "hypothesis"])
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=10)

    # Agent
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-symbols", type=int, default=32)
    parser.add_argument("--symbols-per-msg", type=int, default=3)

    # Channel
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--comm-cost-weight", type=float, default=0.05)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--episodes-per-epoch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=20)

    # Output
    parser.add_argument("--output-dir", type=str, default="experiments")

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
