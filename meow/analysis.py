"""
Meow Symbol Usage Analysis

Analyze communication patterns that emerge from multi-agent training:
- Symbol frequency distribution and entropy
- Per-agent symbol specialization
- Cross-task symbol overlap
- Communication efficiency metrics
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_experiment(path: str) -> Dict:
    """Load experiment results JSON."""
    with open(path) as f:
        return json.load(f)


def symbol_frequency(experiment: Dict) -> Dict:
    """
    Analyze symbol frequency distribution from final evaluation.

    Returns distribution stats and whether a non-uniform pattern emerged.
    """
    analysis = experiment.get("final_eval", {}).get("symbol_analysis", {})
    total = analysis.get("total", 0)
    unique = analysis.get("unique", 0)

    if total == 0:
        return {"total": 0, "unique": 0, "entropy": 0, "pattern_emerged": False}

    # Reconstruct from training history: count symbol usage across late epochs
    # (experiment JSON doesn't store per-symbol counts, so we estimate from unique/total)
    max_entropy = math.log(unique) if unique > 1 else 0

    return {
        "total_symbols_sent": total,
        "unique_symbols_used": unique,
        "max_possible_entropy": round(max_entropy, 3),
        "pattern_emerged": unique < experiment["config"].get("num_symbols", 32),
    }


def learning_curve(experiment: Dict) -> Dict:
    """
    Extract learning curve metrics from training history.

    Returns key trajectory points and improvement statistics.
    """
    history = experiment.get("training_history", [])
    if not history:
        return {}

    successes = [h["task_success"] for h in history]
    rewards = [h["reward"] for h in history]

    # Smoothed curves (window of 20)
    window = min(20, len(successes))
    smoothed_success = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        smoothed_success.append(sum(successes[start:i+1]) / (i - start + 1))

    first_10pct = successes[:max(1, len(successes) // 10)]
    last_10pct = successes[-max(1, len(successes) // 10):]

    return {
        "epochs": len(history),
        "initial_success": round(np.mean(first_10pct), 4),
        "final_success": round(np.mean(last_10pct), 4),
        "peak_success": round(max(smoothed_success), 4),
        "improvement": round(np.mean(last_10pct) - np.mean(first_10pct), 4),
        "initial_reward": round(np.mean([h["reward"] for h in history[:max(1, len(history)//10)]]), 4),
        "final_reward": round(np.mean([h["reward"] for h in history[-max(1, len(history)//10):]]), 4),
    }


def communication_efficiency(experiment: Dict) -> Dict:
    """
    Analyze communication efficiency.

    Compares symbols used vs. task success achieved.
    """
    final = experiment.get("final_eval", {})
    success = final.get("task_success", 0)
    avg_symbols = final.get("avg_symbols", 0)
    config = experiment.get("config", {})
    action_dim = config.get("action_dim", 10)

    # Random baseline: 1/action_dim for single agent, lower for multi-agent agreement
    task = config.get("task", "")
    if task == "coding":
        random_baseline = 1 / action_dim  # both must agree on correct
    elif task == "logic":
        random_baseline = (1 / action_dim) ** 2  # 3 agents, majority
    elif task == "hypothesis":
        random_baseline = 1 / action_dim
    else:
        random_baseline = 1 / action_dim

    improvement_over_random = success / random_baseline if random_baseline > 0 else 0

    # Efficiency: success per symbol
    efficiency = success / avg_symbols if avg_symbols > 0 else 0

    return {
        "task_success": round(success, 4),
        "avg_symbols_per_episode": avg_symbols,
        "random_baseline": round(random_baseline, 4),
        "improvement_over_random": round(improvement_over_random, 2),
        "success_per_symbol": round(efficiency, 4),
    }


def cross_task_comparison(experiment_paths: List[str]) -> Dict:
    """Compare results across multiple task experiments."""
    results = {}
    for path in experiment_paths:
        exp = load_experiment(path)
        task = exp.get("config", {}).get("task", Path(path).stem)
        results[task] = {
            "learning": learning_curve(exp),
            "efficiency": communication_efficiency(exp),
            "symbols": symbol_frequency(exp),
        }
    return results


def generate_report(experiment_dir: str) -> str:
    """
    Generate a human-readable analysis report from all experiments in a directory.
    """
    exp_dir = Path(experiment_dir)
    exp_files = sorted(exp_dir.glob("experiment_*.json"))

    if not exp_files:
        return "No experiment files found."

    lines = [
        "=" * 70,
        "MEOW EMERGENCE ANALYSIS REPORT",
        "=" * 70,
        "",
    ]

    all_results = {}

    for path in exp_files:
        exp = load_experiment(str(path))
        task = exp.get("config", {}).get("task", path.stem)

        lc = learning_curve(exp)
        eff = communication_efficiency(exp)
        sym = symbol_frequency(exp)

        all_results[task] = {"learning": lc, "efficiency": eff, "symbols": sym}

        lines.extend([
            f"--- Task: {task.upper()} ---",
            f"  Epochs: {lc.get('epochs', '?')}",
            f"  Success: {lc.get('initial_success', 0):.1%} → {lc.get('final_success', 0):.1%} (peak: {lc.get('peak_success', 0):.1%})",
            f"  Improvement: +{lc.get('improvement', 0):.1%}",
            f"  vs Random: {eff.get('improvement_over_random', 0):.1f}× better",
            f"  Symbols/episode: {eff.get('avg_symbols_per_episode', 0)}",
            f"  Unique symbols: {sym.get('unique_symbols_used', 0)}",
            f"  Pattern emerged: {sym.get('pattern_emerged', False)}",
            "",
        ])

    # Summary
    lines.extend([
        "=" * 70,
        "SUMMARY",
        "=" * 70,
    ])

    for task, r in all_results.items():
        imp = r["efficiency"]["improvement_over_random"]
        lines.append(f"  {task}: {imp:.1f}× random baseline")

    avg_imp = np.mean([r["efficiency"]["improvement_over_random"] for r in all_results.values()])
    lines.extend([
        f"",
        f"  Average improvement: {avg_imp:.1f}× random baseline",
        "=" * 70,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    print(generate_report(exp_dir))
