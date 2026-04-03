"""Tests for analysis module."""

import json
import pytest
from pathlib import Path

from meow.analysis import (
    symbol_frequency,
    learning_curve,
    communication_efficiency,
    generate_report,
)


@pytest.fixture
def sample_experiment():
    return {
        "task": "coding",
        "config": {"task": "coding", "action_dim": 10, "num_symbols": 32},
        "final_eval": {
            "task_success": 0.35,
            "avg_symbols": 18.0,
            "symbol_analysis": {"total": 1800, "unique": 25},
        },
        "training_history": [
            {"task_success": 0.10, "reward": -0.80, "avg_symbols": 18} for _ in range(50)
        ] + [
            {"task_success": 0.35, "reward": -0.55, "avg_symbols": 18} for _ in range(50)
        ],
    }


class TestSymbolFrequency:
    def test_basic(self, sample_experiment):
        result = symbol_frequency(sample_experiment)
        assert result["total_symbols_sent"] == 1800
        assert result["unique_symbols_used"] == 25
        assert result["pattern_emerged"] is True  # 25 < 32

    def test_empty(self):
        result = symbol_frequency({"final_eval": {"symbol_analysis": {}}})
        assert result["total"] == 0


class TestLearningCurve:
    def test_improvement(self, sample_experiment):
        lc = learning_curve(sample_experiment)
        assert lc["final_success"] > lc["initial_success"]
        assert lc["improvement"] > 0
        assert lc["epochs"] == 100


class TestCommunicationEfficiency:
    def test_above_random(self, sample_experiment):
        eff = communication_efficiency(sample_experiment)
        assert eff["improvement_over_random"] > 1.0
        assert eff["success_per_symbol"] > 0


class TestReport:
    def test_generate(self, sample_experiment, tmp_path):
        path = tmp_path / "experiment_coding.json"
        with open(path, "w") as f:
            json.dump(sample_experiment, f)

        report = generate_report(str(tmp_path))
        assert "MEOW EMERGENCE ANALYSIS REPORT" in report
        assert "coding" in report.lower()
