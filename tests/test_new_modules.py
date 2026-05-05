"""
Tests for datagen, formatting, benchmark, and viz modules.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from aquavect.datagen import (
    DatagenConfig,
    SimScenario,
    generate_scenarios,
    execute_scenario,
    generate_dataset,
    save_results,
)
from aquavect.formatting import (
    format_qa_pair,
    format_agent_scenario,
    format_training_data,
    save_training_data,
    dataset_statistics,
)
from aquavect.benchmark import (
    BenchmarkSuite,
    BenchmarkQuestion,
    BenchmarkTier,
    QuestionType,
)
from aquavect.simulation import SimulationConfig


# ──────────────────────────────────────────────
# Datagen tests
# ──────────────────────────────────────────────

class TestDatagen:
    def test_generate_scenarios_count(self):
        cfg = DatagenConfig(n_examples=100)
        scenarios = generate_scenarios(cfg)
        assert len(scenarios) == 100

    def test_scenarios_have_all_strategies(self):
        cfg = DatagenConfig(n_examples=100)
        scenarios = generate_scenarios(cfg)
        strategies = set(s.sampling_strategy for s in scenarios)
        assert "systematic" in strategies
        assert "random" in strategies
        assert "targeted" in strategies

    def test_execute_scenario(self):
        scenario = SimScenario(
            topology="star", n_agents=6,
            biased_positions=[],
            bias_type="intransigent",
            condition_name="sys_high_bs1.0_noagg",
            config=SimulationConfig(n_rounds=10),
            seed=42, save_trajectory=True,
            sampling_strategy="systematic",
        )
        result, trajectory, metadata = execute_scenario(scenario)
        assert "final_mean_brier" in result
        assert trajectory is not None
        assert len(trajectory) == 10
        assert metadata["sampling_strategy"] == "systematic"

    def test_generate_dataset_small(self):
        cfg = DatagenConfig(n_examples=10, n_jobs=1)
        results, trajectories = generate_dataset(cfg, verbose=False)
        assert len(results) == 10

    def test_save_results(self):
        cfg = DatagenConfig(n_examples=5, n_jobs=1)
        results, trajectories = generate_dataset(cfg, verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_results(results, trajectories, output_dir=tmpdir)
            assert os.path.exists(paths["results"])
            assert os.path.exists(paths["report"])
            # Verify JSONL is valid
            with open(paths["results"]) as f:
                for line in f:
                    json.loads(line)


# ──────────────────────────────────────────────
# Formatting tests
# ──────────────────────────────────────────────

class TestFormatting:
    def _make_result(self, **overrides):
        base = {
            "topology": "star",
            "n_agents": 10,
            "n_biased": 1,
            "biased_centrality": "high",
            "bias_strength": 1.0,
            "final_mean_brier": 0.35,
            "final_mean_credence": 0.41,
            "n_rounds": 200,
            "enable_aggregator": False,
            "agg_method": "none",
            "agg_weight": 0.0,
            "sampling_strategy": "systematic",
            "seed": 42,
            "scenario_id": 0,
        }
        base.update(overrides)
        return base

    def test_format_qa_pair(self):
        result = self._make_result()
        example = format_qa_pair(result)
        assert example is not None
        assert "instruction" in example
        assert "output" in example
        assert "star" in example["instruction"].lower()
        assert len(example["output"]) > 50

    def test_format_qa_control(self):
        result = self._make_result(n_biased=0, biased_centrality="none",
                                   final_mean_brier=0.05)
        example = format_qa_pair(result)
        assert example is not None
        assert "truth-seeking" in example["instruction"].lower()

    def test_format_qa_aggregator(self):
        result = self._make_result(enable_aggregator=True,
                                   agg_method="median", agg_weight=0.1)
        example = format_qa_pair(result)
        assert "aggregator" in example["instruction"].lower()
        assert "median" in example["instruction"].lower()

    def test_format_agent_scenario(self):
        result = self._make_result()
        example = format_agent_scenario(result)
        assert example is not None
        assert "you are" in example["instruction"].lower()
        assert example["input"] != ""

    def test_format_agent_scenario_needs_bias(self):
        result = self._make_result(n_biased=0)
        example = format_agent_scenario(result)
        assert example is None

    def test_format_training_data(self):
        results = [self._make_result(seed=i) for i in range(20)]
        examples = format_training_data(results)
        assert len(examples) == 20
        types = set(ex["metadata"]["example_type"] for ex in examples)
        assert "qa" in types

    def test_save_training_data(self):
        results = [self._make_result(seed=i) for i in range(5)]
        examples = format_training_data(results)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_training_data(examples, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                parsed = json.loads(line)
                assert "instruction" in parsed
                assert "output" in parsed
        finally:
            os.unlink(path)

    def test_dataset_statistics(self):
        results = [self._make_result(seed=i) for i in range(10)]
        examples = format_training_data(results)
        stats = dataset_statistics(examples)
        assert stats["total_examples"] == 10
        assert stats["avg_output_length"] > 0
        assert stats["avg_total_tokens_estimate"] > 0


# ──────────────────────────────────────────────
# Benchmark tests
# ──────────────────────────────────────────────

class TestBenchmark:
    def test_generate_suite(self):
        suite = BenchmarkSuite.generate(n_questions=10, seed=42)
        assert len(suite) > 0

    def test_tier_counts(self):
        suite = BenchmarkSuite.generate()
        counts = suite.tier_counts
        assert any(v > 0 for v in counts.values())

    def test_evaluate_multiple_choice(self):
        q = BenchmarkQuestion(
            question_id="test_1",
            tier=BenchmarkTier.NETWORK_LITERACY,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question="Test?",
            correct_answer="a",
            choices=["a", "b", "c"],
        )
        suite = BenchmarkSuite([q])
        result = suite.evaluate_answer(q, "a")
        assert result.is_correct
        assert result.score == 1.0

        result_wrong = suite.evaluate_answer(q, "b")
        assert not result_wrong.is_correct

    def test_evaluate_numeric(self):
        q = BenchmarkQuestion(
            question_id="test_2",
            tier=BenchmarkTier.DYNAMIC_PREDICTION,
            question_type=QuestionType.NUMERIC,
            question="What is the Brier score?",
            correct_answer="0.35",
            tolerance=0.05,
        )
        suite = BenchmarkSuite([q])
        result = suite.evaluate_answer(q, "0.33")
        assert result.is_correct
        assert result.score > 0.5

    def test_evaluate_boolean(self):
        q = BenchmarkQuestion(
            question_id="test_3",
            tier=BenchmarkTier.STRATEGIC_REASONING,
            question_type=QuestionType.BOOLEAN,
            question="Does the aggregator help?",
            correct_answer="no",
        )
        suite = BenchmarkSuite([q])
        assert suite.evaluate_answer(q, "No").is_correct
        assert not suite.evaluate_answer(q, "Yes").is_correct

    def test_compute_scores(self):
        suite = BenchmarkSuite.generate()
        # Simulate a model that always answers correctly
        results = suite.evaluate(
            model_fn=lambda q: suite.questions[0].correct_answer,
            verbose=False,
        )
        scores = suite.compute_scores(results)
        assert "overall_accuracy" in scores

    def test_save_load(self):
        suite = BenchmarkSuite.generate()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            suite.save(path)
            loaded = BenchmarkSuite.load(path)
            assert len(loaded) == len(suite)
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────
# Viz tests (light — just check they don't crash)
# ──────────────────────────────────────────────

class TestViz:
    def _make_results(self):
        return [
            {"topology": "star", "n_agents": 10, "n_biased": 1,
             "biased_centrality": "high", "final_mean_brier": 0.40,
             "enable_aggregator": False},
            {"topology": "star", "n_agents": 10, "n_biased": 1,
             "biased_centrality": "low", "final_mean_brier": 0.22,
             "enable_aggregator": False},
            {"topology": "star", "n_agents": 10, "n_biased": 0,
             "biased_centrality": "none", "final_mean_brier": 0.05,
             "enable_aggregator": False},
        ]

    def test_plot_position_effect(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from aquavect.viz import plot_position_effect
            fig = plot_position_effect(self._make_results())
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_topology_comparison(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from aquavect.viz import plot_topology_comparison
            results = self._make_results()
            results[0]["topology"] = "star"
            results[1]["topology"] = "wheel"
            # Add matching pairs
            results.append({"topology": "star", "n_agents": 10, "n_biased": 1,
                          "biased_centrality": "low", "final_mean_brier": 0.20})
            results.append({"topology": "wheel", "n_agents": 10, "n_biased": 1,
                          "biased_centrality": "high", "final_mean_brier": 0.35})
            fig = plot_topology_comparison(results)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_set_style_doesnt_crash(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from aquavect.viz import set_aquavect_style
            set_aquavect_style()
        except ImportError:
            pytest.skip("matplotlib not available")
