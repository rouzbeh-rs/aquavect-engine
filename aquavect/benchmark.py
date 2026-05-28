"""
Evaluation benchmark for network-structured decision reasoning.

Tests whether language models can reason about epistemic dynamics
in networks across three tiers:

  Tier 1 - Network Literacy:
    Structural questions about graphs (identify hubs, predict paths,
    determine connectivity, compute clustering).

  Tier 2 - Dynamic Prediction:
    Given agents in a network, predict outcomes after N rounds
    (convergence, cascade formation, market price stabilization).

  Tier 3 - Strategic Reasoning:
    Given an agent's position and local information, recommend
    an action (trust reports, consult aggregator, cooperate/defect).

Each question has a known correct answer derived from simulation
results, providing rigorous ground truth.

This module provides:
  - Question generation from simulation data
  - Model evaluation harness
  - Scoring and leaderboard computation
  - Tier classification and difficulty grading

Usage (Phase 3 — scaffold for now):
    >>> from aquavect.benchmark import BenchmarkSuite
    >>> suite = BenchmarkSuite.generate(n_questions=500, seed=42)
    >>> results = suite.evaluate(model_fn)
    >>> suite.print_leaderboard(results)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import json
import os

import numpy as np


class BenchmarkTier(Enum):
    """Difficulty tier for benchmark questions."""
    NETWORK_LITERACY = "tier1_network_literacy"
    DYNAMIC_PREDICTION = "tier2_dynamic_prediction"
    STRATEGIC_REASONING = "tier3_strategic_reasoning"


class QuestionType(Enum):
    """Answer format expected."""
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    FREE_TEXT = "free_text"
    BOOLEAN = "boolean"


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with known correct answer."""
    question_id: str
    tier: BenchmarkTier
    question_type: QuestionType
    question: str
    correct_answer: str
    choices: Optional[List[str]] = None  # for multiple choice
    tolerance: float = 0.0  # for numeric answers
    difficulty: str = "medium"  # easy, medium, hard
    source: str = ""  # which paper/finding this derives from
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "tier": self.tier.value,
            "question_type": self.question_type.value,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "choices": self.choices,
            "tolerance": self.tolerance,
            "difficulty": self.difficulty,
            "source": self.source,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkQuestion":
        return cls(
            question_id=d["question_id"],
            tier=BenchmarkTier(d["tier"]),
            question_type=QuestionType(d["question_type"]),
            question=d["question"],
            correct_answer=d["correct_answer"],
            choices=d.get("choices"),
            tolerance=d.get("tolerance", 0.0),
            difficulty=d.get("difficulty", "medium"),
            source=d.get("source", ""),
            tags=d.get("tags", []),
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a model on one question."""
    question_id: str
    tier: BenchmarkTier
    model_answer: str
    correct_answer: str
    is_correct: bool
    score: float  # 0.0 to 1.0


class BenchmarkSuite:
    """
    A collection of benchmark questions with evaluation capabilities.

    This is the scaffold for Phase 3. The question bank will be
    populated during Phase 3 development.
    """

    def __init__(self, questions: Optional[List[BenchmarkQuestion]] = None):
        self.questions = questions or []

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def tier_counts(self) -> Dict[str, int]:
        counts = {}
        for q in self.questions:
            tier = q.tier.value
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def add_question(self, question: BenchmarkQuestion) -> None:
        self.questions.append(question)

    def get_tier(self, tier: BenchmarkTier) -> List[BenchmarkQuestion]:
        return [q for q in self.questions if q.tier == tier]

    def evaluate_answer(
        self, question: BenchmarkQuestion, model_answer: str
    ) -> EvaluationResult:
        """
        Score a model's answer against the correct answer.

        Scoring depends on question type:
          - multiple_choice / boolean: exact match (case-insensitive)
          - numeric: within tolerance
          - free_text: placeholder (returns 0.0, needs human eval or LLM judge)
        """
        is_correct = False
        score = 0.0
        model_clean = model_answer.strip().lower()
        correct_clean = question.correct_answer.strip().lower()

        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            is_correct = model_clean == correct_clean
            score = 1.0 if is_correct else 0.0

        elif question.question_type == QuestionType.BOOLEAN:
            is_correct = model_clean == correct_clean
            score = 1.0 if is_correct else 0.0

        elif question.question_type == QuestionType.NUMERIC:
            try:
                model_val = float(model_clean)
                correct_val = float(correct_clean)
                is_correct = abs(model_val - correct_val) <= question.tolerance
                # Partial credit based on distance
                if question.tolerance > 0:
                    error = abs(model_val - correct_val) / question.tolerance
                    score = max(0.0, 1.0 - error)
                else:
                    score = 1.0 if is_correct else 0.0
            except ValueError:
                score = 0.0

        elif question.question_type == QuestionType.FREE_TEXT:
            # Phase 3 will implement LLM-as-judge or keyword matching
            score = 0.0

        return EvaluationResult(
            question_id=question.question_id,
            tier=question.tier,
            model_answer=model_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            score=score,
        )

    def evaluate(
        self,
        model_fn: Callable[[str], str],
        subset: Optional[BenchmarkTier] = None,
        verbose: bool = True,
    ) -> List[EvaluationResult]:
        """
        Evaluate a model on the benchmark.

        Parameters
        ----------
        model_fn : callable
            Function that takes a question string and returns an answer string.
        subset : BenchmarkTier, optional
            Only evaluate questions from this tier.
        verbose : bool
            Print progress.

        Returns
        -------
        list of EvaluationResult
        """
        questions = self.get_tier(subset) if subset else self.questions
        results = []

        for i, q in enumerate(questions):
            answer = model_fn(q.question)
            result = self.evaluate_answer(q, answer)
            results.append(result)

            if verbose and (i + 1) % 50 == 0:
                correct_so_far = sum(r.is_correct for r in results)
                print(f"  {i+1}/{len(questions)} — "
                      f"accuracy: {correct_so_far/(i+1):.1%}")

        return results

    def compute_scores(self, results: List[EvaluationResult]) -> Dict:
        """Compute aggregate scores from evaluation results."""
        if not results:
            return {"overall_accuracy": 0.0, "overall_score": 0.0}

        scores = {
            "overall_accuracy": np.mean([r.is_correct for r in results]),
            "overall_score": np.mean([r.score for r in results]),
            "n_questions": len(results),
            "tier_scores": {},
        }

        for tier in BenchmarkTier:
            tier_results = [r for r in results if r.tier == tier]
            if tier_results:
                scores["tier_scores"][tier.value] = {
                    "accuracy": float(np.mean([r.is_correct for r in tier_results])),
                    "score": float(np.mean([r.score for r in tier_results])),
                    "n_questions": len(tier_results),
                }

        return scores

    def print_leaderboard(self, *model_results: Tuple[str, List[EvaluationResult]]) -> None:
        """Print a formatted leaderboard comparing multiple models."""
        print("\n" + "=" * 70)
        print("AQUAVECT BENCHMARK LEADERBOARD")
        print("=" * 70)

        header = f"{'Model':<30} {'Overall':>8} {'Tier 1':>8} {'Tier 2':>8} {'Tier 3':>8}"
        print(header)
        print("-" * 70)

        for model_name, results in model_results:
            scores = self.compute_scores(results)
            tier_scores = scores.get("tier_scores", {})

            t1 = tier_scores.get(BenchmarkTier.NETWORK_LITERACY.value, {}).get("accuracy", 0)
            t2 = tier_scores.get(BenchmarkTier.DYNAMIC_PREDICTION.value, {}).get("accuracy", 0)
            t3 = tier_scores.get(BenchmarkTier.STRATEGIC_REASONING.value, {}).get("accuracy", 0)

            print(f"{model_name:<30} {scores['overall_accuracy']:>7.1%} "
                  f"{t1:>7.1%} {t2:>7.1%} {t3:>7.1%}")

        print("=" * 70)

    def save(self, path: str) -> None:
        """Save benchmark suite to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "version": "0.1.0",
            "n_questions": len(self.questions),
            "tier_counts": self.tier_counts,
            "questions": [q.to_dict() for q in self.questions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkSuite":
        """Load benchmark suite from JSON."""
        with open(path) as f:
            data = json.load(f)
        questions = [BenchmarkQuestion.from_dict(q) for q in data["questions"]]
        return cls(questions=questions)

    @classmethod
    def generate(
        cls,
        n_questions: int = 500,
        seed: int = 42,
    ) -> "BenchmarkSuite":
        """
        Generate a benchmark suite from simulation data.

        This is a Phase 3 scaffold. Currently generates a small set
        of example questions demonstrating the format. The full
        question bank will be built during Phase 3 development.
        """
        suite = cls()

        # Example Tier 1 questions (network literacy)
        suite.add_question(BenchmarkQuestion(
            question_id="t1_001",
            tier=BenchmarkTier.NETWORK_LITERACY,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question=(
                "In a star network with 10 nodes, how many edges does "
                "the central hub have?"
            ),
            correct_answer="9",
            choices=["4", "5", "9", "10"],
            difficulty="easy",
            source="network_theory",
            tags=["star", "degree"],
        ))

        suite.add_question(BenchmarkQuestion(
            question_id="t1_002",
            tier=BenchmarkTier.NETWORK_LITERACY,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question=(
                "Which topology has the property that all nodes have "
                "identical degree centrality?"
            ),
            correct_answer="complete",
            choices=["star", "scale_free", "complete", "hierarchical"],
            difficulty="easy",
            source="network_theory",
            tags=["centrality", "symmetry"],
        ))

        # Example Tier 2 questions (dynamic prediction)
        suite.add_question(BenchmarkQuestion(
            question_id="t2_001",
            tier=BenchmarkTier.DYNAMIC_PREDICTION,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question=(
                "In a star network, a biased agent is placed at the hub "
                "and fabricates evidence with full intensity. Another "
                "identical simulation places the biased agent at a "
                "peripheral node. Which placement causes more epistemic "
                "damage to the community?"
            ),
            correct_answer="hub",
            choices=["hub", "peripheral", "equal damage", "depends on network size"],
            difficulty="medium",
            source="paper1_position_effect",
            tags=["position_effect", "star", "centrality"],
        ))

        suite.add_question(BenchmarkQuestion(
            question_id="t2_002",
            tier=BenchmarkTier.DYNAMIC_PREDICTION,
            question_type=QuestionType.BOOLEAN,
            question=(
                "Does adding a mean-aggregating market node to a star "
                "network eliminate the position effect (the advantage "
                "a central biased agent has over a peripheral one)?"
            ),
            correct_answer="no",
            difficulty="medium",
            source="paper2_aggregator_position",
            tags=["aggregator", "position_effect"],
        ))

        # Example Tier 3 questions (strategic reasoning)
        suite.add_question(BenchmarkQuestion(
            question_id="t3_001",
            tier=BenchmarkTier.STRATEGIC_REASONING,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question=(
                "You are a researcher in a 15-node scale-free network. "
                "One of your neighbors consistently reports 98-100% "
                "success rates for Treatment B across 50 rounds, while "
                "your other 3 neighbors report 50-60% success for "
                "Treatment A. What should you conclude about the "
                "neighbor reporting extreme results?"
            ),
            correct_answer="likely_fabricating",
            choices=[
                "likely_fabricating",
                "probably_correct",
                "need_more_data",
                "both_treatments_equal"
            ],
            difficulty="medium",
            source="paper1_bias_detection",
            tags=["bias_detection", "evidence_evaluation"],
        ))

        return suite
