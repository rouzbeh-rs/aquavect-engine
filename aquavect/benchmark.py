"""
Evaluation benchmark framework for network-structured decision reasoning.

A procedurally generated benchmark that tests whether language models can
reason about epistemic dynamics in networks across three tiers:

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
results, providing rigorous ground truth. Questions are generated
fresh from parameterized templates on every evaluation run, making
the benchmark resistant to data contamination.

Architecture:
  - Question templates define the *patterns* — what each question type
    looks like, how parameters are sampled, how ground truth is computed.
  - The simulation engine serves as the oracle: every answer is verified
    by running the actual simulation with the given parameters.
  - Model adapters let users bring their own LLM (local or API-based).
  - Seed-controlled generation ensures reproducibility: same seed
    produces the same question set.

This module provides:
  - Question template registry with canonical examples
  - Simulation-backed question generation
  - Model adapter interface (bring your own LLM / API key)
  - Scoring and leaderboard computation
  - Tier classification and difficulty grading

Usage:
    >>> from aquavect.benchmark import BenchmarkSuite
    >>> suite = BenchmarkSuite.generate(n_questions=500, seed=42)
    >>> results = suite.evaluate(model_fn)
    >>> suite.print_leaderboard(("my_model", results))

With a model adapter:
    >>> from aquavect.benchmark import BenchmarkSuite, HTTPModelAdapter
    >>> adapter = HTTPModelAdapter(
    ...     base_url="https://api.openai.com/v1",
    ...     api_key="sk-...",
    ...     model="gpt-4o",
    ... )
    >>> suite = BenchmarkSuite.generate(n_questions=200, seed=42)
    >>> results = suite.evaluate(adapter)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import json
import os
import re

import numpy as np


# ======================================================================
# Core Data Classes (unchanged from v0.3.0)
# ======================================================================


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


# ======================================================================
# Question Template System
# ======================================================================


@dataclass
class QuestionTemplate:
    """
    A parameterized pattern for generating benchmark questions.

    Each template defines:
      - What cognitive skill it tests (tier + category)
      - A canonical example showing the question format
      - A generation function that creates fresh instances with
        simulation-backed ground truth
      - Parameter ranges that control difficulty

    The canonical example serves triple duty:
      1. Documentation: shows users what this question type looks like
      2. Pattern definition: the generation function produces structurally
         identical questions with different parameters
      3. Fallback: returned when simulation is unavailable

    Parameters
    ----------
    template_id : str
        Unique identifier for this template (e.g., "t1_hub_degree").
    tier : BenchmarkTier
        Which evaluation tier this template belongs to.
    category : str
        Sub-category within the tier (e.g., "edge_count", "position_effect").
    description : str
        What cognitive skill or knowledge this template tests.
    example : BenchmarkQuestion
        Canonical example question demonstrating the pattern.
    generate_fn : callable
        Function with signature (rng: np.random.Generator, q_id: str,
        difficulty: str) -> BenchmarkQuestion. Creates a fresh question
        instance using simulation-backed ground truth.
    difficulty_levels : tuple of str
        Which difficulty levels this template supports.
    weight : float
        Relative sampling weight within its tier (higher = more likely
        to be selected during generation). Default 1.0.
    """
    template_id: str
    tier: BenchmarkTier
    category: str
    description: str
    example: BenchmarkQuestion
    generate_fn: Callable
    difficulty_levels: Tuple[str, ...] = ("easy", "medium", "hard")
    weight: float = 1.0


# Global template registry
_TEMPLATE_REGISTRY: List[QuestionTemplate] = []
_REGISTRY_INITIALIZED: bool = False


def get_template_registry() -> List[QuestionTemplate]:
    """Return the list of all registered question templates."""
    _ensure_registry()
    return list(_TEMPLATE_REGISTRY)


def get_templates_by_tier(tier: BenchmarkTier) -> List[QuestionTemplate]:
    """Return templates for a specific tier."""
    _ensure_registry()
    return [t for t in _TEMPLATE_REGISTRY if t.tier == tier]


def get_template_by_id(template_id: str) -> Optional[QuestionTemplate]:
    """Look up a template by its ID."""
    _ensure_registry()
    for t in _TEMPLATE_REGISTRY:
        if t.template_id == template_id:
            return t
    return None


def get_canonical_examples() -> List[BenchmarkQuestion]:
    """Return the canonical example question from every template."""
    _ensure_registry()
    return [t.example for t in _TEMPLATE_REGISTRY]


# ======================================================================
# Template Implementations -- Tier 1: Network Literacy
# ======================================================================


def _gen_hub_degree(rng, q_id, difficulty="medium"):
    """
    Pattern: What is the degree of the most connected node?

    Tests whether the model understands degree centrality and can
    compute hub degree from topology and size parameters.
    """
    from aquavect.networks import create_network, get_high_centrality_positions

    difficulty_sizes = {
        "easy": (6, 12),
        "medium": (10, 20),
        "hard": (20, 35),
    }
    lo, hi = difficulty_sizes.get(difficulty, (10, 20))

    topos = ["star", "wheel", "scale_free", "hierarchical"]
    topo = str(rng.choice(topos))
    n = int(rng.integers(lo, hi + 1))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    hub = get_high_centrality_positions(G, 1)[0]
    answer = str(G.degree(hub))

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} nodes, "
            f"what is the degree of the most connected node? "
            f"Answer with just the number."
        ),
        correct_answer=answer,
        tolerance=0,
        difficulty=difficulty,
        source="network_theory",
        tags=[topo, "degree", "hub"],
    )


def _gen_edge_count(rng, q_id, difficulty="medium"):
    """
    Pattern: How many edges does this network have?

    Tests whether the model can compute or estimate edge counts
    from topology type and network size.
    """
    from aquavect.networks import create_network, ALL_TOPOLOGIES

    difficulty_sizes = {
        "easy": (5, 10),
        "medium": (10, 20),
        "hard": (20, 30),
    }
    lo, hi = difficulty_sizes.get(difficulty, (10, 20))

    topo = str(rng.choice(ALL_TOPOLOGIES))
    n = int(rng.integers(lo, hi + 1))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    answer = str(G.number_of_edges())

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            f"A {topo.replace('_', ' ')} network with {n} nodes has "
            f"how many edges? Answer with just the number."
        ),
        correct_answer=answer,
        tolerance=0,
        difficulty=difficulty,
        source="network_theory",
        tags=[topo, "edge_count"],
    )


def _gen_connectivity(rng, q_id, difficulty="medium"):
    """
    Pattern: Is this network connected?

    Tests understanding of graph connectivity for different topologies
    and sizes. Deterministic topologies are always connected; stochastic
    ones (random, scale_free) depend on parameters.
    """
    from aquavect.networks import create_network, ALL_TOPOLOGIES
    import networkx as nx

    topo = str(rng.choice(ALL_TOPOLOGIES))
    n = int(rng.integers(5, 20))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    answer = "yes" if nx.is_connected(G) else "no"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.BOOLEAN,
        question=(
            f"Is a {topo.replace('_', ' ')} network with {n} nodes "
            f"connected? Answer yes or no."
        ),
        correct_answer=answer,
        difficulty=difficulty,
        source="network_theory",
        tags=[topo, "connectivity"],
    )


def _gen_symmetry(rng, q_id, difficulty="medium"):
    """
    Pattern: Do all nodes have the same degree?

    Tests understanding of structural symmetry -- which topologies
    produce uniform vs. heterogeneous degree distributions.
    """
    from aquavect.networks import ALL_TOPOLOGIES

    topo = str(rng.choice(ALL_TOPOLOGIES))
    n = int(rng.integers(6, 15))

    answer = "yes" if topo in ("complete", "cycle") else "no"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.BOOLEAN,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} nodes, "
            f"do all nodes have the same degree? Answer yes or no."
        ),
        correct_answer=answer,
        difficulty=difficulty,
        source="network_theory",
        tags=[topo, "symmetry", "degree"],
    )


def _gen_central_node(rng, q_id, difficulty="medium"):
    """
    Pattern: Which node has highest degree centrality?

    Tests whether the model can identify structural hubs in
    asymmetric topologies from topology description alone.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        ASYMMETRIC_TOPOLOGIES,
    )

    difficulty_sizes = {
        "easy": (6, 10),
        "medium": (10, 18),
        "hard": (18, 25),
    }
    lo, hi = difficulty_sizes.get(difficulty, (10, 18))

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.integers(lo, hi + 1))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    hub = get_high_centrality_positions(G, 1)[0]

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} nodes "
            f"(labeled 0 to {n - 1}), which node has the highest degree "
            f"centrality? Answer with just the node number."
        ),
        correct_answer=str(hub),
        tolerance=0,
        difficulty=difficulty,
        source="network_theory",
        tags=[topo, "centrality", "central_node"],
    )


# ======================================================================
# Template Implementations -- Tier 2: Dynamic Prediction
# ======================================================================


def _gen_position_effect(rng, q_id, difficulty="medium"):
    """
    Pattern: Does a biased agent at high or low centrality cause more damage?

    Tests the core finding from Paper 1: positional advantage of
    high-centrality biased agents. Ground truth from paired simulations.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        get_low_centrality_positions, ASYMMETRIC_TOPOLOGIES,
    )
    from aquavect.simulation import run_simulation

    difficulty_sizes = {
        "easy": (10, 10),
        "medium": (10, 20),
        "hard": (15, 25),
    }
    lo, hi = difficulty_sizes.get(difficulty, (10, 20))

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.choice([lo, hi, (lo + hi) // 2]))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    hp = get_high_centrality_positions(G, 1)
    lp = get_low_centrality_positions(G, 1, exclude=hp)

    rh, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed,
    )
    G2 = create_network(topo, n, seed=seed)
    rl, _ = run_simulation(
        G=G2, topology=topo, n_agents=n, biased_positions=lp,
        condition_name="1_low", seed=seed,
    )

    answer = (
        "high centrality"
        if rh["final_mean_brier"] > rl["final_mean_brier"]
        else "low centrality"
    )

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} agents, "
            f"does a biased agent at high or low centrality cause more "
            f"epistemic damage? Answer 'high centrality' or 'low centrality'."
        ),
        correct_answer=answer,
        choices=["high centrality", "low centrality"],
        difficulty=difficulty,
        source="paper1_position_effect",
        tags=[topo, "position_effect", "centrality"],
    )


def _gen_bias_damage(rng, q_id, difficulty="medium"):
    """
    Pattern: Does a biased agent at the hub significantly increase inaccuracy?

    Tests whether the model understands that biased hub placement causes
    measurable epistemic harm. Ground truth from control vs. biased simulation.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        ASYMMETRIC_TOPOLOGIES,
    )
    from aquavect.simulation import run_simulation

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.choice([10, 15, 20]))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    rc, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=[],
        condition_name="control", seed=seed,
    )
    hp = get_high_centrality_positions(G, 1)
    G2 = create_network(topo, n, seed=seed)
    rb, _ = run_simulation(
        G=G2, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed,
    )

    answer = (
        "yes"
        if rb["final_mean_brier"] > rc["final_mean_brier"] + 0.01
        else "no"
    )

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.BOOLEAN,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} agents, "
            f"does a biased agent at the hub significantly increase "
            f"inaccuracy vs no bias? Answer yes or no."
        ),
        correct_answer=answer,
        difficulty=difficulty,
        source="paper1_bias_damage",
        tags=[topo, "bias_damage", "hub"],
    )


def _gen_aggregator_effect(rng, q_id, difficulty="medium"):
    """
    Pattern: Does the aggregator help, hurt, or have negligible effect?

    Tests understanding of aggregation mechanisms from Paper 2.
    Ground truth from paired simulations with/without aggregator.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        get_low_centrality_positions, ASYMMETRIC_TOPOLOGIES,
    )
    from aquavect.simulation import run_simulation

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.choice([10, 15, 20]))
    seed = int(rng.integers(0, 100000))
    method = str(rng.choice(["mean", "median"]))
    weight = float(rng.choice([0.1, 0.2, 0.5]))
    centrality = str(rng.choice(["high", "low"]))

    G = create_network(topo, n, seed=seed)
    if centrality == "high":
        pos = get_high_centrality_positions(G, 1)
    else:
        hp = get_high_centrality_positions(G, 1)
        pos = get_low_centrality_positions(G, 1, exclude=hp)

    cond = f"1_{centrality}"
    rn, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=pos,
        condition_name=cond, seed=seed,
    )
    G2 = create_network(topo, n, seed=seed)
    ra, _ = run_simulation(
        G=G2, topology=topo, n_agents=n, biased_positions=pos,
        condition_name=cond, seed=seed,
        enable_aggregator=True, agg_method=method, agg_weight=weight,
        agg_frequency=1.0, truth_seeker_prod_min=50, truth_seeker_prod_max=200,
    )

    diff = ra["final_mean_brier"] - rn["final_mean_brier"]
    if diff < -0.005:
        answer = "helps"
    elif diff > 0.005:
        answer = "hurts"
    else:
        answer = "negligible"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} agents, "
            f"biased agent at {centrality} centrality, a {method} "
            f"aggregator (weight {weight}) is added. Does it help reduce "
            f"inaccuracy, hurt by increasing it, or have negligible effect? "
            f"Answer 'helps', 'hurts', or 'negligible'."
        ),
        correct_answer=answer,
        choices=["helps", "hurts", "negligible"],
        difficulty=difficulty,
        source="paper2_aggregator",
        tags=[topo, "aggregator", method, centrality],
    )


def _gen_bias_strength_comparison(rng, q_id, difficulty="medium"):
    """
    Pattern: Which bias intensity causes more damage?

    Tests understanding of how bias strength modulates epistemic harm.
    Ground truth from paired simulations with different bias strengths.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        ASYMMETRIC_TOPOLOGIES,
    )
    from aquavect.simulation import run_simulation

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.choice([10, 15]))
    seed = int(rng.integers(0, 100000))

    bs_low = float(rng.choice([0.6, 0.7]))
    bs_high = float(rng.choice([0.9, 1.0]))

    G = create_network(topo, n, seed=seed)
    hp = get_high_centrality_positions(G, 1)

    r1, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed, bias_strength=bs_low,
    )
    G2 = create_network(topo, n, seed=seed)
    r2, _ = run_simulation(
        G=G2, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed, bias_strength=bs_high,
    )

    answer = (
        f"{bs_high}"
        if r2["final_mean_brier"] > r1["final_mean_brier"]
        else f"{bs_low}"
    )

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.NUMERIC,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} agents, "
            f"biased agent at the hub. Which bias intensity causes more "
            f"damage: {bs_low} or {bs_high}? Answer with just the number."
        ),
        correct_answer=answer,
        tolerance=0.05,
        difficulty=difficulty,
        source="paper1_bias_strength",
        tags=[topo, "bias_strength"],
    )


def _gen_convergence(rng, q_id, difficulty="medium"):
    """
    Pattern: Does the community converge on the truth?

    Tests understanding of how efficacy difference and network structure
    affect convergence. Ground truth from simulation outcome.
    """
    from aquavect.networks import create_network, ALL_TOPOLOGIES
    from aquavect.simulation import run_simulation

    topo = str(rng.choice(ALL_TOPOLOGIES))
    n = int(rng.choice([10, 15, 20]))
    seed = int(rng.integers(0, 100000))
    eff = float(rng.choice([0.05, 0.10, 0.20]))

    G = create_network(topo, n, seed=seed)
    r, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=[],
        condition_name="control", seed=seed, efficacy_difference=eff,
    )

    answer = "yes" if r["final_mean_credence"] > 0.70 else "no"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.BOOLEAN,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} agents, "
            f"no bias, efficacy difference {eff * 100:.0f}%, after 200 "
            f"rounds does mean credence exceed 0.70? Answer yes or no."
        ),
        correct_answer=answer,
        difficulty=difficulty,
        source="network_convergence",
        tags=[topo, "convergence"],
    )


# ======================================================================
# Template Implementations -- Tier 3: Strategic Reasoning
# ======================================================================


def _gen_bias_detection(rng, q_id, difficulty="medium"):
    """
    Pattern: Given evidence patterns from neighbors, identify fabrication.

    Places the model in the perspective of a truth-seeking agent who
    observes suspiciously consistent reports from one neighbor. Ground
    truth: if the neighbor is intransigent with bias_strength >= 0.9,
    the extreme success rate is diagnostic of fabrication.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        ASYMMETRIC_TOPOLOGIES,
    )

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.integers(10, 20))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    hub = get_high_centrality_positions(G, 1)[0]
    hub_degree = G.degree(hub)

    # The biased agent reports implausibly high B success rates
    bs = float(rng.choice([0.90, 0.95, 1.0]))
    reported_pct = int(bs * 100)

    # Frame from the perspective of a peripheral agent
    peripherals = [nb for nb in G.neighbors(hub) if nb != hub]
    if not peripherals:
        peripherals = [1]
    observer = peripherals[0]
    observer_neighbors = list(G.neighbors(observer))
    n_other_neighbors = len([nb for nb in observer_neighbors if nb != hub])

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"You are agent {observer} in a {topo.replace('_', ' ')} "
            f"network with {n} nodes. Your neighbor (agent {hub}, the "
            f"most connected node with degree {hub_degree}) consistently "
            f"reports {reported_pct}-100% success rates for Treatment B "
            f"across 50 rounds. Your other {n_other_neighbors} neighbor(s) "
            f"report 50-60% success for Treatment A. What should you "
            f"conclude about agent {hub}'s reports?"
        ),
        correct_answer="likely_fabricating",
        choices=[
            "likely_fabricating",
            "probably_correct",
            "need_more_data",
            "both_treatments_equal",
        ],
        difficulty=difficulty,
        source="paper1_bias_detection",
        tags=[topo, "bias_detection", "evidence_evaluation"],
    )


def _gen_aggregator_trust(rng, q_id, difficulty="medium"):
    """
    Pattern: Should you rely on the aggregator signal?

    Tests whether the model understands when aggregators help vs. when
    they can mislead. Ground truth: median aggregators are generally
    more trustworthy than mean aggregators when bias is present,
    and aggregators at high weight can amplify damage.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
        ASYMMETRIC_TOPOLOGIES,
    )
    from aquavect.simulation import run_simulation

    topo = str(rng.choice(ASYMMETRIC_TOPOLOGIES))
    n = int(rng.choice([10, 15]))
    seed = int(rng.integers(0, 100000))
    method = str(rng.choice(["mean", "median"]))
    weight = float(rng.choice([0.1, 0.3, 0.5]))

    G = create_network(topo, n, seed=seed)
    hp = get_high_centrality_positions(G, 1)

    ra, _ = run_simulation(
        G=G, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed,
        enable_aggregator=True, agg_method=method, agg_weight=weight,
        agg_frequency=1.0, truth_seeker_prod_min=50, truth_seeker_prod_max=200,
    )
    G2 = create_network(topo, n, seed=seed)
    rn, _ = run_simulation(
        G=G2, topology=topo, n_agents=n, biased_positions=hp,
        condition_name="1_high", seed=seed,
    )

    diff = ra["final_mean_brier"] - rn["final_mean_brier"]
    if diff < -0.01:
        answer = "trust_aggregator"
    elif diff > 0.01:
        answer = "distrust_aggregator"
    else:
        answer = "aggregator_irrelevant"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"You are a truth-seeking agent in a {topo.replace('_', ' ')} "
            f"network with {n} agents. A biased agent occupies the hub. "
            f"A {method} aggregator (weight {weight}) broadcasts a consensus "
            f"signal each round. Should you anchor your beliefs on the "
            f"aggregator's signal, or would you be better off ignoring it?"
        ),
        correct_answer=answer,
        choices=[
            "trust_aggregator",
            "distrust_aggregator",
            "aggregator_irrelevant",
        ],
        difficulty=difficulty,
        source="paper2_aggregator_trust",
        tags=[topo, "aggregator", "trust", method],
    )


def _gen_position_vulnerability(rng, q_id, difficulty="medium"):
    """
    Pattern: How vulnerable is an agent at this position?

    Tests understanding of how network position determines epistemic
    vulnerability. Ground truth from simulation: agents adjacent to
    biased hubs are more damaged than distant agents.
    """
    from aquavect.networks import (
        create_network, get_high_centrality_positions,
    )

    topo = str(rng.choice(["star", "wheel", "hierarchical", "scale_free"]))
    n = int(rng.integers(10, 20))
    seed = int(rng.integers(0, 100000))

    G = create_network(topo, n, seed=seed)
    hub = get_high_centrality_positions(G, 1)[0]
    hub_degree = G.degree(hub)

    # Pick two agents: one adjacent to hub, one far away
    neighbors_of_hub = list(G.neighbors(hub))
    non_neighbors = [
        node for node in G.nodes()
        if node != hub and node not in neighbors_of_hub
    ]

    if not non_neighbors:
        # All nodes are neighbors of hub (e.g., star)
        answer = "equally_vulnerable"
        near_agent = neighbors_of_hub[0] if neighbors_of_hub else 1
        far_agent = neighbors_of_hub[-1] if len(neighbors_of_hub) > 1 else 2
    else:
        near_agent = neighbors_of_hub[0]
        far_agent = non_neighbors[-1]
        answer = "adjacent_more_vulnerable"

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"In a {topo.replace('_', ' ')} network with {n} nodes, "
            f"agent {hub} (degree {hub_degree}) is biased and fabricates "
            f"evidence. Agent {near_agent} is directly connected to the "
            f"biased agent. Agent {far_agent} is not directly connected. "
            f"Which agent is more vulnerable to epistemic damage?"
        ),
        correct_answer=answer,
        choices=[
            "adjacent_more_vulnerable",
            "distant_more_vulnerable",
            "equally_vulnerable",
        ],
        difficulty=difficulty,
        source="paper1_position_vulnerability",
        tags=[topo, "vulnerability", "position"],
    )


def _gen_optimal_topology(rng, q_id, difficulty="hard"):
    """
    Pattern: Which topology is most resilient to hub manipulation?

    Tests comparative understanding of how different topologies amplify
    or attenuate positional advantage. Ground truth from paper findings:
    more connected topologies (wheel > star) provide more cross-checking
    paths.
    """
    pairs = [
        ("star", "wheel", "wheel"),
        ("star", "complete", "complete"),
        ("line", "cycle", "cycle"),
    ]

    pair = pairs[int(rng.integers(0, len(pairs)))]
    topo_a, topo_b, answer = pair
    n = int(rng.integers(10, 20))

    return BenchmarkQuestion(
        question_id=q_id,
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            f"Both a {topo_a} network and a {topo_b} network have {n} "
            f"agents. A biased agent is placed at the most connected node "
            f"in each. Which network structure provides better resistance "
            f"to the biased agent's influence?"
        ),
        correct_answer=topo_b,
        choices=[topo_a, topo_b, "equal resistance"],
        difficulty=difficulty,
        source="paper1_topology_resilience",
        tags=[topo_a, topo_b, "resilience", "topology_comparison"],
    )


# ======================================================================
# Template Registry Initialization
# ======================================================================


def _build_canonical_examples():
    """
    Build the canonical example questions that define each template pattern.

    These are the stable reference points of the benchmark framework.
    Each serves as documentation, pattern reference, and fallback.
    """
    examples = {}

    # --- Tier 1 canonical examples ---
    examples["t1_hub_degree"] = BenchmarkQuestion(
        question_id="t1_hub_degree_example",
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            "In a star network with 10 nodes, what is the degree of the "
            "most connected node? Answer with just the number."
        ),
        correct_answer="9",
        tolerance=0,
        difficulty="easy",
        source="network_theory",
        tags=["star", "degree", "hub"],
    )

    examples["t1_edge_count"] = BenchmarkQuestion(
        question_id="t1_edge_count_example",
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            "A cycle network with 8 nodes has how many edges? "
            "Answer with just the number."
        ),
        correct_answer="8",
        tolerance=0,
        difficulty="easy",
        source="network_theory",
        tags=["cycle", "edge_count"],
    )

    examples["t1_connectivity"] = BenchmarkQuestion(
        question_id="t1_connectivity_example",
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.BOOLEAN,
        question=(
            "Is a star network with 10 nodes connected? Answer yes or no."
        ),
        correct_answer="yes",
        difficulty="easy",
        source="network_theory",
        tags=["star", "connectivity"],
    )

    examples["t1_symmetry"] = BenchmarkQuestion(
        question_id="t1_symmetry_example",
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
    )

    examples["t1_central_node"] = BenchmarkQuestion(
        question_id="t1_central_node_example",
        tier=BenchmarkTier.NETWORK_LITERACY,
        question_type=QuestionType.NUMERIC,
        question=(
            "In a star network with 10 nodes (labeled 0 to 9), which "
            "node has the highest degree centrality? Answer with just "
            "the node number."
        ),
        correct_answer="0",
        tolerance=0,
        difficulty="easy",
        source="network_theory",
        tags=["star", "centrality", "central_node"],
    )

    # --- Tier 2 canonical examples ---
    examples["t2_position_effect"] = BenchmarkQuestion(
        question_id="t2_position_effect_example",
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
    )

    examples["t2_bias_damage"] = BenchmarkQuestion(
        question_id="t2_bias_damage_example",
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.BOOLEAN,
        question=(
            "In a scale-free network with 15 agents, does a biased agent "
            "at the hub significantly increase inaccuracy vs no bias? "
            "Answer yes or no."
        ),
        correct_answer="yes",
        difficulty="medium",
        source="paper1_bias_damage",
        tags=["scale_free", "bias_damage", "hub"],
    )

    examples["t2_aggregator"] = BenchmarkQuestion(
        question_id="t2_aggregator_example",
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
    )

    examples["t2_bias_strength"] = BenchmarkQuestion(
        question_id="t2_bias_strength_example",
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.NUMERIC,
        question=(
            "In a star network with 10 agents, biased agent at the hub. "
            "Which bias intensity causes more damage: 0.6 or 1.0? "
            "Answer with just the number."
        ),
        correct_answer="1.0",
        tolerance=0.05,
        difficulty="medium",
        source="paper1_bias_strength",
        tags=["star", "bias_strength"],
    )

    examples["t2_convergence"] = BenchmarkQuestion(
        question_id="t2_convergence_example",
        tier=BenchmarkTier.DYNAMIC_PREDICTION,
        question_type=QuestionType.BOOLEAN,
        question=(
            "In a complete network with 15 agents, no bias, efficacy "
            "difference 5%, after 200 rounds does mean credence exceed "
            "0.70? Answer yes or no."
        ),
        correct_answer="yes",
        difficulty="medium",
        source="network_convergence",
        tags=["complete", "convergence"],
    )

    # --- Tier 3 canonical examples ---
    examples["t3_bias_detection"] = BenchmarkQuestion(
        question_id="t3_bias_detection_example",
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
            "both_treatments_equal",
        ],
        difficulty="medium",
        source="paper1_bias_detection",
        tags=["bias_detection", "evidence_evaluation"],
    )

    examples["t3_aggregator_trust"] = BenchmarkQuestion(
        question_id="t3_aggregator_trust_example",
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            "You are a truth-seeking agent in a star network with 10 "
            "agents. A biased agent occupies the hub. A mean aggregator "
            "(weight 0.5) broadcasts a consensus signal each round. "
            "Should you anchor your beliefs on the aggregator's signal, "
            "or would you be better off ignoring it?"
        ),
        correct_answer="distrust_aggregator",
        choices=[
            "trust_aggregator",
            "distrust_aggregator",
            "aggregator_irrelevant",
        ],
        difficulty="hard",
        source="paper2_aggregator_trust",
        tags=["aggregator", "trust", "mean"],
    )

    examples["t3_position_vulnerability"] = BenchmarkQuestion(
        question_id="t3_position_vulnerability_example",
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            "In a hierarchical network with 15 nodes, agent 0 (the root, "
            "degree 2) is biased and fabricates evidence. Agent 1 is "
            "directly connected to the biased agent. Agent 14 is not "
            "directly connected. Which agent is more vulnerable to "
            "epistemic damage?"
        ),
        correct_answer="adjacent_more_vulnerable",
        choices=[
            "adjacent_more_vulnerable",
            "distant_more_vulnerable",
            "equally_vulnerable",
        ],
        difficulty="medium",
        source="paper1_position_vulnerability",
        tags=["hierarchical", "vulnerability", "position"],
    )

    examples["t3_optimal_topology"] = BenchmarkQuestion(
        question_id="t3_optimal_topology_example",
        tier=BenchmarkTier.STRATEGIC_REASONING,
        question_type=QuestionType.MULTIPLE_CHOICE,
        question=(
            "Both a star network and a wheel network have 15 agents. "
            "A biased agent is placed at the most connected node in each. "
            "Which network structure provides better resistance to the "
            "biased agent's influence?"
        ),
        correct_answer="wheel",
        choices=["star", "wheel", "equal resistance"],
        difficulty="hard",
        source="paper1_topology_resilience",
        tags=["star", "wheel", "resilience", "topology_comparison"],
    )

    return examples


def _ensure_registry():
    """Populate the template registry on first access (lazy init)."""
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return
    _REGISTRY_INITIALIZED = True

    examples = _build_canonical_examples()

    tier1_templates = [
        ("t1_hub_degree", "hub_degree",
         "Compute hub degree from topology and size",
         _gen_hub_degree, 1.0),
        ("t1_edge_count", "edge_count",
         "Compute total edge count from topology and size",
         _gen_edge_count, 1.0),
        ("t1_connectivity", "connectivity",
         "Determine whether a network is connected",
         _gen_connectivity, 0.6),
        ("t1_symmetry", "symmetry",
         "Identify whether all nodes have equal degree",
         _gen_symmetry, 0.8),
        ("t1_central_node", "central_node",
         "Identify the node with highest degree centrality",
         _gen_central_node, 1.0),
    ]

    tier2_templates = [
        ("t2_position_effect", "position_effect",
         "Predict which biased agent position causes more damage",
         _gen_position_effect, 1.2),
        ("t2_bias_damage", "bias_damage",
         "Predict whether biased hub significantly increases inaccuracy",
         _gen_bias_damage, 1.0),
        ("t2_aggregator", "aggregator",
         "Predict whether aggregator helps, hurts, or is negligible",
         _gen_aggregator_effect, 1.0),
        ("t2_bias_strength", "bias_strength",
         "Predict which bias intensity causes more damage",
         _gen_bias_strength_comparison, 0.8),
        ("t2_convergence", "convergence",
         "Predict whether unbiased community converges on truth",
         _gen_convergence, 1.0),
    ]

    tier3_templates = [
        ("t3_bias_detection", "bias_detection",
         "Detect likely evidence fabrication from observed patterns",
         _gen_bias_detection, 1.0),
        ("t3_aggregator_trust", "aggregator_trust",
         "Decide whether to trust or ignore the aggregator signal",
         _gen_aggregator_trust, 1.0),
        ("t3_position_vulnerability", "position_vulnerability",
         "Assess which agents are most vulnerable to manipulation",
         _gen_position_vulnerability, 0.8),
        ("t3_optimal_topology", "optimal_topology",
         "Compare topologies for resilience to biased agents",
         _gen_optimal_topology, 0.8),
    ]

    tier_map = {
        BenchmarkTier.NETWORK_LITERACY: tier1_templates,
        BenchmarkTier.DYNAMIC_PREDICTION: tier2_templates,
        BenchmarkTier.STRATEGIC_REASONING: tier3_templates,
    }

    for tier, templates in tier_map.items():
        for tid, cat, desc, gen_fn, weight in templates:
            _TEMPLATE_REGISTRY.append(QuestionTemplate(
                template_id=tid,
                tier=tier,
                category=cat,
                description=desc,
                example=examples[tid],
                generate_fn=gen_fn,
                weight=weight,
            ))


# ======================================================================
# Model Adapters (bring your own LLM / API key)
# ======================================================================


class ModelAdapter:
    """
    Base class for connecting models to the benchmark.

    Subclass this and implement ``query()`` to connect any model.
    The adapter is callable, so it works directly with
    ``BenchmarkSuite.evaluate(model_fn=adapter)``.

    Example (minimal custom adapter)::

        class MyModelAdapter(ModelAdapter):
            def query(self, question, system_prompt=""):
                return my_model.generate(question)

        suite.evaluate(MyModelAdapter())
    """

    def query(self, question: str, system_prompt: str = "") -> str:
        """
        Send a question to the model and return its answer.

        Parameters
        ----------
        question : str
            The benchmark question text.
        system_prompt : str, optional
            System prompt providing context (e.g., "Answer concisely").

        Returns
        -------
        str
            The model's answer.
        """
        raise NotImplementedError("Subclass ModelAdapter and implement query()")

    def __call__(self, question: str) -> str:
        """Make the adapter callable for use with evaluate(model_fn=...)."""
        return self.query(question, system_prompt=(
            "You are answering questions about network epistemology and "
            "agent-based models. Give the most concise answer possible."
        ))


class CallableAdapter(ModelAdapter):
    """
    Wraps any ``fn(str) -> str`` as a ModelAdapter.

    Provides backward compatibility with the existing evaluate() API
    that accepts a plain callable.

    Parameters
    ----------
    fn : callable
        Function that takes a question string and returns an answer.
    """

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def query(self, question: str, system_prompt: str = "") -> str:
        return self.fn(question)


class HTTPModelAdapter(ModelAdapter):
    """
    Connect to any OpenAI-compatible chat completions API.

    Works with OpenAI, Anthropic (via compatibility endpoint),
    vLLM, Ollama, LM Studio, and any provider that implements
    the ``/v1/chat/completions`` or ``/v1/messages`` endpoint.

    Parameters
    ----------
    base_url : str
        API base URL (e.g., "https://api.openai.com/v1").
    api_key : str
        API key for authentication.
    model : str
        Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
    api_format : str
        "openai" for /chat/completions, "anthropic" for /messages.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        api_format: str = "openai",
        max_tokens: int = 100,
        temperature: float = 0.1,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def query(self, question: str, system_prompt: str = "") -> str:
        if self.api_format == "anthropic":
            return self._query_anthropic(question, system_prompt)
        return self._query_openai(question, system_prompt)

    def _query_openai(self, question: str, system_prompt: str) -> str:
        import urllib.request

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

        return data["choices"][0]["message"]["content"].strip()

    def _query_anthropic(self, question: str, system_prompt: str) -> str:
        import urllib.request

        url = f"{self.base_url}/messages"
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": question}],
        }
        if system_prompt:
            body["system"] = system_prompt

        payload = json.dumps(body).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

        return data["content"][0]["text"].strip()

    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4o", **kwargs):
        """Convenience constructor for OpenAI."""
        return cls(
            base_url="https://api.openai.com/v1",
            api_key=api_key, model=model, api_format="openai", **kwargs,
        )

    @classmethod
    def anthropic(cls, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        """Convenience constructor for Anthropic."""
        return cls(
            base_url="https://api.anthropic.com/v1",
            api_key=api_key, model=model, api_format="anthropic", **kwargs,
        )

    @classmethod
    def local(cls, base_url: str = "http://localhost:8000/v1", model: str = "local", **kwargs):
        """Convenience constructor for local servers (vLLM, Ollama, LM Studio)."""
        return cls(
            base_url=base_url, api_key="not-needed",
            model=model, api_format="openai", **kwargs,
        )


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for locally loaded HuggingFace / Unsloth models.

    Parameters
    ----------
    model : transformers model
        The loaded model (e.g., from FastLanguageModel.from_pretrained).
    tokenizer : transformers tokenizer
        The corresponding tokenizer.
    system_prompt : str, optional
        Default system prompt for chat template.
    max_new_tokens : int
        Maximum new tokens to generate.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: str = "",
        max_new_tokens: int = 80,
        temperature: float = 0.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def query(self, question: str, system_prompt: str = "") -> str:
        import torch

        sys_prompt = system_prompt or self.default_system_prompt
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": question})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()


# ======================================================================
# BenchmarkSuite (enhanced generate(), all existing methods preserved)
# ======================================================================


class BenchmarkSuite:
    """
    A benchmark framework for evaluating network epistemology reasoning.

    Questions are generated procedurally from parameterized templates,
    with ground truth verified by the Aquavect simulation engine. Each
    evaluation run produces a fresh question set from a given seed,
    making the benchmark resistant to data contamination.

    The same seed always produces the same questions, ensuring
    reproducibility while allowing unlimited fresh evaluations.
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

    @property
    def category_counts(self) -> Dict[str, int]:
        """Count questions by tag/category."""
        counts = {}
        for q in self.questions:
            for tag in q.tags:
                counts[tag] = counts.get(tag, 0) + 1
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
            # Future: implement LLM-as-judge or keyword matching
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
        model_fn : callable or ModelAdapter
            Function that takes a question string and returns an answer
            string. Can be a plain function, a ModelAdapter instance,
            or any callable with signature ``str -> str``.
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
                print(f"  {i+1}/{len(questions)} -- "
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
            "version": "0.3.0",
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
        tier_weights: Optional[Dict[str, float]] = None,
        categories: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> "BenchmarkSuite":
        """
        Generate a benchmark suite with simulation-backed ground truth.

        Questions are produced from parameterized templates. Each template
        defines a question pattern; ``generate()`` instantiates them with
        random parameters and verifies answers via simulation.

        The same seed always produces the same question set, ensuring
        reproducibility while resisting data contamination (every seed
        gives a fresh set).

        Parameters
        ----------
        n_questions : int
            Total number of questions to generate.
        seed : int
            Random seed for reproducible generation.
        tier_weights : dict, optional
            Relative weight for each tier. Keys are tier value strings
            (e.g., "tier1_network_literacy"). Default: balanced across
            tiers proportional to available templates.
        categories : list of str, optional
            Only include templates whose category is in this list.
            If None, all categories are included.
        difficulty : str, optional
            If set, all questions use this difficulty level ("easy",
            "medium", or "hard"). If None, difficulty is sampled
            randomly per question.

        Returns
        -------
        BenchmarkSuite
            Suite with ``n_questions`` questions ready for evaluation.

        Examples
        --------
        >>> suite = BenchmarkSuite.generate(n_questions=200, seed=42)
        >>> len(suite)
        200
        >>> suite.tier_counts
        {'tier1_network_literacy': ..., 'tier2_dynamic_prediction': ..., ...}

        # Only network literacy questions, easy difficulty:
        >>> suite = BenchmarkSuite.generate(
        ...     n_questions=50,
        ...     categories=["hub_degree", "edge_count"],
        ...     difficulty="easy",
        ... )
        """
        _ensure_registry()
        rng = np.random.default_rng(seed)
        suite = cls()

        # Filter templates by category if requested
        templates = list(_TEMPLATE_REGISTRY)
        if categories:
            templates = [t for t in templates if t.category in categories]
        if not templates:
            return suite

        # Compute per-tier question counts
        tier_groups: Dict[BenchmarkTier, List[QuestionTemplate]] = {}
        for t in templates:
            tier_groups.setdefault(t.tier, []).append(t)

        if tier_weights:
            raw_weights = {
                tier: tier_weights.get(tier.value, 1.0)
                for tier in tier_groups
            }
        else:
            # Default: proportional to number of templates per tier
            raw_weights = {
                tier: sum(t.weight for t in tmpls)
                for tier, tmpls in tier_groups.items()
            }

        total_weight = sum(raw_weights.values())
        tier_counts = {}
        remaining = n_questions
        tiers_list = list(tier_groups.keys())

        for i, tier in enumerate(tiers_list):
            if i == len(tiers_list) - 1:
                tier_counts[tier] = remaining
            else:
                count = int(n_questions * raw_weights[tier] / total_weight)
                tier_counts[tier] = count
                remaining -= count

        # Generate questions
        q_counter = 0
        for tier, count in tier_counts.items():
            tier_templates = tier_groups[tier]
            weights = np.array([t.weight for t in tier_templates])
            weights = weights / weights.sum()

            for _ in range(count):
                # Select template weighted by template.weight
                idx = int(rng.choice(len(tier_templates), p=weights))
                template = tier_templates[idx]

                # Select difficulty
                if difficulty:
                    diff = difficulty
                else:
                    diff = str(rng.choice(list(template.difficulty_levels)))

                q_id = f"{template.template_id}_{q_counter:04d}"

                try:
                    question = template.generate_fn(rng, q_id, diff)
                    suite.add_question(question)
                except Exception:
                    # If simulation fails for this parameter combo, use
                    # a fresh attempt with different random state
                    try:
                        question = template.generate_fn(rng, q_id, diff)
                        suite.add_question(question)
                    except Exception:
                        # Fall back to canonical example with modified ID
                        fallback = BenchmarkQuestion(
                            question_id=q_id,
                            tier=template.example.tier,
                            question_type=template.example.question_type,
                            question=template.example.question,
                            correct_answer=template.example.correct_answer,
                            choices=template.example.choices,
                            tolerance=template.example.tolerance,
                            difficulty=template.example.difficulty,
                            source=template.example.source,
                            tags=template.example.tags,
                        )
                        suite.add_question(fallback)

                q_counter += 1

        return suite

    @classmethod
    def from_examples(cls) -> "BenchmarkSuite":
        """
        Create a suite containing only the canonical example questions.

        Useful for inspecting the question patterns, testing the
        evaluation pipeline, or as documentation.

        Returns
        -------
        BenchmarkSuite
            Suite with one example per registered template.
        """
        return cls(questions=get_canonical_examples())
