"""
Statistical metrics and scoring functions for epistemic simulations.

Primary outcome measure: Brier inaccuracy score (1 - credence)^2,
following Zollman (2021) and Brier (1950) as a proper scoring rule
for measuring epistemic inaccuracy.
"""

import numpy as np
from typing import Sequence

from aquavect.agents import Agent, AgentType


def brier_score(credence: float) -> float:
    """
    Brier inaccuracy: (1 - credence)^2.

    Measures distance from truth (Treatment A is superior, truth = 1).
    Perfect knowledge: 0. Maximally wrong: 1. Uninformed (0.5): 0.25.
    """
    return (1.0 - credence) ** 2


def mean_brier(agents: Sequence[Agent], exclude_biased: bool = True) -> float:
    """Mean Brier inaccuracy of truth-seeking agents."""
    if exclude_biased:
        eligible = [a for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
    else:
        eligible = [a for a in agents if a.agent_type != AgentType.AGGREGATOR]
    if not eligible:
        return 0.25
    return float(np.mean([a.brier_score for a in eligible]))


def mean_credence(agents: Sequence[Agent], exclude_biased: bool = True) -> float:
    """Mean credence of truth-seeking agents."""
    if exclude_biased:
        eligible = [a for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
    else:
        eligible = [a for a in agents if a.agent_type != AgentType.AGGREGATOR]
    if not eligible:
        return 0.5
    return float(np.mean([a.credence for a in eligible]))


def cohens_d(group1, group2) -> float:
    """
    Cohen's d with pooled standard deviation (Hedges-style denominator).

    Positive d means group2 has higher values than group1.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = g1.var(ddof=1)
    var2 = g2.var(ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return 0.0
    return float((g2.mean() - g1.mean()) / pooled_sd)


def proportion_converged(
    agents: Sequence[Agent],
    threshold: float = 0.99,
) -> float:
    """Fraction of truth-seeking agents with credence above threshold."""
    ts = [a for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
    if not ts:
        return 0.0
    return float(np.mean([a.credence > threshold for a in ts]))
