"""
Information aggregation mechanisms for epistemic networks.

Implements the aggregator node from Paper 2: "Does Market Aggregation
Protect or Amplify Manufactured Agnotology?"

The aggregator models the information-aggregation function of a prediction
market. It computes a summary credence from non-intransigent agents and
broadcasts it back to the network. Agents who consult the aggregator
partially update their beliefs toward its signal.

Three aggregation methods are supported:
  - mean: simple average (analogous to a poll)
  - median: robust to outliers (median voter theorem analog)
  - productivity_weighted: volume-weighted (analogous to market prices
    where larger trades move the price more)

Key finding from Paper 2: median aggregation substantially outperforms
mean and productivity-weighted, and eliminates the low-centrality
reversal observed with mean at high weights.
"""

import numpy as np
from typing import List, Sequence

from aquavect.agents import Agent, AgentType


def compute_aggregator_credence(
    agents: Sequence[Agent],
    method: str = "mean",
) -> float:
    """
    Compute the aggregator's consensus credence from eligible agents.

    Excludes intransigent and aggregator-type agents from the computation,
    modeling the assumption that manipulators operate through the evidence
    channel rather than directly trading in the market.

    Parameters
    ----------
    agents : sequence of Agent
        All agents in the network.
    method : str
        Aggregation function: "mean", "median", or "productivity_weighted".

    Returns
    -------
    float
        Consensus credence (0 to 1).
    """
    contributing = [
        a for a in agents
        if a.agent_type not in (AgentType.AGGREGATOR, AgentType.INTRANSIGENT)
    ]
    if not contributing:
        return 0.5

    credences = np.array([a.credence for a in contributing])

    if method == "median":
        return float(np.median(credences))
    elif method == "productivity_weighted":
        weights = np.array(
            [a.effective_productivity for a in contributing], dtype=float
        )
        if weights.sum() > 0:
            return float(np.average(credences, weights=weights))
        return float(np.mean(credences))
    else:  # mean (default)
        return float(np.mean(credences))


def apply_aggregator_update(
    agent: Agent,
    agg_credence: float,
    agg_weight: float,
    rng: np.random.Generator,
) -> None:
    """
    Update an agent's beliefs toward the aggregator's consensus signal.

    Adds pseudo-observations to the agent's Beta distribution that would
    move their credence toward agg_credence. The number of pseudo-observations
    is proportional to agg_weight, scaled relative to the agent's existing
    evidence base.

    The aggregator's absolute influence grows with accumulated evidence,
    but its relative influence stays proportional. This means the aggregator
    matters most in early rounds when evidence is thin — modeling how
    experienced traders are less moved by market prices than novices.

    Parameters
    ----------
    agent : Agent
        The agent to update (modified in place).
    agg_credence : float
        The aggregator's current consensus credence.
    agg_weight : float
        How heavily to weight the aggregator signal (0 to 1).
    rng : numpy.random.Generator
        Random number generator (unused but kept for API consistency).
    """
    if agg_weight <= 0:
        return

    total_evidence = agent.alpha + agent.beta - 2  # subtract priors
    pseudo_n = max(1.0, agg_weight * max(total_evidence, 10.0))

    agent.alpha += pseudo_n * agg_credence
    agent.beta += pseudo_n * (1.0 - agg_credence)
