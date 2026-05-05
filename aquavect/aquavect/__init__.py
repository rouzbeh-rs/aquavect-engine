"""
Aquavect: Agent-based network epistemology simulation engine.

Provides tools for simulating Bayesian agents, strategic agents, and
information aggregators in epistemic networks, based on the Bala-Goyal /
Zollman / Holman-Bruner research lineage.

Quick start:
    >>> from aquavect import create_network, Agent, AgentType, run_simulation
    >>> G = create_network("star", 10, seed=42)
    >>> result, trajectory = run_simulation(
    ...     G=G, topology="star", n_agents=10,
    ...     biased_positions=[0], seed=42
    ... )
    >>> print(f"Mean Brier inaccuracy: {result['final_mean_brier']:.4f}")
"""

__version__ = "0.1.0"

from aquavect.agents import Agent, AgentType
from aquavect.networks import (
    create_network,
    get_high_centrality_positions,
    get_low_centrality_positions,
    get_centrality_measures,
    get_network_properties,
    ASYMMETRIC_TOPOLOGIES,
    SYMMETRIC_TOPOLOGIES,
    ALL_TOPOLOGIES,
)
from aquavect.simulation import run_simulation, SimulationConfig
from aquavect.aggregation import compute_aggregator_credence, apply_aggregator_update
from aquavect.metrics import brier_score, cohens_d

__all__ = [
    "Agent",
    "AgentType",
    "create_network",
    "get_high_centrality_positions",
    "get_low_centrality_positions",
    "get_centrality_measures",
    "get_network_properties",
    "run_simulation",
    "SimulationConfig",
    "compute_aggregator_credence",
    "apply_aggregator_update",
    "brier_score",
    "cohens_d",
    "ASYMMETRIC_TOPOLOGIES",
    "SYMMETRIC_TOPOLOGIES",
    "ALL_TOPOLOGIES",
]
