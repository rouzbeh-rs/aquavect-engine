"""
Core simulation engine for Bayesian network epistemology experiments.

Implements the Bala-Goyal (1998) framework as adapted by Holman & Bruner
(2015, 2017), with the aggregator extension from Paper 2.

Each simulation creates a network of agents who:
  1. Choose which treatment to test based on current credence
  2. Generate evidence via Binomial draws
  3. Observe own and neighbors' evidence
  4. Update Beta distributions via Bayesian inference
  5. Optionally consult an aggregator node

Biased agents never update and fabricate evidence (intransigent mechanism)
or influence methodology through the Moran process (industrial selection).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aquavect.agents import Agent, AgentType
from aquavect.aggregation import compute_aggregator_credence, apply_aggregator_update
from aquavect.networks import (
    get_centrality_measures,
    get_network_properties,
    SYMMETRIC_TOPOLOGIES,
)


@dataclass
class SimulationConfig:
    """
    All parameters controlling a single simulation run.

    This is the canonical configuration object. Every parameter that affects
    simulation behavior is defined here with its default value and documentation.
    """

    # --- Environment ---
    n_rounds: int = 200
    efficacy_difference: float = 0.05  # delta: P_A = 0.50 + delta
    n_trials: int = 100  # trials per round per agent (truth-seekers)

    # --- Priors ---
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    heterogeneous_priors: bool = False

    # --- Intransigent bias ---
    bias_strength: float = 1.0  # fabricated B success rate
    intransigent_n_trials: int = 100  # fixed productivity for biased agent

    # --- Heterogeneous productivity (Paper 2 / v3) ---
    truth_seeker_prod_min: int = 100  # set to 50-200 for Paper 2
    truth_seeker_prod_max: int = 100  # set equal for uniform productivity

    # --- Aggregator (Paper 2) ---
    enable_aggregator: bool = False
    agg_frequency: float = 1.0  # prob per round of consulting aggregator
    agg_weight: float = 0.1  # how heavily agents anchor on signal
    agg_method: str = "mean"  # "mean", "median", "productivity_weighted"

    # --- Industrial selection (Holman-Bruner 2017) ---
    enable_moran: bool = False
    exit_prob: float = 0.20
    meth_bias_variance: float = 0.02
    industry_threshold: float = 0.10
    base_productivity_min: int = 1
    base_productivity_max: int = 5
    funding_boost: int = 50

    # --- Output control ---
    save_trajectory: bool = False
    convergence_threshold: float = 0.99


def run_simulation(
    G,
    topology: str,
    n_agents: int,
    biased_positions: Sequence[int] = (),
    bias_type: str = "intransigent",
    condition_name: str = "",
    phase: str = "",
    seed: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
    # Direct parameter overrides (convenience API)
    **kwargs,
) -> Tuple[Dict, Optional[List[Dict]]]:
    """
    Run a single epistemic network simulation.

    Parameters
    ----------
    G : networkx.Graph
        Pre-built network graph.
    topology : str
        Name of the topology (for labeling output).
    n_agents : int
        Number of agents in the network.
    biased_positions : sequence of int
        Node indices where biased agents are placed.
    bias_type : str
        "intransigent" (Paper 1) or "industrial_selection" (Paper 2 / IS).
    condition_name : str
        Label for this experimental condition (e.g., "1_high", "control").
    phase : str
        Experiment phase identifier for filtering.
    seed : int, optional
        Random seed for reproducibility.
    config : SimulationConfig, optional
        Full configuration object. If None, uses defaults + kwargs.
    **kwargs
        Override individual SimulationConfig fields (e.g., n_rounds=500).

    Returns
    -------
    result : dict
        Summary statistics for this simulation run.
    trajectory : list of dict or None
        Round-by-round data if config.save_trajectory is True.
    """
    # Build config
    if config is None:
        config = SimulationConfig(**kwargs)
    elif kwargs:
        # Apply overrides to a copy
        cfg_dict = {
            f.name: getattr(config, f.name)
            for f in config.__dataclass_fields__.values()
        }
        cfg_dict.update(kwargs)
        config = SimulationConfig(**cfg_dict)

    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)  # for networkx compatibility

    biased_positions = list(biased_positions)
    p_A_true = 0.50 + config.efficacy_difference
    p_B_true = 0.50
    topology_class = "symmetric" if topology in SYMMETRIC_TOPOLOGIES else "asymmetric"

    adj_list = [list(G.neighbors(i)) for i in range(n_agents)]

    # --- Create agents ---
    agents = []
    for i in range(n_agents):
        if i in biased_positions:
            atype = (
                AgentType.INTRANSIGENT
                if bias_type == "intransigent"
                else AgentType.INDUSTRIAL_SELECTION
            )
        else:
            atype = AgentType.TRUTH_SEEKER

        # Priors
        if config.heterogeneous_priors and atype == AgentType.TRUTH_SEEKER:
            alpha = float(rng.uniform(1, 5))
            beta = float(rng.uniform(1, 5))
        else:
            alpha = config.prior_alpha
            beta = config.prior_beta

        # Productivity
        if bias_type == "industrial_selection":
            base_prod = int(rng.integers(
                config.base_productivity_min,
                config.base_productivity_max + 1
            ))
        elif atype == AgentType.INTRANSIGENT:
            base_prod = config.intransigent_n_trials
        else:
            base_prod = int(rng.integers(
                config.truth_seeker_prod_min,
                config.truth_seeker_prod_max + 1
            ))

        agents.append(Agent(
            agent_id=i,
            agent_type=atype,
            alpha=alpha,
            beta=beta,
            personal_p_A=p_A_true,
            personal_p_B=p_B_true,
            base_productivity=base_prod,
            bias_strength=config.bias_strength,
        ))

    # --- Industrial selection setup ---
    if bias_type == "industrial_selection":
        _assign_methodological_biases(
            agents, p_A_true, p_B_true, config.meth_bias_variance, rng
        )
        _apply_industry_funding(
            agents, biased_positions, config.industry_threshold,
            config.funding_boost, p_B_true
        )

    protected_positions = set(biased_positions)
    trajectory = [] if config.save_trajectory else None
    moran_events = 0
    aggregator_credence = 0.5

    # --- Main simulation loop ---
    for round_num in range(config.n_rounds):

        # Compute aggregator credence
        if config.enable_aggregator:
            aggregator_credence = compute_aggregator_credence(
                agents, method=config.agg_method
            )

        # Evidence generation
        evidence = {}
        for agent in agents:
            if agent.agent_type == AgentType.INTRANSIGENT:
                nt = config.intransigent_n_trials
                successes = int(rng.binomial(nt, agent.bias_strength))
                evidence[agent.agent_id] = ("B", successes, nt)
            else:
                nt = agent.effective_productivity
                if agent.credence >= 0.5:
                    p = (
                        agent.personal_p_A
                        if bias_type == "industrial_selection"
                        else p_A_true
                    )
                    successes = int(rng.binomial(nt, p))
                    evidence[agent.agent_id] = ("A", successes, nt)
                else:
                    p = (
                        agent.personal_p_B
                        if bias_type == "industrial_selection"
                        else p_B_true
                    )
                    successes = int(rng.binomial(nt, p))
                    evidence[agent.agent_id] = ("B", successes, nt)

        # Belief updating from network evidence
        for agent in agents:
            if agent.agent_type == AgentType.INTRANSIGENT:
                continue

            tA_succ = tA_tri = tB_succ = tB_tri = 0

            # Own evidence
            own_t, own_s, own_n = evidence[agent.agent_id]
            if own_t == "A":
                tA_succ, tA_tri = own_s, own_n
            else:
                tB_succ, tB_tri = own_s, own_n

            # Neighbor evidence
            for nb in adj_list[agent.agent_id]:
                nt, ns, nn = evidence[nb]
                if nt == "A":
                    tA_succ += ns
                    tA_tri += nn
                else:
                    tB_succ += ns
                    tB_tri += nn

            # Bayesian update (single-Beta simplification)
            # alpha' = alpha + s_A + (n_other - s_other)
            # beta'  = beta  + (n_A - s_A) + s_other
            if tA_tri > 0:
                agent.alpha += tA_succ
                agent.beta += tA_tri - tA_succ
            if tB_tri > 0:
                agent.beta += tB_succ
                agent.alpha += tB_tri - tB_succ

        # Aggregator influence
        if config.enable_aggregator:
            for agent in agents:
                if agent.agent_type == AgentType.INTRANSIGENT:
                    continue
                if rng.random() < config.agg_frequency:
                    apply_aggregator_update(
                        agent, aggregator_credence, config.agg_weight, rng
                    )

        # Moran process (industrial selection)
        if config.enable_moran and rng.random() < config.exit_prob:
            _moran_replacement(
                agents, adj_list, protected_positions,
                p_A_true, p_B_true, config.meth_bias_variance, rng,
                config.base_productivity_min, config.base_productivity_max,
            )
            moran_events += 1
            if bias_type == "industrial_selection":
                _apply_industry_funding(
                    agents, biased_positions, config.industry_threshold,
                    config.funding_boost, p_B_true
                )

        # Save trajectory
        if trajectory is not None:
            ts_agents = [
                a for a in agents if a.agent_type == AgentType.TRUTH_SEEKER
            ]
            if not ts_agents:
                ts_agents = [
                    a for a in agents
                    if a.agent_type != AgentType.INTRANSIGENT
                ]
            if ts_agents:
                ts_briers = [a.brier_score for a in ts_agents]
                ts_creds = [a.credence for a in ts_agents]
                trajectory.append({
                    "round": round_num,
                    "mean_brier": float(np.mean(ts_briers)),
                    "std_brier": float(np.std(ts_briers)),
                    "mean_credence": float(np.mean(ts_creds)),
                    "std_credence": float(np.std(ts_creds)),
                    "aggregator_credence": (
                        aggregator_credence
                        if config.enable_aggregator
                        else None
                    ),
                })

    # --- Compute final metrics ---
    ts_agents = [a for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
    if not ts_agents:
        ts_agents = [
            a for a in agents if a.agent_type != AgentType.INTRANSIGENT
        ]
    ts_briers = [a.brier_score for a in ts_agents]
    ts_creds = [a.credence for a in ts_agents]
    if not ts_briers:
        ts_briers = [0.25]
        ts_creds = [0.5]

    # Centrality label from condition name
    cn = condition_name.lower()
    if "high" in cn:
        bc_label = "high"
    elif "low" in cn:
        bc_label = "low"
    else:
        bc_label = "none"

    result = {
        "phase": phase,
        "topology": topology,
        "topology_class": topology_class,
        "n_agents": n_agents,
        "n_rounds": config.n_rounds,
        "condition": condition_name,
        "bias_type": bias_type,
        "biased_centrality": bc_label,
        "n_biased": len(biased_positions),
        "bias_strength": config.bias_strength,
        "enable_aggregator": config.enable_aggregator,
        "agg_frequency": config.agg_frequency if config.enable_aggregator else 0.0,
        "agg_weight": config.agg_weight if config.enable_aggregator else 0.0,
        "agg_method": config.agg_method if config.enable_aggregator else "none",
        "final_mean_brier": float(np.mean(ts_briers)),
        "final_std_brier": float(np.std(ts_briers)),
        "final_mean_credence": float(np.mean(ts_creds)),
        "final_std_credence": float(np.std(ts_creds)),
        "final_agg_credence": (
            aggregator_credence if config.enable_aggregator else None
        ),
        "proportion_converged": float(np.mean([
            c > config.convergence_threshold for c in ts_creds
        ])),
        "total_moran_events": moran_events,
        "seed": seed,
    }

    return result, trajectory


# ──────────────────────────────────────────────
# Industrial Selection helpers
# ──────────────────────────────────────────────

def _assign_methodological_biases(agents, p_A_true, p_B_true, variance, rng):
    sigma = np.sqrt(variance)
    for agent in agents:
        if agent.agent_type not in (AgentType.INTRANSIGENT, AgentType.AGGREGATOR):
            agent.personal_p_A = float(np.clip(rng.normal(p_A_true, sigma), 0.01, 0.99))
            agent.personal_p_B = float(np.clip(rng.normal(p_B_true, sigma), 0.01, 0.99))


def _apply_industry_funding(agents, is_positions, threshold, boost, p_B_true):
    for agent in agents:
        if agent.agent_type in (AgentType.INTRANSIGENT, AgentType.AGGREGATOR):
            continue
        if agent.agent_id in is_positions:
            agent.funding_boost = boost
            agent.is_funded = True
            if agent.personal_p_B <= p_B_true + threshold:
                agent.personal_p_B = p_B_true + threshold + 0.01
        else:
            if agent.personal_p_B > p_B_true + threshold:
                agent.funding_boost = boost
                agent.is_funded = True
            else:
                agent.funding_boost = 0
                agent.is_funded = False


def _moran_replacement(
    agents, adj_list, protected, p_A_true, p_B_true,
    meth_bias_variance, rng, prod_min, prod_max
):
    replaceable = [a for a in agents if a.agent_id not in protected]
    if not replaceable:
        return
    leaving = rng.choice(replaceable)
    leaving_id = leaving.agent_id
    neighbor_ids = set(adj_list[leaving_id])
    neighbor_agents = [
        a for a in agents
        if a.agent_id in neighbor_ids
        and a.agent_type != AgentType.AGGREGATOR
    ]
    if not neighbor_agents:
        neighbor_agents = [
            a for a in agents
            if a.agent_id != leaving_id
            and a.agent_type != AgentType.AGGREGATOR
        ]
    if not neighbor_agents:
        return
    productivities = np.array(
        [a.effective_productivity for a in neighbor_agents], dtype=float
    )
    if productivities.sum() > 0:
        probs = productivities / productivities.sum()
    else:
        probs = np.ones(len(productivities)) / len(productivities)
    mentor = neighbor_agents[rng.choice(len(neighbor_agents), p=probs)]
    leaving.alpha = 1.0
    leaving.beta = 1.0
    sigma = np.sqrt(meth_bias_variance) * 0.1
    leaving.personal_p_A = float(np.clip(rng.normal(mentor.personal_p_A, sigma), 0.01, 0.99))
    leaving.personal_p_B = float(np.clip(rng.normal(mentor.personal_p_B, sigma), 0.01, 0.99))
    leaving.funding_boost = 0
    leaving.is_funded = False
    leaving.base_productivity = int(rng.integers(prod_min, prod_max + 1))
