"""
Trace-to-text conversion for LLM training data (v2).

Converts raw simulation results and trajectories into structured
training examples using network profile + ego-centric representations
that give LLMs structural intuition about graph dynamics.

Every training example includes a Network Profile preamble that describes:
  - Global topology statistics (edges, density, diameter)
  - Qualitative description of information flow properties
  - Ego-centric agent perspectives for structurally distinct positions

Four example formats:
  - Mechanistic traces (~40%): Step-by-step narratives showing HOW
    damage propagates through the network's structure.
  - Contrastive pairs (~25%): Minimal-pair comparisons where one
    variable changes, teaching the model to isolate causal effects.
  - Precision drills (~20%): Short-answer questions matching
    benchmark format, teaching concise factual responses.
  - Structural grounding (~15%): Connecting topology properties
    to dynamic consequences in short paragraphs.

Output format: JSONL with instruction/input/output fields.

Usage:
    >>> from aquavect.formatting import format_training_data, save_training_data
    >>> examples = format_training_data(results, trajectories)
    >>> save_training_data(examples, "training_data.jsonl")
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np


# ──────────────────────────────────────────────
# Network Profile Generation
# ──────────────────────────────────────────────

TOPOLOGY_FLOW_DESCRIPTIONS = {
    "star": (
        "All information passes through the hub. No leaf can communicate "
        "with another leaf without going through the hub. This creates a "
        "single point of failure for information flow."
    ),
    "wheel": (
        "Information can flow through the hub (reaching all agents in 1 hop) "
        "or along the peripheral ring (slower, but provides an alternative "
        "path that bypasses the hub). The hub remains dominant but is not "
        "the sole pathway."
    ),
    "line": (
        "Information flows sequentially along the chain. Agents at the ends "
        "are most isolated — evidence must traverse the entire chain to reach "
        "them. Central agents in the chain act as information bottlenecks."
    ),
    "hierarchical": (
        "Information flows up and down a tree structure. The root has the "
        "widest reach. Agents at the same depth cannot communicate without "
        "routing through a common ancestor. Lower levels are increasingly "
        "isolated from distant branches."
    ),
    "clustered": (
        "Two dense subgroups are connected by a single bridge. Information "
        "flows freely within each cluster but must pass through the bridge "
        "to cross between them. The bridge agent is a critical bottleneck."
    ),
    "scale_free": (
        "A few highly connected hubs dominate information flow while many "
        "peripheral agents have few connections. Information spreads quickly "
        "through hubs but peripherals may depend on single sources. "
        "Resilient to random failures but vulnerable to targeted hub attacks."
    ),
    "small_world": (
        "High local clustering means neighbors tend to share connections, "
        "creating redundant local information paths. A few long-range "
        "shortcuts keep the network diameter small. Information reaches "
        "distant agents faster than in a regular lattice."
    ),
    "random": (
        "Connections are distributed relatively uniformly with some variance. "
        "No extreme hubs or bottlenecks. Information flow is moderately "
        "distributed across multiple paths."
    ),
    "complete": (
        "Every agent is directly connected to every other agent. Information "
        "is fully shared each round — all agents see all evidence. There are "
        "no structural advantages or bottlenecks. All positions are identical."
    ),
    "cycle": (
        "Agents are connected in a ring. Each agent has exactly 2 neighbors. "
        "Information must travel around the ring to reach distant agents. "
        "All positions are structurally identical."
    ),
}


def generate_network_profile(
    topology: str,
    n_agents: int,
    n_edges: int = 0,
    density: float = 0.0,
    diameter: int = 0,
    biased_positions: Sequence[int] = (),
    biased_centrality: str = "none",
    hub_degree: int = 0,
    leaf_degree: int = 0,
    n_biased: int = 0,
    bias_strength: float = 1.0,
    enable_aggregator: bool = False,
    agg_method: str = "mean",
    agg_weight: float = 0.0,
) -> str:
    """
    Generate a natural language network profile with ego-centric descriptions.

    This is the standard preamble prepended to every training example.
    """
    topo_name = topology.replace("_", " ")
    flow_desc = TOPOLOGY_FLOW_DESCRIPTIONS.get(topology, "Information flows through network connections.")

    profile = f"Network Profile:\n"
    profile += f"Topology: {topo_name} with {n_agents} agents\n"

    if n_edges > 0:
        profile += f"Edges: {n_edges} | Density: {density:.2f}"
        if diameter > 0:
            profile += f" | Diameter: {diameter}"
        profile += "\n"

    profile += f"{flow_desc}\n"

    # Ego-centric agent perspectives
    profile += "\nAgent Perspectives:\n"

    if topology == "star":
        profile += (
            f"- Hub (node 0): Has {hub_degree or n_agents - 1} direct connections "
            f"reaching every agent in 1 hop. Controls all information flow."
        )
        if n_biased > 0 and biased_centrality == "high":
            profile += " THIS AGENT IS BIASED — fabricated evidence reaches the entire community simultaneously."
        profile += "\n"
        profile += (
            f"- Leaves (nodes 1-{n_agents - 1}): Each has 1 connection to the hub only. "
            f"Cannot cross-check the hub's reports against independent sources."
        )
        if n_biased > 0 and biased_centrality == "low":
            profile += " One leaf is biased — but its fabricated evidence only reaches the hub directly."
        profile += "\n"

    elif topology == "wheel":
        profile += (
            f"- Hub (node 0): Has {hub_degree or n_agents - 1} connections to all other agents. "
            f"Dominates information flow but peripheral agents also share evidence along the ring.\n"
            f"- Rim agents (nodes 1-{n_agents - 1}): Each has 3 connections (hub + 2 ring neighbors). "
            f"Can partially cross-check hub's reports against ring neighbors.\n"
        )

    elif topology == "complete":
        profile += (
            f"- All agents: Each has {n_agents - 1} connections. "
            f"Every agent sees every other agent's evidence directly. "
            f"No structural advantage exists — all positions are identical.\n"
        )

    elif topology == "cycle":
        profile += (
            f"- All agents: Each has exactly 2 connections (left and right neighbors). "
            f"Information must travel around the ring. All positions are identical.\n"
        )

    elif topology == "line":
        profile += (
            f"- End agents (nodes 0 and {n_agents - 1}): Have 1 connection. "
            f"Most isolated — information from the other end requires {n_agents - 1} hops.\n"
            f"- Interior agents: Have 2 connections. Act as sequential relays. "
            f"Central agents see evidence from both directions.\n"
        )

    elif topology == "clustered":
        half = n_agents // 2
        profile += (
            f"- Cluster A members (nodes 0-{half - 1}): Densely connected within their group "
            f"(each sees {half - 1} other cluster members). Well-informed locally.\n"
            f"- Cluster B members (nodes {half}-{n_agents - 1}): Similarly dense internal connections.\n"
            f"- Bridge agents (nodes {half - 1} and {half}): Connect the two clusters. "
            f"Only path for cross-cluster information. Critical bottleneck.\n"
        )

    elif topology in ("scale_free", "small_world", "random", "hierarchical"):
        if hub_degree > 0:
            profile += (
                f"- Highest-connected agent: degree {hub_degree}. "
                f"Reaches many agents directly and influences information flow disproportionately."
            )
            if n_biased > 0 and biased_centrality == "high":
                profile += " THIS AGENT IS BIASED."
            profile += "\n"

        if leaf_degree > 0 and leaf_degree < hub_degree:
            profile += (
                f"- Lowest-connected agents: degree {leaf_degree}. "
                f"Limited information sources. Dependent on their few neighbors for evidence."
            )
            if n_biased > 0 and biased_centrality == "low":
                profile += " One of these agents is biased — but its reach is limited."
            profile += "\n"

    # Aggregator info
    if enable_aggregator:
        profile += (
            f"\nAggregator: A {agg_method} aggregator broadcasts a consensus signal "
            f"with weight {agg_weight:.2f}. It computes the {agg_method} credence "
            f"of all honest agents and feeds it back to the community.\n"
        )

    return profile.strip()


# ──────────────────────────────────────────────
# Qualitative descriptors
# ──────────────────────────────────────────────

def _describe_brier(brier: float) -> str:
    if brier < 0.05:
        return "very accurate beliefs (near-perfect convergence to truth)"
    elif brier < 0.15:
        return "moderately accurate beliefs"
    elif brier < 0.25:
        return "slightly better than uninformed"
    elif brier < 0.30:
        return "near-uninformed accuracy (significant epistemic damage)"
    elif brier < 0.45:
        return "substantial epistemic damage (worse than uninformed)"
    else:
        return "severe epistemic damage (strong convergence on falsehood)"


def _describe_credence(credence: float) -> str:
    if credence > 0.90:
        return "strong belief in the correct treatment"
    elif credence > 0.70:
        return "moderate belief in the correct treatment"
    elif credence > 0.55:
        return "slight lean toward the correct treatment"
    elif credence > 0.45:
        return "near-total uncertainty"
    elif credence > 0.30:
        return "slight lean toward the incorrect treatment"
    else:
        return "strong belief in the incorrect treatment"


# ──────────────────────────────────────────────
# Example type formatters
# ──────────────────────────────────────────────

def _build_profile(result: Dict) -> str:
    """Build a network profile from a simulation result dict."""
    return generate_network_profile(
        topology=result.get("topology", "unknown"),
        n_agents=result.get("n_agents", 0),
        n_biased=result.get("n_biased", 0),
        biased_centrality=result.get("biased_centrality", "none"),
        bias_strength=result.get("bias_strength", 1.0),
        enable_aggregator=result.get("enable_aggregator", False),
        agg_method=result.get("agg_method", "none"),
        agg_weight=result.get("agg_weight", 0.0),
    )


def format_mechanistic_trace(result: Dict) -> Optional[Dict]:
    """
    Step-by-step narrative showing HOW outcomes emerge from structure.
    Teaches the causal chain: structure → evidence flow → belief update → outcome.
    """
    topology = result.get("topology", "unknown")
    n_agents = result.get("n_agents", 0)
    n_biased = result.get("n_biased", 0)
    centrality = result.get("biased_centrality", "none")
    brier = result.get("final_mean_brier", 0.25)
    credence = result.get("final_mean_credence", 0.5)
    n_rounds = result.get("n_rounds", 200)
    topo_name = topology.replace("_", " ")

    profile = _build_profile(result)

    if n_biased == 0:
        question = (
            f"Given this network with no biased agents, how do the community's "
            f"beliefs evolve over {n_rounds} rounds as they investigate which "
            f"treatment is more effective?"
        )
        answer = (
            f"Without manufactured ignorance, evidence accumulates honestly. "
            f"Each round, agents test the treatments and share results with neighbors. "
            f"Because Treatment A is genuinely slightly more effective, agents testing it "
            f"observe marginally higher success rates. Over {n_rounds} rounds, this "
            f"statistical signal accumulates through the network. Agents with more "
            f"neighbors gather evidence faster. The community reaches "
            f"{_describe_credence(credence)} with mean credence {credence:.3f} "
            f"and Brier inaccuracy {brier:.3f}."
        )
    elif centrality == "high":
        question = (
            f"A biased agent is placed at the most central position in this network. "
            f"Step by step, how does this corrupt the community's beliefs?"
        )
        answer = (
            f"Round 1: The biased agent at the hub generates fabricated evidence "
            f"showing Treatment B succeeding at {result.get('bias_strength', 1.0):.0%} "
            f"rate and shares it with all its direct neighbors. "
            f"Early rounds: Neighbors receive both the fabricated evidence and their "
            f"own genuine experimental results. The fabricated signal is strong and "
            f"consistent, while genuine evidence has natural statistical noise. "
            f"Mid rounds: As neighbors' credences shift downward from the constant "
            f"fabricated input, they begin choosing to test Treatment B themselves, "
            f"further reinforcing the incorrect direction. "
            f"Late rounds: The network settles into {_describe_brier(brier)} "
            f"with mean Brier inaccuracy {brier:.3f}. The central position amplified "
            f"the damage because the biased agent's evidence reached many agents "
            f"simultaneously with no alternative information pathway to counteract it."
        )
    else:
        question = (
            f"A biased agent is placed at a peripheral position in this network. "
            f"How does the damage unfold differently compared to central placement?"
        )
        answer = (
            f"The peripheral biased agent has few direct connections, so its "
            f"fabricated evidence initially reaches only its immediate neighbors. "
            f"These neighbors also receive genuine evidence from their other connections, "
            f"which partially counteracts the fabrication. The corruption spreads slowly "
            f"outward from the periphery, diluting at each hop as agents integrate "
            f"evidence from multiple sources. After {n_rounds} rounds, the network "
            f"shows {_describe_brier(brier)} with Brier inaccuracy {brier:.3f} — "
            f"less damage than central placement because the network structure "
            f"provided redundant information paths that limited the bias's reach."
        )

    return {
        "instruction": profile + "\n\n" + question,
        "input": "",
        "output": answer,
        "metadata": {
            "example_type": "mechanistic",
            "topology": topology,
            "n_agents": n_agents,
            "n_biased": n_biased,
            "centrality": centrality,
            "brier": brier,
        },
    }


def format_contrastive_pair(result_a: Dict, result_b: Dict, contrast_var: str) -> Optional[Dict]:
    """
    Minimal-pair comparison: two scenarios differ in exactly one variable.
    Teaches the model to isolate causal effects.
    """
    profile = _build_profile(result_a)
    topo = result_a.get("topology", "unknown").replace("_", " ")
    n = result_a.get("n_agents", 0)

    brier_a = result_a.get("final_mean_brier", 0.25)
    brier_b = result_b.get("final_mean_brier", 0.25)

    if contrast_var == "position":
        question = (
            f"In this network, compare two scenarios: (A) a biased agent at the "
            f"most central position, and (B) the same biased agent at a peripheral "
            f"position. Which causes more damage and why?"
        )
        diff_pct = abs(brier_a - brier_b) / max(brier_b, 0.01) * 100
        if brier_a > brier_b:
            answer = (
                f"Scenario A (central position) causes more damage: Brier inaccuracy "
                f"{brier_a:.3f} vs {brier_b:.3f} for peripheral placement. "
                f"The central position amplifies damage by {diff_pct:.0f}% because "
                f"the biased agent's fabricated evidence reaches more agents directly. "
                f"Peripheral agents have limited reach — their fabrication only affects "
                f"their few immediate neighbors, and those neighbors can cross-check "
                f"against other information sources."
            )
        else:
            answer = (
                f"In this case, both positions cause similar damage: central {brier_a:.3f} "
                f"vs peripheral {brier_b:.3f}. This can happen in dense networks where "
                f"the structural difference between positions is small."
            )

    elif contrast_var == "aggregator":
        question = (
            f"Compare this network (A) without any aggregator vs (B) with a "
            f"{result_b.get('agg_method', 'mean')} aggregator at weight "
            f"{result_b.get('agg_weight', 0.1):.2f}. Does the aggregator help?"
        )
        if brier_b < brier_a - 0.005:
            answer = (
                f"The aggregator helps: Brier inaccuracy drops from {brier_a:.3f} "
                f"to {brier_b:.3f}. The {result_b.get('agg_method', 'mean')} aggregator "
                f"computes a consensus signal from honest agents and broadcasts it, "
                f"partially counteracting the biased evidence. However, the protection "
                f"is modest because the aggregator's signal itself is influenced by "
                f"agents whose beliefs are already partially corrupted."
            )
        elif brier_b > brier_a + 0.005:
            answer = (
                f"The aggregator actually hurts: Brier inaccuracy increases from "
                f"{brier_a:.3f} to {brier_b:.3f}. This happens because the aggregated "
                f"signal incorporates the corrupted beliefs of agents near the biased "
                f"agent, then broadcasts this corrupted consensus to agents who "
                f"previously had better local information."
            )
        else:
            answer = (
                f"The aggregator has negligible effect: Brier inaccuracy is "
                f"{brier_a:.3f} without and {brier_b:.3f} with the aggregator. "
                f"The consensus signal neither helps nor hurts significantly."
            )

    elif contrast_var == "bias_strength":
        bs_a = result_a.get("bias_strength", 1.0)
        bs_b = result_b.get("bias_strength", 1.0)
        question = (
            f"Compare two bias intensities in this network: (A) bias strength "
            f"{bs_a} vs (B) bias strength {bs_b}. Which causes more damage?"
        )
        if brier_a > brier_b:
            answer = (
                f"Bias strength {bs_a} causes more damage: Brier {brier_a:.3f} vs "
                f"{brier_b:.3f}. Stronger fabrication intensity means the biased agent "
                f"reports more extreme false evidence each round, creating a larger "
                f"statistical pull away from truth."
            )
        else:
            answer = (
                f"Bias strength {bs_b} causes more damage: Brier {brier_b:.3f} vs "
                f"{brier_a:.3f}. Higher intensity fabrication overwhelms the genuine "
                f"evidence signal more effectively."
            )

    else:
        return None

    return {
        "instruction": profile + "\n\n" + question,
        "input": "",
        "output": answer,
        "metadata": {
            "example_type": "contrastive",
            "contrast_variable": contrast_var,
            "topology": result_a.get("topology"),
            "n_agents": result_a.get("n_agents"),
            "brier_a": brier_a,
            "brier_b": brier_b,
        },
    }


def format_precision_drill(result: Dict, drill_type: str, rng: np.random.Generator) -> Optional[Dict]:
    """
    Short-answer question matching benchmark format.
    Teaches concise factual responses.
    """
    topology = result.get("topology", "unknown")
    n_agents = result.get("n_agents", 0)
    topo_name = topology.replace("_", " ")
    profile = _build_profile(result)

    if drill_type == "hub_degree":
        # We need to compute this from the topology
        from aquavect.networks import create_network, get_high_centrality_positions
        G = create_network(topology, n_agents, seed=result.get("seed", 42))
        hub = get_high_centrality_positions(G, 1)[0]
        degree = G.degree(hub)
        question = f"What is the degree of the most connected node in this network? Answer with just the number."
        answer = str(degree)

    elif drill_type == "edge_count":
        from aquavect.networks import create_network
        G = create_network(topology, n_agents, seed=result.get("seed", 42))
        n_edges = G.number_of_edges()
        question = f"How many edges does this network have? Answer with just the number."
        answer = str(n_edges)

    elif drill_type == "brier_range":
        brier = result.get("final_mean_brier", 0.25)
        if brier < 0.15:
            correct = "low"
        elif brier < 0.30:
            correct = "moderate"
        else:
            correct = "high"
        question = (
            f"A biased agent is in this network. After 200 rounds, is the "
            f"epistemic damage low (Brier < 0.15), moderate (0.15-0.30), "
            f"or high (> 0.30)? Answer: low, moderate, or high."
        )
        answer = correct

    elif drill_type == "position_effect":
        question = (
            f"In this network, does placing a biased agent at the most "
            f"central position cause more damage than a peripheral position? "
            f"Answer yes or no."
        )
        answer = "yes"  # True for all asymmetric topologies in our research

    elif drill_type == "convergence":
        credence = result.get("final_mean_credence", 0.5)
        question = (
            f"After 200 rounds with no biased agents, does this network's "
            f"mean credence exceed 0.70? Answer yes or no."
        )
        answer = "yes" if credence > 0.70 else "no"

    elif drill_type == "symmetry":
        from aquavect.networks import SYMMETRIC_TOPOLOGIES
        is_sym = topology in SYMMETRIC_TOPOLOGIES
        question = (
            f"In this {topo_name} network, do all nodes have the same degree? "
            f"Answer yes or no."
        )
        answer = "yes" if is_sym else "no"

    else:
        return None

    return {
        "instruction": profile + "\n\n" + question,
        "input": "",
        "output": answer,
        "metadata": {
            "example_type": "drill",
            "drill_type": drill_type,
            "topology": topology,
            "n_agents": n_agents,
        },
    }


def format_structural_grounding(result: Dict) -> Optional[Dict]:
    """
    Short paragraph connecting topology properties to dynamic consequences.
    Bridges structural facts to epistemic outcomes.
    """
    topology = result.get("topology", "unknown")
    n_agents = result.get("n_agents", 0)
    brier = result.get("final_mean_brier", 0.25)
    centrality = result.get("biased_centrality", "none")
    topo_name = topology.replace("_", " ")
    profile = _build_profile(result)

    question = (
        f"What structural properties of this network are most relevant "
        f"for understanding how manufactured ignorance would spread through it?"
    )

    if topology == "star":
        answer = (
            f"The star topology's defining feature is extreme centralization: "
            f"the hub has degree {n_agents - 1} while all leaves have degree 1. "
            f"This means the hub is a single point of failure. A biased hub's "
            f"evidence reaches {n_agents - 1} agents in one hop with no alternative "
            f"pathway for verification. Leaves are maximally vulnerable because "
            f"they have zero independent information sources."
        )
    elif topology == "complete":
        answer = (
            f"The complete topology provides maximum redundancy: every agent "
            f"has {n_agents - 1} connections. A biased agent's evidence is just "
            f"one signal among {n_agents - 1} that each agent receives. This "
            f"dilution effect makes complete networks highly resilient to "
            f"individual manipulation. All positions are structurally identical, "
            f"so there is no position effect."
        )
    elif topology == "clustered":
        half = n_agents // 2
        answer = (
            f"The clustered topology creates two information environments: "
            f"within each cluster ({half} agents, densely connected), information "
            f"circulates freely. Between clusters, the single bridge is a bottleneck. "
            f"A biased agent inside one cluster corrupts that cluster's beliefs "
            f"but the damage may not spread to the other cluster unless it "
            f"reaches the bridge agent."
        )
    elif topology == "scale_free":
        answer = (
            f"The scale-free topology has a heavy-tailed degree distribution. "
            f"A few hubs concentrate information flow while many peripherals "
            f"depend on single connections. This makes the network resilient "
            f"to random agent corruption (most agents are low-degree) but "
            f"highly vulnerable to targeted corruption of hubs. The position "
            f"effect is strong because hub vs peripheral placement determines "
            f"how much of the network the biased evidence can reach."
        )
    else:
        answer = (
            f"This {topo_name} network with {n_agents} agents has structural "
            f"asymmetry that creates differential vulnerability. Agents with "
            f"more connections receive diverse evidence and can cross-check "
            f"sources, making them harder to mislead. Agents with fewer "
            f"connections are dependent on limited sources and more susceptible "
            f"to fabricated evidence from any single neighbor."
        )

    return {
        "instruction": profile + "\n\n" + question,
        "input": "",
        "output": answer,
        "metadata": {
            "example_type": "structural",
            "topology": topology,
            "n_agents": n_agents,
        },
    }


# ──────────────────────────────────────────────
# Main formatting pipeline
# ──────────────────────────────────────────────

def format_training_data(
    results: List[Dict],
    trajectories: Optional[List[Dict]] = None,
    mechanistic_fraction: float = 0.40,
    contrastive_fraction: float = 0.25,
    drill_fraction: float = 0.20,
    structural_fraction: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """
    Convert simulation results into formatted training examples.

    Uses network profile + ego-centric representations across four
    example types: mechanistic traces, contrastive pairs, precision
    drills, and structural grounding.
    """
    rng = np.random.default_rng(seed)
    examples = []
    n_total = len(results)

    n_mechanistic = int(n_total * mechanistic_fraction)
    n_contrastive = int(n_total * contrastive_fraction)
    n_drill = int(n_total * drill_fraction)
    n_structural = n_total - n_mechanistic - n_contrastive - n_drill

    # Shuffle results
    indices = rng.permutation(n_total)

    # --- Mechanistic traces ---
    for idx in indices[:n_mechanistic]:
        ex = format_mechanistic_trace(results[idx])
        if ex:
            examples.append(ex)

    # --- Contrastive pairs ---
    # Group results by topology+size for pairing
    groups = {}
    for r in results:
        key = (r.get("topology"), r.get("n_agents"))
        groups.setdefault(key, []).append(r)

    contrastive_count = 0
    for key, group in groups.items():
        if contrastive_count >= n_contrastive:
            break
        high = [r for r in group if r.get("biased_centrality") == "high"]
        low = [r for r in group if r.get("biased_centrality") == "low"]
        no_agg = [r for r in group if not r.get("enable_aggregator", False) and r.get("n_biased", 0) > 0]
        with_agg = [r for r in group if r.get("enable_aggregator", False)]

        # Position contrasts
        for h, l in zip(high[:3], low[:3]):
            if contrastive_count >= n_contrastive:
                break
            ex = format_contrastive_pair(h, l, "position")
            if ex:
                examples.append(ex)
                contrastive_count += 1

        # Aggregator contrasts
        for na, wa in zip(no_agg[:2], with_agg[:2]):
            if contrastive_count >= n_contrastive:
                break
            ex = format_contrastive_pair(na, wa, "aggregator")
            if ex:
                examples.append(ex)
                contrastive_count += 1

    # --- Precision drills ---
    drill_types = ["hub_degree", "edge_count", "brier_range", "position_effect",
                   "convergence", "symmetry"]
    drill_idx = n_mechanistic + n_contrastive
    for i, idx in enumerate(indices[drill_idx:drill_idx + n_drill]):
        dt = drill_types[i % len(drill_types)]
        ex = format_precision_drill(results[idx], dt, rng)
        if ex:
            examples.append(ex)

    # --- Structural grounding ---
    struct_idx = drill_idx + n_drill
    for idx in indices[struct_idx:struct_idx + n_structural]:
        ex = format_structural_grounding(results[idx])
        if ex:
            examples.append(ex)

    # Shuffle final dataset
    rng.shuffle(examples)
    return examples


def save_training_data(
    examples: List[Dict],
    output_path: str = "training_data.jsonl",
    include_metadata: bool = False,
) -> str:
    """Save training examples as JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            if include_metadata:
                f.write(json.dumps(example) + "\n")
            else:
                clean = {
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": example["output"],
                }
                f.write(json.dumps(clean) + "\n")

    return output_path


def dataset_statistics(examples: List[Dict]) -> Dict:
    """Compute statistics about a formatted dataset."""
    stats = {
        "total_examples": len(examples),
        "type_counts": {},
        "topology_counts": {},
        "strategy_counts": {},
        "avg_instruction_length": 0,
        "avg_output_length": 0,
        "avg_total_tokens_estimate": 0,
    }

    inst_lengths = []
    out_lengths = []

    for ex in examples:
        meta = ex.get("metadata", {})
        etype = meta.get("example_type", "unknown")
        stats["type_counts"][etype] = stats["type_counts"].get(etype, 0) + 1

        topo = meta.get("topology", "unknown")
        stats["topology_counts"][topo] = stats["topology_counts"].get(topo, 0) + 1

        strat = meta.get("sampling_strategy", "unknown")
        stats["strategy_counts"][strat] = stats["strategy_counts"].get(strat, 0) + 1

        inst_lengths.append(len(ex.get("instruction", "")))
        out_lengths.append(len(ex.get("output", "")))

    if inst_lengths:
        stats["avg_instruction_length"] = int(np.mean(inst_lengths))
        stats["avg_output_length"] = int(np.mean(out_lengths))
        stats["avg_total_tokens_estimate"] = int(
            (np.mean(inst_lengths) + np.mean(out_lengths)) / 4
        )

    return stats
