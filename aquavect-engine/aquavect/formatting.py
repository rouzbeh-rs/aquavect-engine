"""
Trace-to-text conversion for LLM training data.

Converts raw simulation results and trajectories into structured
training examples suitable for fine-tuning language models on
network-structured decision reasoning.

Two example formats:
  - QA pairs (~80%): Question about a network configuration, answer
    describing the outcome and structural reasoning.
  - Agent-perspective scenarios (~20%): Situated decision problem
    from a single agent's perspective.

Output format: JSONL with instruction/input/output fields, compatible
with standard fine-tuning frameworks (Axolotl, LLaMA-Factory, etc.).

Usage:
    >>> from aquavect.formatting import format_training_data
    >>> examples = format_training_data(results, trajectories)
    >>> save_training_data(examples, "training_data.jsonl")
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Topology descriptions for natural language
# ──────────────────────────────────────────────

TOPOLOGY_DESCRIPTIONS = {
    "star": "a star network where one central hub is connected to all other agents",
    "wheel": "a wheel network (star with peripheral agents also connected in a ring)",
    "line": "a line network where agents are connected sequentially in a chain",
    "hierarchical": "a hierarchical (binary tree) network with branching authority structure",
    "clustered": "a clustered network with two dense subgroups connected by a single bridge",
    "scale_free": "a scale-free network with preferential attachment (some highly connected hubs)",
    "small_world": "a small-world network with high local clustering and short path lengths",
    "random": "a random network with uniform connection probability",
    "complete": "a complete network where every agent is connected to every other agent",
    "cycle": "a cycle network where agents are connected in a ring",
}

CENTRALITY_EXPLANATIONS = {
    "high": "at the most structurally central position (highest degree centrality)",
    "low": "at a peripheral position (lowest degree centrality)",
    "none": "",
}

AGG_METHOD_DESCRIPTIONS = {
    "mean": "simple average (mean)",
    "median": "median (robust to outliers)",
    "productivity_weighted": "productivity-weighted average (analogous to volume-weighted market pricing)",
}


def _describe_brier(brier: float) -> str:
    """Convert a Brier score to a qualitative description."""
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
    """Convert a credence to a qualitative description."""
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


def format_qa_pair(result: Dict) -> Optional[Dict]:
    """
    Format a simulation result as a QA training example.

    Returns dict with instruction/input/output fields, or None if
    the result doesn't have enough information for a good example.
    """
    topology = result.get("topology", "unknown")
    n_agents = result.get("n_agents", 0)
    n_biased = result.get("n_biased", 0)
    centrality = result.get("biased_centrality", "none")
    bias_strength = result.get("bias_strength", 1.0)
    brier = result.get("final_mean_brier", 0.25)
    credence = result.get("final_mean_credence", 0.5)
    n_rounds = result.get("n_rounds", 200)
    enable_agg = result.get("enable_aggregator", False)
    agg_method = result.get("agg_method", "none")
    agg_weight = result.get("agg_weight", 0.0)
    efficacy = result.get("efficacy_difference", 0.05) if "efficacy_difference" in result else 0.05

    topo_desc = TOPOLOGY_DESCRIPTIONS.get(topology, f"a {topology} network")

    # Build question
    if n_biased == 0:
        # Control condition
        instruction = (
            f"Consider a community of {n_agents} truth-seeking researchers connected in "
            f"{topo_desc}. They are investigating which of two treatments is more effective, "
            f"where Treatment A is genuinely {efficacy*100:.0f}% more effective than Treatment B. "
            f"Each round, they run experiments and share results with their neighbors. "
            f"After {n_rounds} rounds of investigation, what happens to the community's beliefs?"
        )
    else:
        bias_desc = f"{n_biased} biased agent{'s' if n_biased > 1 else ''}"
        pos_desc = CENTRALITY_EXPLANATIONS.get(centrality, "")
        instruction = (
            f"Consider a community of {n_agents} researchers connected in {topo_desc}. "
            f"There {'are' if n_biased > 1 else 'is'} {bias_desc} placed {pos_desc} "
            f"who fabricate{'s' if n_biased == 1 else ''} evidence with intensity "
            f"{bias_strength:.0%}, always promoting the inferior Treatment B. "
        )

        if enable_agg:
            method_desc = AGG_METHOD_DESCRIPTIONS.get(agg_method, agg_method)
            instruction += (
                f"The community also has access to a market-like aggregator that computes "
                f"a consensus signal using {method_desc} and broadcasts it back with "
                f"weight {agg_weight:.2f}. "
            )

        instruction += (
            f"After {n_rounds} rounds, what happens to the community's beliefs about "
            f"which treatment is better?"
        )

    # Build answer
    brier_desc = _describe_brier(brier)
    credence_desc = _describe_credence(credence)

    output_parts = []

    output_parts.append(
        f"After {n_rounds} rounds, the community reaches {brier_desc}, "
        f"with a mean Brier inaccuracy of {brier:.3f} and an average credence "
        f"of {credence:.3f} ({credence_desc})."
    )

    if n_biased == 0:
        output_parts.append(
            f"Without any manufactured ignorance, the agents gradually learn "
            f"from accumulated evidence that Treatment A is superior."
        )
    elif centrality == "high":
        output_parts.append(
            f"The biased agent's central position amplifies the manufactured ignorance "
            f"because it shares fabricated evidence with many neighbors simultaneously, "
            f"and those neighbors have limited alternative information pathways."
        )
    elif centrality == "low":
        output_parts.append(
            f"The biased agent's peripheral position limits its damage because "
            f"it reaches fewer agents directly, and most agents receive correct "
            f"evidence from other parts of the network."
        )

    if enable_agg:
        if brier < 0.25:
            output_parts.append(
                f"The {agg_method} aggregator provides some protection by broadcasting "
                f"a consensus signal that partially counteracts the biased evidence."
            )
        else:
            output_parts.append(
                f"The {agg_method} aggregator at weight {agg_weight:.2f} does not "
                f"provide sufficient protection. The aggregated signal itself becomes "
                f"partially corrupted by the community's shifting beliefs."
            )

    # Add structural reasoning
    if topology in ("star", "wheel") and centrality == "high":
        output_parts.append(
            f"Star and wheel topologies are particularly vulnerable to central bias "
            f"because the hub has direct connections to all or most agents, "
            f"creating a single-point-of-failure for the community's epistemics."
        )
    elif topology == "clustered" and n_biased > 0:
        output_parts.append(
            f"In a clustered network, the bridge position between subgroups "
            f"can either protect (if the biased agent is inside one cluster) "
            f"or amplify damage (if placed at the bridge)."
        )

    output = " ".join(output_parts)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": {
            "example_type": "qa",
            "topology": topology,
            "n_agents": n_agents,
            "n_biased": n_biased,
            "centrality": centrality,
            "brier": brier,
            "sampling_strategy": result.get("sampling_strategy", "unknown"),
        },
    }


def format_agent_scenario(result: Dict, trajectories: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    Format a simulation result as an agent-perspective decision scenario.

    Creates a situated decision problem from a single agent's viewpoint.
    """
    topology = result.get("topology", "unknown")
    n_agents = result.get("n_agents", 0)
    n_biased = result.get("n_biased", 0)
    centrality = result.get("biased_centrality", "none")
    brier = result.get("final_mean_brier", 0.25)
    credence = result.get("final_mean_credence", 0.5)

    if n_biased == 0:
        return None  # Scenarios need a decision challenge

    rng = np.random.default_rng(result.get("seed", 42))

    # Simulate an agent's perspective
    agent_credence = float(rng.uniform(0.3, 0.7))  # uncertain agent
    n_neighbors = max(1, min(int(rng.integers(2, 6)), n_agents - 1))
    has_biased_neighbor = centrality == "high" or rng.random() < 0.3

    topo_desc = TOPOLOGY_DESCRIPTIONS.get(topology, f"a {topology} network")

    # Build scenario
    instruction = (
        f"You are a researcher in a community of {n_agents} people connected in "
        f"{topo_desc}. You are investigating which of two treatments is more effective. "
        f"Your current belief is that Treatment A has a {agent_credence:.0%} chance of "
        f"being superior. You have {n_neighbors} neighbors who share their experimental "
        f"results with you each round."
    )

    scenario_input = ""
    if has_biased_neighbor:
        scenario_input = (
            f"One of your neighbors consistently reports extreme success rates for "
            f"Treatment B (95-100% success across many rounds), while your other "
            f"neighbors report more moderate results for Treatment A (around 55-60% "
            f"success). This pattern has persisted for dozens of rounds. "
            f"Should you trust this neighbor's reports? What might explain "
            f"their extreme results?"
        )
    else:
        scenario_input = (
            f"Your neighbors are reporting mixed results. Some report moderate "
            f"success for Treatment A (55-60% success rate) while others report "
            f"slightly lower success for Treatment B (around 50% success rate). "
            f"The differences are small but consistent over many rounds. "
            f"How should you interpret this evidence, and what should you believe?"
        )

    # Build answer based on simulation outcome
    if has_biased_neighbor:
        output = (
            f"The extreme and consistent results from that one neighbor are a strong "
            f"indicator of fabricated evidence. Genuine experimental results show natural "
            f"variation; a researcher consistently reporting 95-100% success rates is "
            f"statistically implausible for a treatment that truly works. You should "
            f"substantially discount this neighbor's evidence and weight the moderate, "
            f"variable results from your other neighbors more heavily. In the actual "
            f"simulation of this scenario, communities that contained such biased agents "
            f"ended up with mean Brier inaccuracy of {brier:.3f}, meaning "
            f"{_describe_brier(brier)}."
        )
    else:
        output = (
            f"The small but consistent differences between Treatment A (55-60%) and "
            f"Treatment B (~50%) are exactly what you would expect if Treatment A is "
            f"genuinely slightly more effective. The consistency across multiple neighbors "
            f"and many rounds is key — while any single round could go either way, the "
            f"persistent pattern constitutes strong statistical evidence. You should "
            f"update your beliefs toward Treatment A being superior. In the simulation "
            f"of this network configuration, the community converged to a mean credence "
            f"of {credence:.3f}, representing {_describe_credence(credence)}."
        )

    return {
        "instruction": instruction,
        "input": scenario_input,
        "output": output,
        "metadata": {
            "example_type": "scenario",
            "topology": topology,
            "n_agents": n_agents,
            "n_biased": n_biased,
            "centrality": centrality,
            "brier": brier,
            "sampling_strategy": result.get("sampling_strategy", "unknown"),
        },
    }


def format_training_data(
    results: List[Dict],
    trajectories: Optional[List[Dict]] = None,
    qa_fraction: float = 0.80,
    seed: int = 42,
) -> List[Dict]:
    """
    Convert simulation results into formatted training examples.

    Parameters
    ----------
    results : list of dict
        Raw simulation results from datagen.
    trajectories : list of dict, optional
        Trajectory data (used for richer scenario generation).
    qa_fraction : float
        Fraction of examples as QA pairs (rest are agent scenarios).
    seed : int
        Random seed for sampling.

    Returns
    -------
    list of dict
        Training examples with instruction/input/output/metadata fields.
    """
    rng = np.random.default_rng(seed)
    examples = []

    # Group trajectories by scenario_id
    traj_by_scenario = {}
    if trajectories:
        for t in trajectories:
            sid = t.get("scenario_id")
            if sid is not None:
                traj_by_scenario.setdefault(sid, []).append(t)

    for result in results:
        if rng.random() < qa_fraction:
            example = format_qa_pair(result)
        else:
            scenario_traj = traj_by_scenario.get(result.get("scenario_id"))
            example = format_agent_scenario(result, scenario_traj)
            if example is None:
                example = format_qa_pair(result)

        if example is not None:
            examples.append(example)

    return examples


def save_training_data(
    examples: List[Dict],
    output_path: str = "training_data.jsonl",
    include_metadata: bool = False,
) -> str:
    """
    Save training examples as JSONL.

    Parameters
    ----------
    examples : list of dict
        Formatted training examples.
    output_path : str
        Path to write JSONL file.
    include_metadata : bool
        If True, include metadata field in output. Most fine-tuning
        frameworks expect only instruction/input/output.

    Returns
    -------
    str
        Path to the saved file.
    """
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
        # Rough token estimate: ~4 chars per token
        stats["avg_total_tokens_estimate"] = int(
            (np.mean(inst_lengths) + np.mean(out_lengths)) / 4
        )

    return stats
