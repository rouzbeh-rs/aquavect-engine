"""
Synthetic data generation pipeline for LLM fine-tuning.

Generates diverse simulation configurations, runs them in parallel,
and stores both summary results and round-by-round traces for
downstream formatting into training examples.

Three sampling strategies:
  - systematic: grid over canonical parameter space (paper configurations)
  - random: randomized topologies and parameters (generalization)
  - targeted: configurations near known phase transitions (hard cases)

Usage:
    >>> from aquavect.datagen import generate_dataset, DatagenConfig
    >>> config = DatagenConfig(n_examples=1000, strategy="mixed")
    >>> traces, results = generate_dataset(config)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
import json
import os

import numpy as np

from aquavect.agents import AgentType
from aquavect.networks import (
    create_network,
    get_high_centrality_positions,
    get_low_centrality_positions,
    ASYMMETRIC_TOPOLOGIES,
    SYMMETRIC_TOPOLOGIES,
    ALL_TOPOLOGIES,
)
from aquavect.simulation import run_simulation, SimulationConfig


@dataclass
class DatagenConfig:
    """Configuration for synthetic data generation."""

    # Total target
    n_examples: int = 10000

    # Strategy mix (fractions, must sum to 1.0)
    systematic_fraction: float = 0.50
    random_fraction: float = 0.30
    targeted_fraction: float = 0.20

    # Systematic sweep parameters
    systematic_topologies: Sequence[str] = tuple(ASYMMETRIC_TOPOLOGIES)
    systematic_sizes: Sequence[int] = (6, 10, 15, 20, 30)
    systematic_bias_strengths: Sequence[float] = (0.6, 0.8, 1.0)
    systematic_agg_methods: Sequence[str] = ("mean", "median", "productivity_weighted")
    systematic_agg_weights: Sequence[float] = (0.0, 0.1, 0.5)

    # Random generation parameters
    random_size_range: Tuple[int, int] = (5, 40)
    random_ba_m_range: Tuple[int, int] = (1, 4)
    random_ws_p_range: Tuple[float, float] = (0.05, 0.5)
    random_er_density_range: Tuple[float, float] = (0.1, 0.5)

    # Trace control
    save_trajectories: bool = True
    trajectory_sample_rate: float = 0.3  # fraction of sims that save full traces

    # Parallelism
    n_jobs: int = -1

    # Output
    output_dir: str = "aquavect_data"
    seed: int = 42


@dataclass
class SimScenario:
    """A fully specified simulation scenario ready to execute."""
    topology: str
    n_agents: int
    biased_positions: List[int]
    bias_type: str
    condition_name: str
    config: SimulationConfig
    seed: int
    save_trajectory: bool
    sampling_strategy: str  # "systematic", "random", or "targeted"
    scenario_id: int = 0

    # Network generation params (for non-standard topologies)
    network_params: Optional[Dict] = None


def _generate_systematic_scenarios(
    cfg: DatagenConfig, n_target: int, base_seed: int
) -> List[SimScenario]:
    """Generate scenarios from a grid over the canonical parameter space."""
    scenarios = []
    rng = np.random.default_rng(base_seed)
    scenario_id = 0

    for topo in cfg.systematic_topologies:
        for n_agents in cfg.systematic_sizes:
            for bias_strength in cfg.systematic_bias_strengths:
                for centrality in ["high", "low", "none"]:
                    # Without aggregator
                    seed = int(rng.integers(0, 100000))
                    sim_cfg = SimulationConfig(
                        bias_strength=bias_strength,
                        save_trajectory=rng.random() < cfg.trajectory_sample_rate,
                    )
                    scenarios.append(SimScenario(
                        topology=topo, n_agents=n_agents,
                        biased_positions=[],  # filled at execution time
                        bias_type="intransigent",
                        condition_name=f"sys_{centrality}_bs{bias_strength}_noagg",
                        config=sim_cfg, seed=seed,
                        save_trajectory=sim_cfg.save_trajectory,
                        sampling_strategy="systematic",
                        scenario_id=scenario_id,
                    ))
                    scenario_id += 1

                    # With aggregator (skip for no-bias control)
                    if centrality != "none":
                        for agg_method in cfg.systematic_agg_methods:
                            for agg_weight in cfg.systematic_agg_weights:
                                if agg_weight == 0.0:
                                    continue
                                seed = int(rng.integers(0, 100000))
                                sim_cfg = SimulationConfig(
                                    bias_strength=bias_strength,
                                    enable_aggregator=True,
                                    agg_method=agg_method,
                                    agg_weight=agg_weight,
                                    agg_frequency=1.0,
                                    truth_seeker_prod_min=50,
                                    truth_seeker_prod_max=200,
                                    save_trajectory=rng.random() < cfg.trajectory_sample_rate,
                                )
                                scenarios.append(SimScenario(
                                    topology=topo, n_agents=n_agents,
                                    biased_positions=[],
                                    bias_type="intransigent",
                                    condition_name=f"sys_{centrality}_bs{bias_strength}_{agg_method}_w{agg_weight}",
                                    config=sim_cfg, seed=seed,
                                    save_trajectory=sim_cfg.save_trajectory,
                                    sampling_strategy="systematic",
                                    scenario_id=scenario_id,
                                ))
                                scenario_id += 1

    # Subsample to target
    if len(scenarios) > n_target:
        indices = rng.choice(len(scenarios), size=n_target, replace=False)
        scenarios = [scenarios[i] for i in sorted(indices)]
    elif len(scenarios) < n_target:
        # Repeat with different seeds
        extra_needed = n_target - len(scenarios)
        extra = []
        for i in range(extra_needed):
            base = scenarios[i % len(scenarios)]
            new_scenario = SimScenario(
                topology=base.topology, n_agents=base.n_agents,
                biased_positions=base.biased_positions,
                bias_type=base.bias_type,
                condition_name=base.condition_name + f"_rep{i}",
                config=base.config,
                seed=int(rng.integers(0, 100000)),
                save_trajectory=rng.random() < cfg.trajectory_sample_rate,
                sampling_strategy="systematic",
                scenario_id=scenario_id + i,
            )
            extra.append(new_scenario)
        scenarios.extend(extra)

    return scenarios[:n_target]


def _generate_random_scenarios(
    cfg: DatagenConfig, n_target: int, base_seed: int
) -> List[SimScenario]:
    """Generate scenarios with randomized topologies and parameters."""
    scenarios = []
    rng = np.random.default_rng(base_seed)

    # Random topology generators
    random_topo_types = [
        "random_er",       # Erdos-Renyi with random density
        "random_ba",       # Barabasi-Albert with random m
        "random_ws",       # Watts-Strogatz with random p
        "random_named",    # One of the 10 standard topologies
    ]

    for i in range(n_target):
        topo_type = rng.choice(random_topo_types)
        n_agents = int(rng.integers(cfg.random_size_range[0], cfg.random_size_range[1] + 1))

        if topo_type == "random_er":
            topology = "random"
            network_params = {
                "density": float(rng.uniform(*cfg.random_er_density_range))
            }
        elif topo_type == "random_ba":
            topology = "scale_free"
            network_params = {
                "m": int(rng.integers(*cfg.random_ba_m_range))
            }
        elif topo_type == "random_ws":
            topology = "small_world"
            network_params = {
                "p": float(rng.uniform(*cfg.random_ws_p_range))
            }
        else:
            topology = rng.choice(ALL_TOPOLOGIES)
            network_params = None

        # Random parameters
        bias_strength = float(rng.uniform(0.5, 1.0))
        n_biased = int(rng.choice([0, 1, 1, 1, 2, 3]))  # weighted toward 1
        centrality = rng.choice(["high", "low"]) if n_biased > 0 else "none"

        # Random aggregator
        enable_agg = bool(rng.random() < 0.4)
        agg_method = rng.choice(["mean", "median", "productivity_weighted"])
        agg_weight = float(rng.uniform(0.01, 0.8))
        agg_frequency = float(rng.uniform(0.1, 1.0))

        # Random efficacy
        efficacy = float(rng.choice([0.01, 0.02, 0.05, 0.10, 0.20]))

        # Random rounds
        n_rounds = int(rng.choice([50, 100, 200, 500]))

        sim_cfg = SimulationConfig(
            n_rounds=n_rounds,
            efficacy_difference=efficacy,
            bias_strength=bias_strength,
            enable_aggregator=enable_agg,
            agg_method=agg_method if enable_agg else "mean",
            agg_weight=agg_weight if enable_agg else 0.0,
            agg_frequency=agg_frequency if enable_agg else 0.0,
            truth_seeker_prod_min=50 if enable_agg else 100,
            truth_seeker_prod_max=200 if enable_agg else 100,
            save_trajectory=rng.random() < cfg.trajectory_sample_rate,
        )

        condition_name = f"rnd_{topology}_{centrality}_n{n_agents}"
        scenarios.append(SimScenario(
            topology=topology, n_agents=n_agents,
            biased_positions=[],
            bias_type="intransigent",
            condition_name=condition_name,
            config=sim_cfg,
            seed=int(rng.integers(0, 100000)),
            save_trajectory=sim_cfg.save_trajectory,
            sampling_strategy="random",
            scenario_id=i,
            network_params=network_params,
        ))

    return scenarios


def _generate_targeted_scenarios(
    cfg: DatagenConfig, n_target: int, base_seed: int
) -> List[SimScenario]:
    """
    Generate scenarios near known phase transitions and interesting regimes.

    These are the most informative training examples because they sit at
    boundaries where small parameter changes produce qualitatively different
    outcomes. Derived from Papers 1 and 2 findings.
    """
    scenarios = []
    rng = np.random.default_rng(base_seed)

    # Regime 1: Aggregator help-vs-hurt boundary
    # Paper 2 found the aggregator transitions from protective to harmful
    # around weight 0.2-0.5 for low-centrality bias
    for i in range(n_target // 4):
        topo = rng.choice(ASYMMETRIC_TOPOLOGIES)
        n_agents = int(rng.choice([10, 15, 20]))
        weight = float(rng.uniform(0.15, 0.6))  # near transition
        method = rng.choice(["mean", "median"])
        sim_cfg = SimulationConfig(
            enable_aggregator=True,
            agg_weight=weight,
            agg_method=method,
            agg_frequency=1.0,
            truth_seeker_prod_min=50,
            truth_seeker_prod_max=200,
            save_trajectory=rng.random() < cfg.trajectory_sample_rate,
        )
        scenarios.append(SimScenario(
            topology=topo, n_agents=n_agents,
            biased_positions=[],
            bias_type="intransigent",
            condition_name=f"tgt_agg_boundary_{method}_w{weight:.2f}",
            config=sim_cfg,
            seed=int(rng.integers(0, 100000)),
            save_trajectory=sim_cfg.save_trajectory,
            sampling_strategy="targeted",
            scenario_id=i,
        ))

    # Regime 2: Centrality equivalence boundary
    # Paper 1 found 1 high-centrality ≈ 4 low-centrality agents
    for i in range(n_target // 4):
        topo = rng.choice(["star", "wheel", "scale_free"])
        n_agents = int(rng.choice([15, 20, 30]))
        n_biased = int(rng.choice([2, 3, 4, 5]))  # near equivalence
        sim_cfg = SimulationConfig(
            save_trajectory=rng.random() < cfg.trajectory_sample_rate,
        )
        scenarios.append(SimScenario(
            topology=topo, n_agents=n_agents,
            biased_positions=[],
            bias_type="intransigent",
            condition_name=f"tgt_equiv_{n_biased}low",
            config=sim_cfg,
            seed=int(rng.integers(0, 100000)),
            save_trajectory=sim_cfg.save_trajectory,
            sampling_strategy="targeted",
            scenario_id=n_target // 4 + i,
        ))

    # Regime 3: Weak bias where position might not matter
    for i in range(n_target // 4):
        topo = rng.choice(ASYMMETRIC_TOPOLOGIES)
        n_agents = int(rng.choice([10, 15, 20]))
        bias_strength = float(rng.uniform(0.52, 0.65))
        centrality = rng.choice(["high", "low"])
        sim_cfg = SimulationConfig(
            bias_strength=bias_strength,
            save_trajectory=rng.random() < cfg.trajectory_sample_rate,
        )
        scenarios.append(SimScenario(
            topology=topo, n_agents=n_agents,
            biased_positions=[],
            bias_type="intransigent",
            condition_name=f"tgt_weakbias_{centrality}_bs{bias_strength:.2f}",
            config=sim_cfg,
            seed=int(rng.integers(0, 100000)),
            save_trajectory=sim_cfg.save_trajectory,
            sampling_strategy="targeted",
            scenario_id=2 * (n_target // 4) + i,
        ))

    # Regime 4: High efficacy where agents should overcome bias
    for i in range(n_target - 3 * (n_target // 4)):
        topo = rng.choice(ASYMMETRIC_TOPOLOGIES)
        n_agents = int(rng.choice([10, 15, 20]))
        efficacy = float(rng.uniform(0.10, 0.25))
        centrality = rng.choice(["high", "low"])
        sim_cfg = SimulationConfig(
            efficacy_difference=efficacy,
            save_trajectory=rng.random() < cfg.trajectory_sample_rate,
        )
        scenarios.append(SimScenario(
            topology=topo, n_agents=n_agents,
            biased_positions=[],
            bias_type="intransigent",
            condition_name=f"tgt_highefficacy_{centrality}_d{efficacy:.2f}",
            config=sim_cfg,
            seed=int(rng.integers(0, 100000)),
            save_trajectory=sim_cfg.save_trajectory,
            sampling_strategy="targeted",
            scenario_id=3 * (n_target // 4) + i,
        ))

    return scenarios


def _resolve_biased_positions(scenario: SimScenario, G) -> List[int]:
    """Resolve biased agent positions for a scenario given a built graph."""
    cn = scenario.condition_name.lower()

    if "none" in cn or scenario.config.bias_strength == 0:
        return []

    # Determine count
    n_biased = 1
    for prefix in ["2low", "3low", "4low", "5low"]:
        if prefix in cn.replace("_", ""):
            n_biased = int(prefix[0])
            break

    if "high" in cn:
        return get_high_centrality_positions(G, n_biased)
    elif "low" in cn:
        high_pos = get_high_centrality_positions(G, 1)
        return get_low_centrality_positions(G, n_biased, exclude=high_pos)
    else:
        # Random position
        rng = np.random.default_rng(scenario.seed)
        return list(rng.choice(G.number_of_nodes(), size=min(n_biased, G.number_of_nodes()), replace=False))


def execute_scenario(scenario: SimScenario) -> Tuple[Dict, Optional[List[Dict]], Dict]:
    """
    Execute a single simulation scenario. Returns (result, trajectory, metadata).

    This is the worker function called in parallel.
    """
    # Build network
    G = create_network(scenario.topology, scenario.n_agents, seed=scenario.seed)

    # Resolve positions
    biased_pos = _resolve_biased_positions(scenario, G)

    # Override trajectory setting
    scenario.config.save_trajectory = scenario.save_trajectory

    # Run simulation
    result, trajectory = run_simulation(
        G=G,
        topology=scenario.topology,
        n_agents=scenario.n_agents,
        biased_positions=biased_pos,
        bias_type=scenario.bias_type,
        condition_name=scenario.condition_name,
        phase=f"datagen_{scenario.sampling_strategy}",
        seed=scenario.seed,
        config=scenario.config,
    )

    # Build metadata
    metadata = {
        "scenario_id": scenario.scenario_id,
        "sampling_strategy": scenario.sampling_strategy,
        "network_params": scenario.network_params,
        "biased_positions_resolved": biased_pos,
    }

    return result, trajectory, metadata


def generate_scenarios(cfg: DatagenConfig) -> List[SimScenario]:
    """
    Generate all simulation scenarios according to the config.

    Returns a list of SimScenario objects ready for execution.
    """
    rng = np.random.default_rng(cfg.seed)

    n_systematic = int(cfg.n_examples * cfg.systematic_fraction)
    n_random = int(cfg.n_examples * cfg.random_fraction)
    n_targeted = cfg.n_examples - n_systematic - n_random

    scenarios = []
    scenarios.extend(_generate_systematic_scenarios(
        cfg, n_systematic, base_seed=int(rng.integers(0, 100000))
    ))
    scenarios.extend(_generate_random_scenarios(
        cfg, n_random, base_seed=int(rng.integers(0, 100000))
    ))
    scenarios.extend(_generate_targeted_scenarios(
        cfg, n_targeted, base_seed=int(rng.integers(0, 100000))
    ))

    # Reassign sequential IDs
    for i, s in enumerate(scenarios):
        s.scenario_id = i

    return scenarios


def generate_dataset(
    cfg: Optional[DatagenConfig] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Full pipeline: generate scenarios, execute simulations, return results.

    Parameters
    ----------
    cfg : DatagenConfig, optional
        Generation configuration. Uses defaults if None.
    verbose : bool
        Print progress information.

    Returns
    -------
    all_results : list of dict
        Summary results for every simulation.
    all_trajectories : list of dict
        Trajectory entries (round-by-round data) for simulations
        where save_trajectory was True. Each entry includes the
        scenario_id for joining back to results.
    """
    if cfg is None:
        cfg = DatagenConfig()

    if verbose:
        print(f"Aquavect Data Generation Pipeline v{__import__('aquavect').__version__}")
        print(f"  Target examples: {cfg.n_examples}")
        print(f"  Strategy mix: {cfg.systematic_fraction:.0%} systematic, "
              f"{cfg.random_fraction:.0%} random, "
              f"{cfg.targeted_fraction:.0%} targeted")

    # Generate scenarios
    scenarios = generate_scenarios(cfg)
    if verbose:
        print(f"  Generated {len(scenarios)} scenarios")

    # Execute
    all_results = []
    all_trajectories = []

    try:
        from joblib import Parallel, delayed
        from tqdm import tqdm

        if verbose:
            print(f"  Running simulations (n_jobs={cfg.n_jobs})...")

        outputs = Parallel(n_jobs=cfg.n_jobs)(
            delayed(execute_scenario)(s)
            for s in tqdm(scenarios, desc="Simulations", disable=not verbose)
        )

        for result, trajectory, metadata in outputs:
            result.update(metadata)
            all_results.append(result)
            if trajectory:
                for entry in trajectory:
                    entry["scenario_id"] = metadata["scenario_id"]
                    all_trajectories.append(entry)

    except ImportError:
        if verbose:
            print("  Running simulations sequentially (install joblib+tqdm for parallel)...")

        for i, scenario in enumerate(scenarios):
            result, trajectory, metadata = execute_scenario(scenario)
            result.update(metadata)
            all_results.append(result)
            if trajectory:
                for entry in trajectory:
                    entry["scenario_id"] = metadata["scenario_id"]
                    all_trajectories.append(entry)

            if verbose and (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(scenarios)} complete")

    if verbose:
        n_with_traj = sum(1 for r in all_results if r.get("save_trajectory"))
        print(f"  Complete: {len(all_results)} results, "
              f"{len(all_trajectories)} trajectory entries "
              f"from {n_with_traj} traced simulations")

    return all_results, all_trajectories


def save_results(
    results: List[Dict],
    trajectories: List[Dict],
    output_dir: str = "aquavect_data",
) -> Dict[str, str]:
    """
    Save raw results and trajectories to disk.

    Returns dict mapping output type to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save results as JSON Lines
    results_path = os.path.join(output_dir, "simulation_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            clean = {k: v for k, v in r.items() if v is not None}
            f.write(json.dumps(clean, cls=_NumpyEncoder) + "\n")
    paths["results"] = results_path

    # Save trajectories as JSON Lines
    if trajectories:
        traj_path = os.path.join(output_dir, "trajectories.jsonl")
        with open(traj_path, "w") as f:
            for t in trajectories:
                clean = {k: v for k, v in t.items() if v is not None}
                f.write(json.dumps(clean, cls=_NumpyEncoder) + "\n")
        paths["trajectories"] = traj_path

    # Try Parquet if available
    try:
        import pandas as pd
        results_pq = os.path.join(output_dir, "simulation_results.parquet")
        pd.DataFrame(results).to_parquet(results_pq, index=False)
        paths["results_parquet"] = results_pq

        if trajectories:
            traj_pq = os.path.join(output_dir, "trajectories.parquet")
            pd.DataFrame(trajectories).to_parquet(traj_pq, index=False)
            paths["trajectories_parquet"] = traj_pq
    except ImportError:
        pass

    # Save generation report
    report = {
        "n_results": len(results),
        "n_trajectory_entries": len(trajectories),
        "strategy_counts": {},
        "topology_counts": {},
        "bias_type_counts": {},
    }
    for r in results:
        strat = r.get("sampling_strategy", "unknown")
        report["strategy_counts"][strat] = report["strategy_counts"].get(strat, 0) + 1
        topo = r.get("topology", "unknown")
        report["topology_counts"][topo] = report["topology_counts"].get(topo, 0) + 1
        bt = r.get("bias_type", "unknown")
        report["bias_type_counts"][bt] = report["bias_type_counts"].get(bt, 0) + 1

    report_path = os.path.join(output_dir, "datagen_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    paths["report"] = report_path

    return paths
