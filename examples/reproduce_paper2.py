"""
Reproduce the core finding from Paper 2:
The aggregator provides weak but significant protection against
manufactured ignorance, and does not flatten the position effect.
"""

from aquavect import (
    create_network,
    get_high_centrality_positions,
    get_low_centrality_positions,
    run_simulation,
    SimulationConfig,
    cohens_d,
    ASYMMETRIC_TOPOLOGIES,
)
import numpy as np

SEEDS = 50
N_AGENTS = 15

# Config for Paper 2: heterogeneous productivity
agg_config = SimulationConfig(
    truth_seeker_prod_min=50,
    truth_seeker_prod_max=200,
    enable_aggregator=True,
    agg_frequency=1.0,
    agg_weight=0.1,
    agg_method="mean",
)

no_agg_config = SimulationConfig(
    truth_seeker_prod_min=50,
    truth_seeker_prod_max=200,
    enable_aggregator=False,
)

print("Reproducing Paper 2 Aggregator Effect")
print("=" * 50)

for centrality in ["high", "low"]:
    briers_no_agg = []
    briers_with_agg = []

    for topo in ASYMMETRIC_TOPOLOGIES:
        for seed in range(SEEDS):
            G = create_network(topo, N_AGENTS, seed=seed)

            if centrality == "high":
                pos = get_high_centrality_positions(G, 1)
            else:
                high_pos = get_high_centrality_positions(G, 1)
                pos = get_low_centrality_positions(G, 1, exclude=high_pos)

            # Without aggregator
            res_no, _ = run_simulation(
                G=G, topology=topo, n_agents=N_AGENTS,
                biased_positions=pos,
                condition_name=f"1_{centrality}",
                seed=seed, config=no_agg_config,
            )
            briers_no_agg.append(res_no["final_mean_brier"])

            # With aggregator
            res_agg, _ = run_simulation(
                G=G, topology=topo, n_agents=N_AGENTS,
                biased_positions=pos,
                condition_name=f"1_{centrality}",
                seed=seed, config=agg_config,
            )
            briers_with_agg.append(res_agg["final_mean_brier"])

    d = cohens_d(briers_no_agg, briers_with_agg)
    direction = "PROTECTS" if np.mean(briers_with_agg) < np.mean(briers_no_agg) else "AMPLIFIES"
    print(f"\n  {centrality.upper()} centrality:")
    print(f"    No aggregator:   Brier = {np.mean(briers_no_agg):.4f}")
    print(f"    With aggregator: Brier = {np.mean(briers_with_agg):.4f}")
    print(f"    d = {d:.2f} -> {direction}")
