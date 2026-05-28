"""
Reproduce the core finding from Paper 1:
High-centrality biased agents cause more epistemic damage than
low-centrality ones (the position effect, d = 1.36).

This script runs a minimal version of Phase 1 across all 8 asymmetric
topologies at a single network size, demonstrating the library's API.
"""

from aquavect import (
    create_network,
    get_high_centrality_positions,
    get_low_centrality_positions,
    run_simulation,
    cohens_d,
    ASYMMETRIC_TOPOLOGIES,
)
import numpy as np

SEEDS = 50
N_AGENTS = 15

results_high = []
results_low = []

print("Reproducing Paper 1 Position Effect")
print("=" * 50)
print(f"Topologies: {len(ASYMMETRIC_TOPOLOGIES)}")
print(f"Network size: {N_AGENTS}")
print(f"Seeds per condition: {SEEDS}")
print()

for topo in ASYMMETRIC_TOPOLOGIES:
    high_briers = []
    low_briers = []

    for seed in range(SEEDS):
        G = create_network(topo, N_AGENTS, seed=seed)

        # High-centrality biased agent
        high_pos = get_high_centrality_positions(G, 1)
        res_h, _ = run_simulation(
            G=G, topology=topo, n_agents=N_AGENTS,
            biased_positions=high_pos,
            condition_name="1_high", seed=seed,
        )
        high_briers.append(res_h["final_mean_brier"])

        # Low-centrality biased agent
        low_pos = get_low_centrality_positions(G, 1, exclude=high_pos)
        res_l, _ = run_simulation(
            G=G, topology=topo, n_agents=N_AGENTS,
            biased_positions=low_pos,
            condition_name="1_low", seed=seed,
        )
        low_briers.append(res_l["final_mean_brier"])

    results_high.extend(high_briers)
    results_low.extend(low_briers)

    d = cohens_d(low_briers, high_briers)
    print(f"  {topo:<15}  High={np.mean(high_briers):.4f}  "
          f"Low={np.mean(low_briers):.4f}  d={d:.2f}")

# Overall
d_overall = cohens_d(results_low, results_high)
print()
print(f"OVERALL:  High={np.mean(results_high):.4f}  "
      f"Low={np.mean(results_low):.4f}")
print(f"Cohen's d = {d_overall:.2f}")
print(f"(Paper 1 reported d = 1.36)")
