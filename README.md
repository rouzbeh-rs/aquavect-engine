# aquavect-engine

**Agent-based network epistemology simulation engine.**

Aquavect provides tools for simulating Bayesian agents, strategic agents, and information aggregators in epistemic networks. It implements the Bala-Goyal / Zollman / Holman-Bruner research framework and extends it with market-like aggregation mechanisms.

## Installation

```bash
pip install -e .

# With full dependencies (pandas, matplotlib, tqdm, joblib):
pip install -e ".[full]"

# For development:
pip install -e ".[dev]"
```

## Quick Start

```python
from aquavect import create_network, run_simulation
from aquavect import get_high_centrality_positions, get_low_centrality_positions

# Create a scale-free network with 15 agents
G = create_network("scale_free", 15, seed=42)

# Place a biased agent at the hub
hub = get_high_centrality_positions(G, 1)
result, trajectory = run_simulation(
    G=G, topology="scale_free", n_agents=15,
    biased_positions=hub,
    condition_name="1_high",
    seed=42,
    save_trajectory=True,
)

print(f"Mean Brier inaccuracy: {result['final_mean_brier']:.4f}")
print(f"Mean credence: {result['final_mean_credence']:.4f}")
```

## Core Concepts

**Agents** maintain Beta(α, β) distributions representing beliefs about which of two treatments is more effective. Treatment A is genuinely better (P_A = 0.55 vs P_B = 0.50 by default). Each round, agents generate evidence, observe neighbors' results, and update beliefs via Bayesian inference.

**Biased agents** (agnotologists) never update their beliefs and fabricate evidence favoring the inferior treatment. Their structural position in the network determines how much damage they cause — the central finding of Paper 1.

**Aggregator nodes** compute a consensus signal from all honest agents and broadcast it back, modeling prediction markets. Paper 2 found this provides weak protection that does not flatten the position effect.

## Supported Topologies

| Topology | Type | Description |
|----------|------|-------------|
| `star` | Asymmetric | One hub connected to all others |
| `wheel` | Asymmetric | Star with peripheral cycle |
| `line` | Asymmetric | Sequential chain |
| `hierarchical` | Asymmetric | Binary tree |
| `clustered` | Asymmetric | Two dense groups, single bridge |
| `scale_free` | Asymmetric | Barabási-Albert preferential attachment |
| `small_world` | Asymmetric | Watts-Strogatz (p=0.3) |
| `random` | Asymmetric | Erdős-Rényi with connectivity guarantee |
| `complete` | Symmetric | All-to-all (negative control) |
| `cycle` | Symmetric | Ring (negative control) |

## Configuration

Use `SimulationConfig` for full parameter control:

```python
from aquavect import SimulationConfig, run_simulation

config = SimulationConfig(
    n_rounds=500,
    efficacy_difference=0.10,
    enable_aggregator=True,
    agg_method="median",
    agg_weight=0.2,
    truth_seeker_prod_min=50,
    truth_seeker_prod_max=200,
    save_trajectory=True,
)

result, trajectory = run_simulation(
    G=G, topology="star", n_agents=15,
    biased_positions=[0],
    condition_name="1_high",
    seed=42, config=config,
)
```

Or pass parameters directly as keyword arguments:

```python
result, _ = run_simulation(
    G=G, topology="star", n_agents=15,
    biased_positions=[0], condition_name="1_high",
    seed=42, n_rounds=500, enable_aggregator=True,
)
```

## Examples

See the `examples/` directory:

- `reproduce_paper1.py` — Reproduces the position effect (d ≈ 1.36)
- `reproduce_paper2.py` — Reproduces the aggregator protection finding

## Tests

```bash
pytest tests/ -v
```

## Research Papers

This engine implements the simulation framework from:

1. **"Does Network Position Amplify Manufactured Agnotology?"** — Establishes that high-centrality biased agents cause significantly more epistemic damage than low-centrality ones (d = 1.36), with degree centrality as the best predictor (R² = .82).

## License

MIT
