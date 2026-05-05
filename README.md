# aquavect-engine

**Agent-based network epistemology simulation engine.**

Aquavect provides tools for simulating Bayesian agents, strategic agents, and information aggregators in epistemic networks. It implements the Bala-Goyal / Zollman / Holman-Bruner research framework and extends it with market-like aggregation mechanisms, synthetic data generation for LLM fine-tuning, and an evaluation benchmark for network-structured decision reasoning.

## Installation

```bash
pip install -e .

# With full dependencies (pandas, matplotlib, tqdm, joblib, pyarrow):
pip install -e ".[full]"

# For development:
pip install -e ".[dev]"
```

## Quick Start

```python
from aquavect import create_network, run_simulation
from aquavect import get_high_centrality_positions

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
```

## Modules

| Module | Description |
|--------|-------------|
| `agents` | Agent types (TruthSeeker, Intransigent, IndustrialSelection, Aggregator) |
| `networks` | 10 topology generators + centrality utilities |
| `simulation` | Core simulation engine with `SimulationConfig` |
| `aggregation` | Mean, median, productivity-weighted aggregation |
| `metrics` | Brier score, Cohen's d, convergence detection |
| `datagen` | Synthetic data generation pipeline for LLM training |
| `formatting` | Trace-to-text conversion (QA pairs + agent scenarios) |
| `benchmark` | 3-tier evaluation benchmark for network reasoning |
| `viz` | Aquavect-branded visualization (publication-ready figures) |

## CLI

```bash
# Generate training data
python -m aquavect generate -n 10000 --output my_data

# Generate benchmark questions
python -m aquavect benchmark --generate -n 500

# Package info
python -m aquavect info
```

## Data Generation Pipeline

Generate synthetic training data for fine-tuning LLMs on network-structured decision reasoning:

```python
from aquavect.datagen import DatagenConfig, generate_dataset, save_results
from aquavect.formatting import format_training_data, save_training_data

# Configure: 10K examples with mixed sampling strategies
cfg = DatagenConfig(n_examples=10000)

# Run simulations and collect traces
results, trajectories = generate_dataset(cfg)
save_results(results, trajectories, output_dir="my_data")

# Format as training examples (80% QA, 20% agent scenarios)
examples = format_training_data(results, trajectories)
save_training_data(examples, "my_data/training_data.jsonl")
```

Three sampling strategies ensure diverse coverage:
- **Systematic** (50%): Grid over canonical parameters from Papers 1 & 2
- **Random** (30%): Randomized topologies and parameters for generalization
- **Targeted** (20%): Configurations near known phase transitions

## Visualization

Publication-ready figures in the Aquavect visual style:

```python
from aquavect.viz import set_aquavect_style, plot_position_effect

set_aquavect_style()
fig = plot_position_effect(results, save_path="position_effect.png")
```

Available plots: `plot_position_effect`, `plot_topology_comparison`, `plot_aggregator_effect`, `plot_trajectory`, `plot_network`, `plot_benchmark_leaderboard`, `plot_dataset_overview`.

## Benchmark

Three-tier evaluation for network-structured decision reasoning:

| Tier | Tests | Example |
|------|-------|---------|
| 1: Network Literacy | Structural graph reasoning | "How many edges does the hub have?" |
| 2: Dynamic Prediction | Simulation outcome prediction | "Will the community converge on truth?" |
| 3: Strategic Reasoning | Situated decision-making | "Should you trust this neighbor's reports?" |

```python
from aquavect.benchmark import BenchmarkSuite

suite = BenchmarkSuite.generate(n_questions=500)
results = suite.evaluate(model_fn=my_model)
scores = suite.compute_scores(results)
```

## Supported Topologies

| Topology | Type | Description |
|----------|------|-------------|
| `star` | Asymmetric | One hub connected to all others |
| `wheel` | Asymmetric | Star with peripheral cycle |
| `line` | Asymmetric | Sequential chain |
| `hierarchical` | Asymmetric | Binary tree |
| `clustered` | Asymmetric | Two dense groups, single bridge |
| `scale_free` | Asymmetric | Barabasi-Albert preferential attachment |
| `small_world` | Asymmetric | Watts-Strogatz (p=0.3) |
| `random` | Asymmetric | Erdos-Renyi with connectivity guarantee |
| `complete` | Symmetric | All-to-all (negative control) |
| `cycle` | Symmetric | Ring (negative control) |

## Examples

See the `examples/` directory:

- `reproduce_paper1.py` — Reproduces the position effect (d ≈ 1.36)
- `reproduce_paper2.py` — Reproduces the aggregator protection finding
- `generate_dataset.py` — Full data generation pipeline demo

## Tests

```bash
pytest tests/ -v
```

## Research Papers

1. **"Does Network Position Amplify Manufactured Agnotology?"** — High-centrality biased agents cause significantly more epistemic damage (d = 1.36), with degree centrality as the best predictor (R² = .82).

## License

MIT
