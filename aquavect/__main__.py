"""
CLI entry point for aquavect.

Run with: python -m aquavect <command>

Commands:
    generate    Generate synthetic training data
    benchmark   Run benchmark evaluation (Phase 3)
    info        Print package information
"""

import argparse
import sys
import json


def cmd_generate(args):
    """Generate synthetic training data."""
    from aquavect.datagen import DatagenConfig, generate_dataset, save_results
    from aquavect.formatting import format_training_data, save_training_data, dataset_statistics

    cfg = DatagenConfig(
        n_examples=args.n,
        output_dir=args.output,
        seed=args.seed,
        n_jobs=args.jobs,
    )

    print(f"\n{'='*60}")
    print(f"Aquavect Synthetic Data Generation")
    print(f"{'='*60}\n")

    # Generate raw simulation data
    results, trajectories = generate_dataset(cfg, verbose=True)

    # Save raw data
    print(f"\nSaving raw data...")
    paths = save_results(results, trajectories, output_dir=args.output)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Format training examples
    print(f"\nFormatting training examples...")
    examples = format_training_data(results, trajectories)

    # Save training data
    training_path = f"{args.output}/training_data.jsonl"
    save_training_data(examples, training_path, include_metadata=True)
    print(f"  training_data: {training_path}")

    # Statistics
    stats = dataset_statistics(examples)
    stats_path = f"{args.output}/dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats: {stats_path}")

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Total examples:        {stats['total_examples']}")
    print(f"  Example types:         {stats['type_counts']}")
    print(f"  Avg tokens (estimate): {stats['avg_total_tokens_estimate']}")
    print(f"  Output directory:      {args.output}")
    print(f"{'='*60}\n")

    # Optional: generate overview plot
    if not args.no_plot:
        try:
            from aquavect.viz import plot_dataset_overview
            plot_path = f"{args.output}/dataset_overview.png"
            plot_dataset_overview(stats, save_path=plot_path)
            print(f"  Overview plot: {plot_path}")
        except ImportError:
            pass


def cmd_benchmark(args):
    """Run benchmark evaluation."""
    from aquavect.benchmark import BenchmarkSuite

    print(f"\n{'='*60}")
    print(f"Aquavect Benchmark")
    print(f"{'='*60}\n")

    if args.generate:
        suite = BenchmarkSuite.generate(n_questions=args.n, seed=args.seed)
        suite.save(args.output)
        print(f"Generated {len(suite)} questions -> {args.output}")
        print(f"Tier distribution: {suite.tier_counts}")
    else:
        print("Benchmark evaluation requires a model function.")
        print("Use the Python API: suite.evaluate(model_fn)")
        print("Or generate questions: python -m aquavect benchmark --generate")


def cmd_info(args):
    """Print package information."""
    import aquavect
    print(f"\nAquavect v{aquavect.__version__}")
    print(f"Agent-based network epistemology simulation engine")
    print(f"\nModules:")
    print(f"  agents       Agent types and data model")
    print(f"  networks     Topology generators (10 topologies)")
    print(f"  simulation   Core simulation engine")
    print(f"  aggregation  Information aggregation (mean/median/weighted)")
    print(f"  metrics      Brier score, Cohen's d, convergence")
    print(f"  datagen      Synthetic data generation pipeline")
    print(f"  formatting   Trace-to-text for LLM training")
    print(f"  benchmark    Evaluation benchmark (3 tiers)")
    print(f"  viz          Aquavect-branded visualization")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="aquavect",
        description="Aquavect: Agent-based network epistemology engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen.add_argument("-n", type=int, default=1000, help="Number of examples (default: 1000)")
    gen.add_argument("--output", default="aquavect_data", help="Output directory")
    gen.add_argument("--seed", type=int, default=42, help="Random seed")
    gen.add_argument("--jobs", type=int, default=-1, help="Parallel jobs (-1=all cores)")
    gen.add_argument("--no-plot", action="store_true", help="Skip overview plot")
    gen.set_defaults(func=cmd_generate)

    # benchmark
    bench = subparsers.add_parser("benchmark", help="Run benchmark evaluation")
    bench.add_argument("--generate", action="store_true", help="Generate benchmark questions")
    bench.add_argument("-n", type=int, default=500, help="Number of questions")
    bench.add_argument("--output", default="benchmark.json", help="Output file")
    bench.add_argument("--seed", type=int, default=42, help="Random seed")
    bench.set_defaults(func=cmd_benchmark)

    # info
    info = subparsers.add_parser("info", help="Print package information")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
