"""
Generate a training dataset using the aquavect data pipeline.

This example generates 1,000 training examples (use -n flag for more),
saves raw traces and formatted JSONL, and prints dataset statistics.

Usage:
    python examples/generate_dataset.py
    python examples/generate_dataset.py -n 10000 --output my_data
"""

import argparse
import json

from aquavect.datagen import DatagenConfig, generate_dataset, save_results
from aquavect.formatting import format_training_data, save_training_data, dataset_statistics


def main():
    parser = argparse.ArgumentParser(description="Generate aquavect training data")
    parser.add_argument("-n", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", default="aquavect_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Configure
    cfg = DatagenConfig(
        n_examples=args.n,
        output_dir=args.output,
        seed=args.seed,
        n_jobs=-1,
    )

    # Generate simulations
    print("Step 1: Running simulations...")
    results, trajectories = generate_dataset(cfg, verbose=True)

    # Save raw data
    print("\nStep 2: Saving raw data...")
    paths = save_results(results, trajectories, output_dir=args.output)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Format training examples
    print("\nStep 3: Formatting training examples...")
    examples = format_training_data(results, trajectories, qa_fraction=0.80)
    training_path = f"{args.output}/training_data.jsonl"
    save_training_data(examples, training_path, include_metadata=True)
    print(f"  Saved: {training_path}")

    # Statistics
    stats = dataset_statistics(examples)
    print(f"\nDataset Statistics:")
    print(f"  Total examples:        {stats['total_examples']}")
    print(f"  QA pairs:              {stats['type_counts'].get('qa', 0)}")
    print(f"  Agent scenarios:       {stats['type_counts'].get('scenario', 0)}")
    print(f"  Avg instruction chars: {stats['avg_instruction_length']}")
    print(f"  Avg output chars:      {stats['avg_output_length']}")
    print(f"  Avg tokens (estimate): {stats['avg_total_tokens_estimate']}")
    print(f"\n  Topology distribution:")
    for topo, count in sorted(stats["topology_counts"].items()):
        print(f"    {topo:<20} {count}")
    print(f"\n  Strategy distribution:")
    for strat, count in sorted(stats["strategy_counts"].items()):
        print(f"    {strat:<20} {count}")

    # Show a sample
    print(f"\n{'='*60}")
    print("SAMPLE TRAINING EXAMPLE:")
    print(f"{'='*60}")
    sample = examples[0]
    print(f"\n[Instruction]\n{sample['instruction']}")
    if sample.get("input"):
        print(f"\n[Input]\n{sample['input']}")
    print(f"\n[Output]\n{sample['output']}")

    # Optional visualization
    try:
        from aquavect.viz import plot_dataset_overview
        plot_path = f"{args.output}/dataset_overview.png"
        plot_dataset_overview(stats, save_path=plot_path)
        print(f"\nOverview plot: {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
