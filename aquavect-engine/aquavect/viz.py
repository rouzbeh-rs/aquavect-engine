"""
Visualization module for Aquavect simulations and benchmarks.

Renders simulation results, network topologies, and benchmark
leaderboards using the Aquavect visual identity:

  Colors:
    - Accent green:  #4A6A2A
    - Dark text:     #2C3328
    - Background:    #FCFDFB
    - High damage:   #C0392B (red)
    - Low damage:    #27AE60 (green)
    - Control:       #3498DB (blue)
    - Aggregator:    #8E44AD (purple)

  Typography: Georgia serif for titles, sans-serif for data.

All functions return matplotlib Figure objects and optionally
save to disk. The style is consistent with the Aquavect website
(aquavect.com) for use in papers, presentations, and marketing.

Usage:
    >>> from aquavect.viz import plot_position_effect, set_aquavect_style
    >>> set_aquavect_style()
    >>> fig = plot_position_effect(results)
    >>> fig.savefig("position_effect.png", dpi=300)
"""

from typing import Dict, List, Optional, Sequence, Tuple
import os

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ──────────────────────────────────────────────
# Aquavect Brand Palette
# ──────────────────────────────────────────────

COLORS = {
    "accent":       "#4A6A2A",
    "accent_light": "#6A8A4A",
    "dark":         "#2C3328",
    "background":   "#FCFDFB",
    "bg_panel":     "#F5F7F2",
    "high_damage":  "#C0392B",
    "low_damage":   "#27AE60",
    "control":      "#3498DB",
    "aggregator":   "#8E44AD",
    "neutral":      "#7F8C8D",
    "grid":         "#E8EDE4",
    "text_muted":   "#95A5A6",
    "warm_red":     "#E74C3C",
    "warm_orange":  "#F39C12",
    "warm_gold":    "#D4AC0D",
}

# Ordered palette for multi-series plots
PALETTE = [
    COLORS["accent"],
    COLORS["high_damage"],
    COLORS["control"],
    COLORS["aggregator"],
    COLORS["warm_orange"],
    COLORS["low_damage"],
    COLORS["warm_gold"],
    COLORS["neutral"],
]


def _require_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install aquavect[full]"
        )


def set_aquavect_style():
    """
    Apply the Aquavect visual style globally to matplotlib.

    Call this once before creating any plots.
    """
    _require_mpl()
    plt.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["dark"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "text.color": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 9,
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "serif"],
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": COLORS["background"],
    })


def _add_watermark(ax, text="aquavect"):
    """Add subtle Aquavect watermark to bottom-right."""
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color=COLORS["text_muted"], alpha=0.4,
        fontstyle="italic", fontfamily="serif",
    )


def _style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent axis styling."""
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=COLORS["dark"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=COLORS["dark"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=COLORS["dark"])
    ax.tick_params(colors=COLORS["dark"], which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    _add_watermark(ax)


# ──────────────────────────────────────────────
# Simulation result plots
# ──────────────────────────────────────────────

def plot_position_effect(
    results: List[Dict],
    title: str = "Position Effect on Epistemic Damage",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Bar chart comparing high vs low centrality bias damage.

    Parameters
    ----------
    results : list of dict
        Simulation results with 'biased_centrality' and 'final_mean_brier'.
    """
    _require_mpl()
    set_aquavect_style()

    high = [r["final_mean_brier"] for r in results if r.get("biased_centrality") == "high"]
    low = [r["final_mean_brier"] for r in results if r.get("biased_centrality") == "low"]
    control = [r["final_mean_brier"] for r in results if r.get("n_biased", 0) == 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = []
    means = []
    stds = []
    colors = []

    if control:
        categories.append("Control\n(no bias)")
        means.append(np.mean(control))
        stds.append(np.std(control))
        colors.append(COLORS["control"])

    if high:
        categories.append("High\nCentrality")
        means.append(np.mean(high))
        stds.append(np.std(high))
        colors.append(COLORS["high_damage"])

    if low:
        categories.append("Low\nCentrality")
        means.append(np.mean(low))
        stds.append(np.std(low))
        colors.append(COLORS["low_damage"])

    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor=COLORS["dark"],
                  linewidth=0.8, capsize=6, width=0.6, alpha=0.85)

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLORS["dark"])

    ax.axhline(0.25, color=COLORS["neutral"], linestyle="--", alpha=0.6,
               label="Uninformed baseline (0.25)")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc="upper right")

    _style_axis(ax, title=title, ylabel="Mean Brier Inaccuracy")
    ax.set_ylim(0, max(means) * 1.3 if means else 0.5)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_topology_comparison(
    results: List[Dict],
    title: str = "Position Effect by Topology",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Horizontal bar chart showing high-low Brier difference per topology.
    """
    _require_mpl()
    set_aquavect_style()

    single = [r for r in results if r.get("n_biased", 0) == 1]
    topologies = sorted(set(r["topology"] for r in single))

    diffs = []
    topos = []
    for topo in topologies:
        high_vals = [r["final_mean_brier"] for r in single
                     if r["topology"] == topo and r.get("biased_centrality") == "high"]
        low_vals = [r["final_mean_brier"] for r in single
                    if r["topology"] == topo and r.get("biased_centrality") == "low"]
        if high_vals and low_vals:
            diffs.append(np.mean(high_vals) - np.mean(low_vals))
            topos.append(topo)

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(topos))
    bar_colors = [COLORS["high_damage"] if d > 0 else COLORS["low_damage"] for d in diffs]
    ax.barh(y, diffs, color=bar_colors, edgecolor=COLORS["dark"],
            linewidth=0.6, height=0.6, alpha=0.85)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(topos, fontsize=10)

    _style_axis(ax, title=title, xlabel="Brier Difference (High − Low)")

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_aggregator_effect(
    results: List[Dict],
    title: str = "Aggregator Effect on Epistemic Damage",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Grouped bar chart: with vs without aggregator, by centrality.
    """
    _require_mpl()
    set_aquavect_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    categories = []
    no_agg_means = []
    with_agg_means = []

    for cent in ["high", "low"]:
        no_agg = [r["final_mean_brier"] for r in results
                  if r.get("biased_centrality") == cent
                  and not r.get("enable_aggregator", False)]
        with_agg = [r["final_mean_brier"] for r in results
                    if r.get("biased_centrality") == cent
                    and r.get("enable_aggregator", False)]

        if no_agg and with_agg:
            categories.append(f"{cent.capitalize()}\nCentrality")
            no_agg_means.append(np.mean(no_agg))
            with_agg_means.append(np.mean(with_agg))

    if categories:
        x = np.arange(len(categories))
        width = 0.3
        ax.bar(x - width/2, no_agg_means, width, label="No Aggregator",
               color=COLORS["warm_red"], edgecolor=COLORS["dark"],
               linewidth=0.6, alpha=0.85)
        ax.bar(x + width/2, with_agg_means, width, label="With Aggregator",
               color=COLORS["aggregator"], edgecolor=COLORS["dark"],
               linewidth=0.6, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend()
        ax.axhline(0.25, color=COLORS["neutral"], linestyle="--", alpha=0.6)

    _style_axis(ax, title=title, ylabel="Mean Brier Inaccuracy")

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_trajectory(
    trajectories: List[Dict],
    group_by: str = "condition",
    title: str = "Belief Dynamics Over Time",
    metric: str = "mean_brier",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Line plot of simulation dynamics over rounds.

    Parameters
    ----------
    trajectories : list of dict
        Trajectory entries with 'round', metric, and group_by fields.
    group_by : str
        Field to group traces by (e.g., 'condition', 'topology').
    metric : str
        Which metric to plot ('mean_brier' or 'mean_credence').
    """
    _require_mpl()
    set_aquavect_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    groups = sorted(set(t.get(group_by, "unknown") for t in trajectories))

    for i, group in enumerate(groups):
        group_traj = [t for t in trajectories if t.get(group_by) == group]
        rounds = sorted(set(t["round"] for t in group_traj))

        means = []
        for r in rounds:
            vals = [t[metric] for t in group_traj if t["round"] == r]
            means.append(np.mean(vals))

        color = PALETTE[i % len(PALETTE)]
        ax.plot(rounds, means, color=color, linewidth=2.5, label=group, alpha=0.85)

    if metric == "mean_brier":
        ax.axhline(0.25, color=COLORS["neutral"], linestyle="--", alpha=0.5,
                   label="Uninformed (0.25)")
        ylabel = "Mean Brier Inaccuracy"
    else:
        ax.axhline(0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5,
                   label="Uninformed (0.50)")
        ylabel = "Mean Credence"

    ax.legend(loc="best")
    _style_axis(ax, title=title, xlabel="Round", ylabel=ylabel)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_network(
    G,
    biased_positions: Sequence[int] = (),
    aggregator_position: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Render a network topology in Aquavect visual style.

    Uses the website's color coding: blue for truth-seekers,
    red for biased agents, purple diamond for aggregator.
    """
    _require_mpl()
    if not HAS_NX:
        raise ImportError("networkx required for network plotting")
    set_aquavect_style()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(COLORS["background"])

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COLORS["grid"],
                           width=1.5, alpha=0.6)

    # Node colors
    node_colors = []
    node_sizes = []
    biased_set = set(biased_positions)
    for node in G.nodes():
        if aggregator_position is not None and node == aggregator_position:
            node_colors.append(COLORS["aggregator"])
            node_sizes.append(500)
        elif node in biased_set:
            node_colors.append(COLORS["high_damage"])
            node_sizes.append(400)
        else:
            node_colors.append(COLORS["control"])
            node_sizes.append(300)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors=COLORS["dark"],
                           linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold",
                            font_color="white")

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold",
                     color=COLORS["dark"], pad=15, fontfamily="serif")
    _add_watermark(ax)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_benchmark_leaderboard(
    model_scores: Dict[str, Dict],
    title: str = "Aquavect Benchmark Leaderboard",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Render a benchmark leaderboard as a styled table/chart.

    Parameters
    ----------
    model_scores : dict
        {model_name: {"overall": float, "tier1": float, "tier2": float, "tier3": float}}
    """
    _require_mpl()
    set_aquavect_style()

    models = list(model_scores.keys())
    tiers = ["overall", "tier1", "tier2", "tier3"]
    tier_labels = ["Overall", "Network\nLiteracy", "Dynamic\nPrediction", "Strategic\nReasoning"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.8 + 2)))

    x = np.arange(len(tiers))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        scores = model_scores[model]
        values = [scores.get(t, 0) for t in tiers]
        color = PALETTE[i % len(PALETTE)]
        bars = ax.bar(x + i * width - (len(models) - 1) * width / 2,
                      values, width, label=model, color=color,
                      edgecolor=COLORS["dark"], linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    _style_axis(ax, title=title, ylabel="Accuracy")

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_dataset_overview(
    stats: Dict,
    title: str = "Training Dataset Overview",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Multi-panel overview of a generated training dataset.

    Parameters
    ----------
    stats : dict
        Output from formatting.dataset_statistics().
    """
    _require_mpl()
    set_aquavect_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Example type distribution
    ax = axes[0]
    type_counts = stats.get("type_counts", {})
    if type_counts:
        labels = list(type_counts.keys())
        values = list(type_counts.values())
        type_colors = [COLORS["accent"], COLORS["aggregator"]][:len(labels)]
        ax.pie(values, labels=labels, colors=type_colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 10, "color": COLORS["dark"]})
    _style_axis(ax, title="Example Types")
    ax.grid(False)

    # Panel 2: Topology distribution
    ax = axes[1]
    topo_counts = stats.get("topology_counts", {})
    if topo_counts:
        topos = sorted(topo_counts.keys())
        counts = [topo_counts[t] for t in topos]
        y = np.arange(len(topos))
        ax.barh(y, counts, color=COLORS["accent"], edgecolor=COLORS["dark"],
                linewidth=0.5, height=0.6, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(topos, fontsize=9)
    _style_axis(ax, title="Topology Distribution", xlabel="Count")

    # Panel 3: Strategy distribution
    ax = axes[2]
    strat_counts = stats.get("strategy_counts", {})
    if strat_counts:
        labels = list(strat_counts.keys())
        values = list(strat_counts.values())
        strat_colors = [COLORS["control"], COLORS["low_damage"], COLORS["warm_orange"]][:len(labels)]
        ax.pie(values, labels=labels, colors=strat_colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 10, "color": COLORS["dark"]})
    _style_axis(ax, title="Sampling Strategy")
    ax.grid(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", color=COLORS["dark"],
                 y=1.02, fontfamily="serif")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
