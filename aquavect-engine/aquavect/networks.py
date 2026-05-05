"""
Network topology generators and centrality utilities.

Implements all 10 topologies from the Aquavect research lineage:
  - 8 asymmetric: star, wheel, line, hierarchical, clustered,
    scale_free, small_world, random
  - 2 symmetric (negative controls): complete, cycle

All generators return undirected networkx.Graph objects with integer
node labels from 0 to n-1, matching the convention in Bala & Goyal (1998).
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Sequence


ASYMMETRIC_TOPOLOGIES = [
    "star", "wheel", "line", "hierarchical",
    "clustered", "scale_free", "small_world", "random",
]

SYMMETRIC_TOPOLOGIES = ["complete", "cycle"]

ALL_TOPOLOGIES = ASYMMETRIC_TOPOLOGIES + SYMMETRIC_TOPOLOGIES


def create_network(topology: str, n: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Create a network with the given topology and number of nodes.

    Parameters
    ----------
    topology : str
        One of the 10 supported topologies (see ALL_TOPOLOGIES).
    n : int
        Number of nodes (agents).
    seed : int, optional
        Random seed for reproducible stochastic topologies
        (scale_free, small_world, random).

    Returns
    -------
    networkx.Graph
        Undirected graph with nodes 0..n-1.

    Raises
    ------
    ValueError
        If topology is not recognized.
    """
    if seed is not None:
        np.random.seed(seed)

    if topology == "star":
        return nx.star_graph(n - 1)
    elif topology == "wheel":
        return nx.wheel_graph(n)
    elif topology == "cycle":
        return nx.cycle_graph(n)
    elif topology == "complete":
        return nx.complete_graph(n)
    elif topology == "line":
        return nx.path_graph(n)
    elif topology == "hierarchical":
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(1, n):
            G.add_edge((i - 1) // 2, i)
        return G
    elif topology == "clustered":
        G = nx.Graph()
        G.add_nodes_from(range(n))
        half = n // 2
        for i in range(half):
            for j in range(i + 1, half):
                G.add_edge(i, j)
        for i in range(half, n):
            for j in range(i + 1, n):
                G.add_edge(i, j)
        G.add_edge(half - 1, half)
        return G
    elif topology == "scale_free":
        return nx.barabasi_albert_graph(n, m=min(2, n - 1), seed=seed)
    elif topology == "small_world":
        k = min(4, n - 1)
        if k % 2 == 1:
            k = max(2, k - 1)
        return nx.watts_strogatz_graph(n, k, p=0.3, seed=seed)
    elif topology == "random":
        G = nx.erdos_renyi_graph(n, 2 * np.log(n) / n, seed=seed)
        # Ensure connectivity
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(list(comps[i])[0], list(comps[i + 1])[0])
        return G
    else:
        raise ValueError(
            f"Unknown topology: {topology!r}. "
            f"Choose from: {ALL_TOPOLOGIES}"
        )


def get_centrality_measures(G: nx.Graph) -> Dict[int, Dict[str, float]]:
    """
    Compute four centrality measures for every node.

    Returns a dict mapping node_id -> {degree, betweenness, closeness, eigenvector}.
    All values are normalized to [0, 1].
    """
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        eigenvector = {n: 0.0 for n in G.nodes()}

    return {
        n: {
            "degree": degree[n],
            "betweenness": betweenness[n],
            "closeness": closeness[n],
            "eigenvector": eigenvector[n],
        }
        for n in G.nodes()
    }


def get_network_properties(G: nx.Graph) -> Dict[str, float]:
    """Compute global network statistics: density, avg_clustering, diameter."""
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = -1
    return {
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "diameter": diameter,
    }


def get_high_centrality_positions(
    G: nx.Graph, n_positions: int = 1
) -> List[int]:
    """
    Return the n_positions nodes with highest degree centrality.

    In asymmetric topologies, these are the structural hubs.
    """
    degree = nx.degree_centrality(G)
    return sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[
        :n_positions
    ]


def get_low_centrality_positions(
    G: nx.Graph,
    n_positions: int = 1,
    exclude: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Return the n_positions nodes with lowest degree centrality,
    optionally excluding specified nodes (e.g., already-assigned hub positions).
    """
    degree = nx.degree_centrality(G)
    exclude_set = set(exclude) if exclude else set()
    candidates = [n for n in G.nodes() if n not in exclude_set]
    return sorted(candidates, key=lambda x: degree[x])[:n_positions]
