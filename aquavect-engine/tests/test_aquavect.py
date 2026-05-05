"""
Tests for the aquavect simulation engine.

Validates core mechanics against known properties:
  - Network topology generation
  - Bayesian updating direction
  - Biased agent intransigence
  - Aggregator computation
  - Position effect direction
"""

import numpy as np
import pytest
from aquavect import (
    Agent, AgentType,
    create_network,
    get_high_centrality_positions,
    get_low_centrality_positions,
    get_centrality_measures,
    run_simulation,
    SimulationConfig,
    compute_aggregator_credence,
    apply_aggregator_update,
    brier_score,
    cohens_d,
    ALL_TOPOLOGIES,
    ASYMMETRIC_TOPOLOGIES,
    SYMMETRIC_TOPOLOGIES,
)


# ──────────────────────────────────────────────
# Network tests
# ──────────────────────────────────────────────

class TestNetworks:
    def test_all_topologies_create(self):
        """Every topology should produce a connected graph."""
        for topo in ALL_TOPOLOGIES:
            import networkx as nx
            G = create_network(topo, 10, seed=42)
            assert len(G.nodes()) == 10
            assert nx.is_connected(G)

    def test_star_hub_has_max_degree(self):
        G = create_network("star", 10, seed=0)
        degrees = dict(G.degree())
        hub = max(degrees, key=degrees.get)
        assert hub == 0
        assert degrees[0] == 9

    def test_complete_all_equal_degree(self):
        G = create_network("complete", 8, seed=0)
        degrees = set(dict(G.degree()).values())
        assert len(degrees) == 1  # all nodes have same degree

    def test_cycle_all_degree_two(self):
        G = create_network("cycle", 10, seed=0)
        degrees = set(dict(G.degree()).values())
        assert degrees == {2}

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError):
            create_network("nonexistent", 10)

    def test_high_low_positions_disjoint(self):
        G = create_network("star", 10, seed=0)
        high = get_high_centrality_positions(G, 1)
        low = get_low_centrality_positions(G, 1, exclude=high)
        assert set(high).isdisjoint(set(low))

    def test_centrality_measures_keys(self):
        G = create_network("scale_free", 10, seed=42)
        measures = get_centrality_measures(G)
        for node in G.nodes():
            assert set(measures[node].keys()) == {
                "degree", "betweenness", "closeness", "eigenvector"
            }


# ──────────────────────────────────────────────
# Agent tests
# ──────────────────────────────────────────────

class TestAgents:
    def test_uniform_prior_credence(self):
        a = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER)
        assert a.credence == pytest.approx(0.5)

    def test_brier_score_uninformed(self):
        a = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER)
        assert a.brier_score == pytest.approx(0.25)

    def test_brier_score_perfect(self):
        a = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER, alpha=999, beta=1)
        assert a.brier_score < 0.001

    def test_brier_score_maximally_wrong(self):
        a = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER, alpha=1, beta=999)
        assert a.brier_score > 0.99

    def test_is_biased_flag(self):
        ts = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER)
        bi = Agent(agent_id=1, agent_type=AgentType.INTRANSIGENT)
        assert not ts.is_biased
        assert bi.is_biased

    def test_copy_independence(self):
        a = Agent(agent_id=0, agent_type=AgentType.TRUTH_SEEKER, alpha=5.0)
        b = a.copy()
        b.alpha = 10.0
        assert a.alpha == 5.0


# ──────────────────────────────────────────────
# Aggregation tests
# ──────────────────────────────────────────────

class TestAggregation:
    def test_mean_aggregation(self):
        agents = [
            Agent(0, AgentType.TRUTH_SEEKER, alpha=8, beta=2),  # cred=0.8
            Agent(1, AgentType.TRUTH_SEEKER, alpha=2, beta=8),  # cred=0.2
        ]
        cred = compute_aggregator_credence(agents, method="mean")
        assert cred == pytest.approx(0.5, abs=0.01)

    def test_median_aggregation(self):
        agents = [
            Agent(0, AgentType.TRUTH_SEEKER, alpha=9, beta=1),  # 0.9
            Agent(1, AgentType.TRUTH_SEEKER, alpha=8, beta=2),  # 0.8
            Agent(2, AgentType.TRUTH_SEEKER, alpha=1, beta=9),  # 0.1 outlier
        ]
        cred = compute_aggregator_credence(agents, method="median")
        assert cred == pytest.approx(0.8, abs=0.01)

    def test_excludes_intransigent(self):
        agents = [
            Agent(0, AgentType.TRUTH_SEEKER, alpha=9, beta=1),
            Agent(1, AgentType.INTRANSIGENT, alpha=1, beta=999),  # excluded
        ]
        cred = compute_aggregator_credence(agents, method="mean")
        assert cred == pytest.approx(0.9, abs=0.01)

    def test_empty_agents_returns_0_5(self):
        assert compute_aggregator_credence([], method="mean") == 0.5

    def test_aggregator_update_increases_alpha(self):
        a = Agent(0, AgentType.TRUTH_SEEKER, alpha=5, beta=5)
        rng = np.random.default_rng(0)
        old_credence = a.credence
        apply_aggregator_update(a, agg_credence=0.9, agg_weight=0.5, rng=rng)
        assert a.credence > old_credence


# ──────────────────────────────────────────────
# Simulation tests
# ──────────────────────────────────────────────

class TestSimulation:
    def test_control_converges_toward_truth(self):
        """Without bias, agents should move toward credence > 0.5."""
        G = create_network("complete", 10, seed=0)
        res, _ = run_simulation(
            G=G, topology="complete", n_agents=10,
            biased_positions=[], condition_name="control",
            seed=0, n_rounds=200, efficacy_difference=0.10,
        )
        # With delta=0.10, truth-seekers should at least move above 0.5
        assert res["final_mean_credence"] > 0.5
        # And Brier score should be below the uninformed baseline of 0.25
        assert res["final_mean_brier"] < 0.25

    def test_biased_agent_hurts(self):
        """A biased agent should increase mean Brier inaccuracy."""
        G = create_network("star", 10, seed=42)
        res_ctrl, _ = run_simulation(
            G=G, topology="star", n_agents=10,
            biased_positions=[], condition_name="control",
            seed=42, n_rounds=200,
        )
        res_biased, _ = run_simulation(
            G=G, topology="star", n_agents=10,
            biased_positions=[0], condition_name="1_high",
            seed=42, n_rounds=200,
        )
        assert res_biased["final_mean_brier"] > res_ctrl["final_mean_brier"]

    def test_position_effect_direction(self):
        """High-centrality bias should cause more damage than low-centrality."""
        briers_high = []
        briers_low = []
        for seed in range(20):
            G = create_network("star", 10, seed=seed)
            high_pos = get_high_centrality_positions(G, 1)
            low_pos = get_low_centrality_positions(G, 1, exclude=high_pos)

            res_h, _ = run_simulation(
                G=G, topology="star", n_agents=10,
                biased_positions=high_pos,
                condition_name="1_high", seed=seed,
            )
            res_l, _ = run_simulation(
                G=G, topology="star", n_agents=10,
                biased_positions=low_pos,
                condition_name="1_low", seed=seed,
            )
            briers_high.append(res_h["final_mean_brier"])
            briers_low.append(res_l["final_mean_brier"])

        assert np.mean(briers_high) > np.mean(briers_low)

    def test_trajectory_output(self):
        G = create_network("cycle", 6, seed=0)
        res, traj = run_simulation(
            G=G, topology="cycle", n_agents=6,
            biased_positions=[], condition_name="control",
            seed=0, save_trajectory=True, n_rounds=50,
        )
        assert traj is not None
        assert len(traj) == 50
        assert "mean_brier" in traj[0]
        assert "mean_credence" in traj[0]

    def test_aggregator_runs(self):
        G = create_network("star", 10, seed=0)
        res, _ = run_simulation(
            G=G, topology="star", n_agents=10,
            biased_positions=[0], condition_name="1_high",
            seed=0, enable_aggregator=True,
            agg_frequency=1.0, agg_weight=0.1,
        )
        assert res["enable_aggregator"] is True
        assert res["final_agg_credence"] is not None

    def test_config_object(self):
        cfg = SimulationConfig(n_rounds=50, bias_strength=0.8)
        G = create_network("cycle", 6, seed=0)
        res, _ = run_simulation(
            G=G, topology="cycle", n_agents=6,
            biased_positions=[0], condition_name="1_high",
            seed=0, config=cfg,
        )
        assert res["n_rounds"] == 50
        assert res["bias_strength"] == 0.8

    def test_seed_reproducibility(self):
        G = create_network("scale_free", 10, seed=42)
        res1, _ = run_simulation(
            G=G, topology="scale_free", n_agents=10,
            biased_positions=[0], condition_name="1_high", seed=99,
        )
        G2 = create_network("scale_free", 10, seed=42)
        res2, _ = run_simulation(
            G=G2, topology="scale_free", n_agents=10,
            biased_positions=[0], condition_name="1_high", seed=99,
        )
        assert res1["final_mean_brier"] == pytest.approx(
            res2["final_mean_brier"], abs=1e-10
        )


# ──────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────

class TestMetrics:
    def test_brier_score_function(self):
        assert brier_score(1.0) == 0.0
        assert brier_score(0.0) == 1.0
        assert brier_score(0.5) == pytest.approx(0.25)

    def test_cohens_d_identical(self):
        a = [1.0, 2.0, 3.0]
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_cohens_d_direction(self):
        low = [1.0, 2.0, 3.0]
        high = [4.0, 5.0, 6.0]
        assert cohens_d(low, high) > 0
        assert cohens_d(high, low) < 0
