"""
Agent types for network epistemology simulations.

Implements the Bala-Goyal (1998) / Holman-Bruner (2015) agent framework:
  - Truth-seeking agents that update beliefs via Bayesian inference
  - Intransigently biased agents that fabricate evidence
  - Industrial selection agents (Holman-Bruner 2017 / Moran process)
  - Aggregator nodes that compute consensus signals (Paper 2)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class AgentType(Enum):
    """Agent role in the epistemic network."""
    TRUTH_SEEKER = "truth_seeker"
    INTRANSIGENT = "intransigent"
    INDUSTRIAL_SELECTION = "industrial_selection"
    AGGREGATOR = "aggregator"


@dataclass
class Agent:
    """
    A single agent in the epistemic network.

    Maintains a Beta(alpha, beta) distribution representing credences about
    the comparative efficacy of Treatment A. The credence (alpha / (alpha + beta))
    represents the agent's current belief that Treatment A is superior.

    Parameters
    ----------
    agent_id : int
        Unique identifier (matches node index in the network graph).
    agent_type : AgentType
        Role in the simulation.
    alpha : float
        Alpha parameter of the Beta distribution (default: 1.0 = uniform prior).
    beta : float
        Beta parameter of the Beta distribution (default: 1.0 = uniform prior).
    personal_p_A : float
        Agent's personal success probability for Treatment A. Equals the true
        probability for truth-seekers under intransigent bias; may differ under
        industrial selection due to methodological bias.
    personal_p_B : float
        Agent's personal success probability for Treatment B.
    base_productivity : int
        Number of trials the agent runs per round before funding boosts.
    funding_boost : int
        Additional trials per round from industry funding (IS mechanism).
    is_funded : bool
        Whether the agent currently receives industry funding.
    bias_strength : float
        For intransigent agents: the fabricated success rate reported for
        Treatment B (0.5 = truthful, 1.0 = always reports success).
    """
    agent_id: int
    agent_type: AgentType
    alpha: float = 1.0
    beta: float = 1.0
    personal_p_A: float = 0.55
    personal_p_B: float = 0.50
    base_productivity: int = 100
    funding_boost: int = 0
    is_funded: bool = False
    bias_strength: float = 1.0

    @property
    def credence(self) -> float:
        """Current belief that Treatment A is superior (0 to 1)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def brier_score(self) -> float:
        """
        Brier inaccuracy score: (1 - credence)^2.

        Measures epistemic inaccuracy relative to the truth that Treatment A
        is genuinely superior (truth value = 1). A perfectly informed agent
        has Brier score 0; a maximally wrong agent has Brier score 1;
        an uninformed agent (credence = 0.5) has Brier score 0.25.

        References: Brier (1950), Zollman (2021).
        """
        return (1.0 - self.credence) ** 2

    @property
    def effective_productivity(self) -> int:
        """Total trials per round (base + funding boost)."""
        return self.base_productivity + self.funding_boost

    @property
    def is_biased(self) -> bool:
        """Whether this agent is an intransigent or industrial-selection agent."""
        return self.agent_type in (AgentType.INTRANSIGENT, AgentType.INDUSTRIAL_SELECTION)

    def copy(self) -> "Agent":
        """Create a shallow copy of this agent."""
        return Agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            alpha=self.alpha,
            beta=self.beta,
            personal_p_A=self.personal_p_A,
            personal_p_B=self.personal_p_B,
            base_productivity=self.base_productivity,
            funding_boost=self.funding_boost,
            is_funded=self.is_funded,
            bias_strength=self.bias_strength,
        )
