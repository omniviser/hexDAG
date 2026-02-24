"""State Memory Plugin for structured entities and belief states."""

import time
from typing import Any

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.data_store import SupportsKeyValue
from hexdag.stdlib.adapters.memory.schemas import BeliefState, EntityState

logger = get_logger(__name__)


class StateMemoryPlugin:
    """Memory plugin for structured state (entities, relationships, beliefs).

    Wraps a base Memory adapter and provides domain-specific operations for:
    - Entity state: entities with properties and relationships
    - Belief state: Hinton-style probability distributions over hypotheses

    This plugin does NOT store data itself - it delegates to the underlying
    Memory port implementation and adds state management logic.

    Example
    -------
        from hexdag.stdlib.adapters.memory import InMemoryMemory

        storage = InMemoryMemory()
        state_memory = StateMemoryPlugin(storage=storage)

        # Entity operations
        await state_memory.update_entity("agent1", "user_123", {"name": "Alice"})
        await state_memory.add_relationship("agent1", "user_123", "knows", "user_456")
        entities = await state_memory.get_entities("agent1")

        # Belief operations (Hinton-style Bayesian updates)
        await state_memory.update_beliefs("agent1", {"hypothesis_a": 0.7}, "new evidence")
        belief = await state_memory.get_belief_state("agent1")
    """

    plugin_type = "state"

    def __init__(self, storage: SupportsKeyValue):
        """Initialize state memory plugin.

        Args
        ----
            storage: Base Memory port implementation (InMemoryMemory, SQLiteMemoryAdapter, etc.)
        """
        self.storage = storage

    async def aget(self, key: str) -> Any:
        """Get value from state scope.

        Delegates to underlying storage with state:: prefix.
        """
        return await self.storage.aget(f"state::{key}")

    async def aset(self, key: str, value: Any) -> None:
        """Set value in state scope.

        Delegates to underlying storage with state:: prefix.
        """
        await self.storage.aset(f"state::{key}", value)

    # Entity state operations

    async def get_entities(self, agent_id: str) -> EntityState:
        """Get entity state for agent.

        Args
        ----
            agent_id: Agent identifier

        Returns
        -------
            EntityState with entities and relationships
        """
        data = await self.aget(f"entities:{agent_id}")
        if data is None:
            return EntityState(
                entities={},
                relationships=[],
                updated_at=time.time(),
            )
        return EntityState.model_validate(data)

    async def update_entity(
        self,
        agent_id: str,
        entity_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Update or create entity with properties.

        Args
        ----
            agent_id: Agent identifier
            entity_id: Entity identifier
            properties: Entity properties to set/update
        """
        state = await self.get_entities(agent_id)
        state.entities[entity_id] = properties
        state.updated_at = time.time()
        await self.aset(f"entities:{agent_id}", state.model_dump())
        logger.debug("Updated entity %s for agent %s", entity_id, agent_id)

    async def get_entity(
        self,
        agent_id: str,
        entity_id: str,
    ) -> dict[str, Any] | None:
        """Get single entity by ID.

        Args
        ----
            agent_id: Agent identifier
            entity_id: Entity identifier

        Returns
        -------
            Entity properties or None if not found
        """
        state = await self.get_entities(agent_id)
        return state.entities.get(entity_id)

    async def add_relationship(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        object: str,
    ) -> None:
        """Add relationship between entities.

        Args
        ----
            agent_id: Agent identifier
            subject: Subject entity ID
            predicate: Relationship type
            object: Object entity ID
        """
        state = await self.get_entities(agent_id)
        relationship = (subject, predicate, object)

        # Avoid duplicates
        if relationship not in state.relationships:
            state.relationships.append(relationship)
            state.updated_at = time.time()
            await self.aset(f"entities:{agent_id}", state.model_dump())
            logger.debug(
                "Added relationship (%s, %s, %s) for agent %s",
                subject,
                predicate,
                object,
                agent_id,
            )

    async def query_relationships(
        self,
        agent_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Query relationships by subject, predicate, or object.

        Args
        ----
            agent_id: Agent identifier
            subject: Optional subject to filter by
            predicate: Optional predicate to filter by
            object: Optional object to filter by

        Returns
        -------
            List of matching relationships
        """
        state = await self.get_entities(agent_id)
        results = []

        for rel in state.relationships:
            s, p, o = rel
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if object is not None and o != object:
                continue
            results.append(rel)

        return results

    # Belief state operations (Hinton-style)

    async def get_belief_state(self, agent_id: str) -> BeliefState:
        """Get Hinton-style belief state for agent.

        Args
        ----
            agent_id: Agent identifier

        Returns
        -------
            BeliefState with probability distribution over hypotheses
        """
        data = await self.aget(f"belief:{agent_id}")
        if data is None:
            return BeliefState(
                beliefs={},
                confidence=0.0,
                evidence=[],
                updated_at=time.time(),
            )
        return BeliefState.model_validate(data)

    async def update_beliefs(
        self,
        agent_id: str,
        new_beliefs: dict[str, float],
        evidence: str,
    ) -> None:
        """Bayesian belief update with new evidence.

        Args
        ----
            agent_id: Agent identifier
            new_beliefs: New belief values (likelihoods)
            evidence: Description of the evidence
        """
        state = await self.get_belief_state(agent_id)

        # Bayesian update: P(H|E) ∝ P(E|H) * P(H)
        for hypothesis, prior in list(state.beliefs.items()):
            if hypothesis in new_beliefs:
                likelihood = new_beliefs[hypothesis]
                state.beliefs[hypothesis] = likelihood * prior

        # Add new hypotheses
        for hypothesis, prob in new_beliefs.items():
            if hypothesis not in state.beliefs:
                state.beliefs[hypothesis] = prob

        # Normalize to sum to 1.0
        total = sum(state.beliefs.values())
        if total > 0:
            state.beliefs = {h: p / total for h, p in state.beliefs.items()}

        # Update confidence (max posterior probability)
        state.confidence = max(state.beliefs.values()) if state.beliefs else 0.0

        # Record evidence
        state.evidence.append(evidence)
        state.updated_at = time.time()

        await self.aset(f"belief:{agent_id}", state.model_dump())
        logger.debug(
            "Updated beliefs for agent %s: confidence=%.2f",
            agent_id,
            state.confidence,
        )

    async def set_belief(
        self,
        agent_id: str,
        hypothesis: str,
        probability: float,
    ) -> None:
        """Set belief probability for single hypothesis.

        Args
        ----
            agent_id: Agent identifier
            hypothesis: Hypothesis name
            probability: Belief probability (0.0-1.0)
        """
        state = await self.get_belief_state(agent_id)
        state.beliefs[hypothesis] = max(0.0, min(1.0, probability))
        state.confidence = max(state.beliefs.values()) if state.beliefs else 0.0
        state.updated_at = time.time()
        await self.aset(f"belief:{agent_id}", state.model_dump())
