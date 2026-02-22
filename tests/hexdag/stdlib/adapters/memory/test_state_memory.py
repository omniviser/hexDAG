"""Tests for StateMemoryPlugin."""

import pytest

from hexdag.stdlib.adapters.memory import InMemoryMemory, StateMemoryPlugin
from hexdag.stdlib.adapters.memory.schemas import BeliefState, EntityState


@pytest.fixture
def storage():
    """Create in-memory storage for testing."""
    return InMemoryMemory()


@pytest.fixture
def state_memory(storage):
    """Create StateMemoryPlugin with in-memory storage."""
    return StateMemoryPlugin(storage=storage)


class TestStateMemoryPluginEntities:
    """Test entity and relationship management."""

    @pytest.mark.asyncio
    async def test_plugin_type(self, state_memory):
        """Test plugin type identifier."""
        assert state_memory.plugin_type == "state"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entities(self, state_memory):
        """Test getting entities for non-existent agent creates empty state."""
        state = await state_memory.get_entities("agent_123")

        assert isinstance(state, EntityState)
        assert len(state.entities) == 0
        assert len(state.relationships) == 0

    @pytest.mark.asyncio
    async def test_add_single_entity(self, state_memory):
        """Test adding a single entity."""
        agent_id = "agent_456"

        await state_memory.update_entity(
            agent_id,
            entity_id="user_1",
            properties={"name": "Alice", "age": 30},
        )

        state = await state_memory.get_entities(agent_id)

        assert "user_1" in state.entities
        # Entities store properties directly
        assert state.entities["user_1"]["name"] == "Alice"
        assert state.entities["user_1"]["age"] == 30

    @pytest.mark.asyncio
    async def test_add_multiple_entities(self, state_memory):
        """Test adding multiple entities."""
        agent_id = "agent_789"

        # Add first entity
        await state_memory.update_entity(
            agent_id,
            entity_id="user_1",
            properties={"name": "Alice"},
        )

        # Add second entity
        await state_memory.update_entity(
            agent_id,
            entity_id="user_2",
            properties={"name": "Bob"},
        )

        state = await state_memory.get_entities(agent_id)

        assert len(state.entities) == 2
        assert "user_1" in state.entities
        assert "user_2" in state.entities

    @pytest.mark.asyncio
    async def test_update_existing_entity(self, state_memory):
        """Test updating an existing entity."""
        agent_id = "agent_update"

        # Add entity
        await state_memory.update_entity(
            agent_id,
            entity_id="user_1",
            properties={"name": "Alice", "status": "active"},
        )

        # Update same entity
        await state_memory.update_entity(
            agent_id,
            entity_id="user_1",
            properties={"name": "Alice", "status": "inactive", "last_seen": "2024-01-15"},
        )

        state = await state_memory.get_entities(agent_id)

        assert len(state.entities) == 1
        assert state.entities["user_1"]["status"] == "inactive"
        assert state.entities["user_1"]["last_seen"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_add_relationship(self, state_memory):
        """Test adding relationships between entities."""
        agent_id = "agent_rel"

        # Add entities first
        await state_memory.update_entity(agent_id, entity_id="user_1", properties={"name": "Alice"})
        await state_memory.update_entity(agent_id, entity_id="user_2", properties={"name": "Bob"})

        # Add relationship
        await state_memory.add_relationship(
            agent_id, subject="user_1", predicate="knows", object="user_2"
        )

        state = await state_memory.get_entities(agent_id)

        assert len(state.relationships) == 1
        assert state.relationships[0] == ("user_1", "knows", "user_2")

    @pytest.mark.asyncio
    async def test_multiple_relationships(self, state_memory):
        """Test multiple relationships between entities."""
        agent_id = "agent_multi_rel"

        # Add entities
        await state_memory.update_entity(agent_id, entity_id="alice", properties={})
        await state_memory.update_entity(agent_id, entity_id="bob", properties={})
        await state_memory.update_entity(agent_id, entity_id="company", properties={})

        # Add relationships
        await state_memory.add_relationship(agent_id, "alice", "knows", "bob")
        await state_memory.add_relationship(agent_id, "alice", "works_at", "company")
        await state_memory.add_relationship(agent_id, "bob", "works_at", "company")

        state = await state_memory.get_entities(agent_id)

        assert len(state.relationships) == 3
        assert ("alice", "knows", "bob") in state.relationships
        assert ("alice", "works_at", "company") in state.relationships
        assert ("bob", "works_at", "company") in state.relationships

    @pytest.mark.asyncio
    async def test_storage_namespacing_entities(self, storage, state_memory):
        """Test that entity state uses 'entity::' prefix in storage."""
        agent_id = "test_namespace"

        await state_memory.update_entity(agent_id, entity_id="user_1", properties={})

        # Direct storage access should require state::entities: prefix
        raw_data = await storage.aget(f"state::entities:{agent_id}")
        assert raw_data is not None
        assert "user_1" in raw_data["entities"]


class TestStateMemoryPluginBeliefs:
    """Test Bayesian belief management."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_beliefs(self, state_memory):
        """Test getting beliefs for non-existent agent creates empty state."""
        state = await state_memory.get_belief_state("agent_123")

        assert isinstance(state, BeliefState)
        assert len(state.beliefs) == 0
        assert state.confidence == 0.0
        assert len(state.evidence) == 0

    @pytest.mark.asyncio
    async def test_update_beliefs_initial(self, state_memory):
        """Test initial belief update (no priors)."""
        agent_id = "agent_belief_1"

        new_beliefs = {"hypothesis_a": 0.7, "hypothesis_b": 0.3}
        await state_memory.update_beliefs(agent_id, new_beliefs, "Initial evidence")

        state = await state_memory.get_belief_state(agent_id)

        # Should normalize to sum to 1.0
        total = sum(state.beliefs.values())
        assert abs(total - 1.0) < 0.0001

        # Confidence should be max belief
        assert state.confidence == max(state.beliefs.values())
        assert "Initial evidence" in state.evidence

    @pytest.mark.asyncio
    async def test_bayesian_belief_update(self, state_memory):
        """Test Bayesian belief update: P(H|E) ∝ P(E|H) × P(H)."""
        agent_id = "agent_bayesian"

        # Set initial priors
        initial_beliefs = {"rain": 0.3, "sunny": 0.7}
        await state_memory.update_beliefs(agent_id, initial_beliefs, "Weather forecast")

        # Update with new evidence (likelihood)
        # If we see dark clouds: P(clouds|rain) = 0.9, P(clouds|sunny) = 0.2
        likelihoods = {"rain": 0.9, "sunny": 0.2}
        await state_memory.update_beliefs(agent_id, likelihoods, "Dark clouds observed")

        state = await state_memory.get_belief_state(agent_id)

        # Posterior ∝ likelihood × prior
        # rain: 0.9 × 0.3 = 0.27
        # sunny: 0.2 × 0.7 = 0.14
        # Normalized: rain = 0.27 / 0.41 ≈ 0.66, sunny ≈ 0.34

        assert state.beliefs["rain"] > state.beliefs["sunny"]
        assert abs(state.beliefs["rain"] - 0.66) < 0.01
        assert abs(sum(state.beliefs.values()) - 1.0) < 0.0001

    @pytest.mark.asyncio
    async def test_belief_confidence_tracking(self, state_memory):
        """Test that confidence tracks maximum belief probability."""
        agent_id = "agent_confidence"

        # Low confidence (beliefs are spread)
        low_conf_beliefs = {"a": 0.3, "b": 0.3, "c": 0.4}
        await state_memory.update_beliefs(agent_id, low_conf_beliefs, "Evidence 1")

        state = await state_memory.get_belief_state(agent_id)
        assert state.confidence == 0.4

        # High confidence (one dominant belief)
        high_conf_beliefs = {"a": 0.1, "b": 0.05, "c": 0.9}
        await state_memory.update_beliefs(agent_id, high_conf_beliefs, "Evidence 2")

        state = await state_memory.get_belief_state(agent_id)
        assert state.confidence > 0.85

    @pytest.mark.asyncio
    async def test_evidence_accumulation(self, state_memory):
        """Test that evidence list accumulates over updates."""
        agent_id = "agent_evidence"

        await state_memory.update_beliefs(agent_id, {"h1": 0.5}, "Evidence A")
        await state_memory.update_beliefs(agent_id, {"h1": 0.8}, "Evidence B")
        await state_memory.update_beliefs(agent_id, {"h1": 0.9}, "Evidence C")

        state = await state_memory.get_belief_state(agent_id)

        assert len(state.evidence) == 3
        assert "Evidence A" in state.evidence
        assert "Evidence B" in state.evidence
        assert "Evidence C" in state.evidence

    @pytest.mark.asyncio
    async def test_new_hypothesis_introduction(self, state_memory):
        """Test introducing new hypotheses during update."""
        agent_id = "agent_new_hyp"

        # Initial beliefs
        await state_memory.update_beliefs(
            agent_id, {"hypothesis_a": 0.6, "hypothesis_b": 0.4}, "Initial"
        )

        # Update with new hypothesis
        await state_memory.update_beliefs(
            agent_id,
            {"hypothesis_a": 0.5, "hypothesis_b": 0.3, "hypothesis_c": 0.7},
            "New evidence",
        )

        state = await state_memory.get_belief_state(agent_id)

        # hypothesis_c should appear (0.7 × 0 = 0, but we handle new hypotheses)
        # Note: Current implementation multiplies by prior (0), so new hypotheses get 0 probability
        # This documents current behavior - may want to handle this differently

        assert "hypothesis_a" in state.beliefs
        assert "hypothesis_b" in state.beliefs

    @pytest.mark.asyncio
    async def test_storage_namespacing_beliefs(self, storage, state_memory):
        """Test that belief state uses 'belief::' prefix in storage."""
        agent_id = "test_namespace"

        await state_memory.update_beliefs(agent_id, {"h1": 0.5}, "Test")

        # Direct storage access should require state::belief: prefix
        raw_data = await storage.aget(f"state::belief:{agent_id}")
        assert raw_data is not None
        assert "h1" in raw_data["beliefs"]


class TestStateMemoryPluginEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_entity_with_empty_properties(self, state_memory):
        """Test adding entity with no properties."""
        agent_id = "agent_empty"

        await state_memory.update_entity(agent_id, entity_id="entity_1", properties={})

        state = await state_memory.get_entities(agent_id)

        assert "entity_1" in state.entities
        assert state.entities["entity_1"] == {}

    @pytest.mark.asyncio
    async def test_relationship_with_nonexistent_entities(self, state_memory):
        """Test adding relationship where entities don't exist."""
        agent_id = "agent_no_entities"

        # Add relationship without creating entities first
        await state_memory.add_relationship(
            agent_id, subject="user_1", predicate="knows", object="user_2"
        )

        state = await state_memory.get_entities(agent_id)

        # Relationship should still be added (no validation)
        assert len(state.relationships) == 1

    @pytest.mark.asyncio
    async def test_belief_update_with_zero_probabilities(self, state_memory):
        """Test belief update when some probabilities are zero."""
        agent_id = "agent_zero"

        # Initial beliefs
        await state_memory.update_beliefs(agent_id, {"h1": 0.5, "h2": 0.5}, "Initial")

        # Update with zero likelihood for h2
        await state_memory.update_beliefs(agent_id, {"h1": 0.8, "h2": 0.0}, "Update")

        state = await state_memory.get_belief_state(agent_id)

        # h2 should become 0 (0.0 × 0.5 = 0)
        # h1 should be 1.0 after normalization
        assert state.beliefs["h1"] == 1.0
        assert state.beliefs["h2"] == 0.0

    @pytest.mark.asyncio
    async def test_belief_update_with_all_zeros(self, state_memory):
        """Test belief update when all likelihoods are zero."""
        agent_id = "agent_all_zero"

        await state_memory.update_beliefs(agent_id, {"h1": 0.5, "h2": 0.5}, "Initial")

        # Update with all zeros
        await state_memory.update_beliefs(agent_id, {"h1": 0.0, "h2": 0.0}, "Impossible")

        state = await state_memory.get_belief_state(agent_id)

        # Total is 0, cannot normalize - beliefs should remain 0
        assert sum(state.beliefs.values()) == 0.0
        assert state.confidence == 0.0

    @pytest.mark.asyncio
    async def test_multiple_agents_independent_entities(self, state_memory):
        """Test that multiple agents have independent entity states."""
        agent_1 = "agent_one"
        agent_2 = "agent_two"

        await state_memory.update_entity(agent_1, entity_id="entity_1", properties={})
        await state_memory.update_entity(agent_2, entity_id="entity_2", properties={})

        state_1 = await state_memory.get_entities(agent_1)
        state_2 = await state_memory.get_entities(agent_2)

        assert "entity_1" in state_1.entities
        assert "entity_1" not in state_2.entities
        assert "entity_2" in state_2.entities
        assert "entity_2" not in state_1.entities

    @pytest.mark.asyncio
    async def test_multiple_agents_independent_beliefs(self, state_memory):
        """Test that multiple agents have independent belief states."""
        agent_1 = "agent_one"
        agent_2 = "agent_two"

        await state_memory.update_beliefs(agent_1, {"h1": 0.8}, "Agent 1 evidence")
        await state_memory.update_beliefs(agent_2, {"h2": 0.9}, "Agent 2 evidence")

        state_1 = await state_memory.get_belief_state(agent_1)
        state_2 = await state_memory.get_belief_state(agent_2)

        assert "h1" in state_1.beliefs
        assert "h1" not in state_2.beliefs
        assert "h2" in state_2.beliefs
        assert "h2" not in state_1.beliefs
