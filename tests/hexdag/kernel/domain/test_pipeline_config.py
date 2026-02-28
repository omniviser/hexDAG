"""Tests for PipelineConfig orchestrator/limits/caps fields and model_rebuild safety."""

from __future__ import annotations

from hexdag.kernel.config.models import DefaultCaps, DefaultLimits
from hexdag.kernel.domain.pipeline_config import PipelineConfig
from hexdag.kernel.orchestration.models import OrchestratorConfig


class TestPipelineConfigOrchestratorField:
    """Tests for PipelineConfig.orchestrator field."""

    def test_default_is_none(self) -> None:
        pc = PipelineConfig()
        assert pc.orchestrator is None

    def test_accepts_orchestrator_config(self) -> None:
        orch = OrchestratorConfig(max_concurrent_nodes=5)
        pc = PipelineConfig(orchestrator=orch)
        assert pc.orchestrator is not None
        assert pc.orchestrator.max_concurrent_nodes == 5

    def test_model_validate_from_dict(self) -> None:
        pc = PipelineConfig.model_validate({
            "orchestrator": {
                "max_concurrent_nodes": 3,
                "strict_validation": True,
                "default_node_timeout": 60.0,
            },
        })
        assert pc.orchestrator is not None
        assert pc.orchestrator.max_concurrent_nodes == 3
        assert pc.orchestrator.strict_validation is True
        assert pc.orchestrator.default_node_timeout == 60.0

    def test_model_validate_with_none(self) -> None:
        pc = PipelineConfig.model_validate({"orchestrator": None})
        assert pc.orchestrator is None


class TestPipelineConfigLimitsField:
    """Tests for PipelineConfig.limits field."""

    def test_default_is_none(self) -> None:
        pc = PipelineConfig()
        assert pc.limits is None

    def test_accepts_default_limits(self) -> None:
        limits = DefaultLimits(max_llm_calls=50, max_cost_usd=5.0)
        pc = PipelineConfig(limits=limits)
        assert pc.limits is not None
        assert pc.limits.max_llm_calls == 50
        assert pc.limits.max_cost_usd == 5.0

    def test_model_validate_from_dict(self) -> None:
        pc = PipelineConfig.model_validate({
            "limits": {
                "max_llm_calls": 100,
                "max_cost_usd": 10.0,
                "warning_threshold": 0.9,
            },
        })
        assert pc.limits is not None
        assert pc.limits.max_llm_calls == 100
        assert pc.limits.max_cost_usd == 10.0
        assert pc.limits.warning_threshold == 0.9

    def test_partial_limits(self) -> None:
        pc = PipelineConfig.model_validate({
            "limits": {"max_llm_calls": 50},
        })
        assert pc.limits is not None
        assert pc.limits.max_llm_calls == 50
        assert pc.limits.max_cost_usd is None  # default


class TestPipelineConfigCapsField:
    """Tests for PipelineConfig.caps field."""

    def test_default_is_none(self) -> None:
        pc = PipelineConfig()
        assert pc.caps is None

    def test_accepts_default_caps(self) -> None:
        caps = DefaultCaps(deny=["secret"])
        pc = PipelineConfig(caps=caps)
        assert pc.caps is not None
        assert pc.caps.deny == ["secret"]

    def test_model_validate_from_dict(self) -> None:
        pc = PipelineConfig.model_validate({
            "caps": {
                "default_set": ["llm", "memory"],
                "deny": ["secret", "spawner"],
            },
        })
        assert pc.caps is not None
        assert pc.caps.default_set == ["llm", "memory"]
        assert pc.caps.deny == ["secret", "spawner"]


class TestPipelineConfigFromYamlSpec:
    """Test that YAML spec dict is correctly parsed into PipelineConfig."""

    def test_full_spec_with_all_new_fields(self) -> None:
        """Simulate what yaml_builder._extract_pipeline_config does."""
        spec = {
            "nodes": [],
            "ports": {},
            "orchestrator": {"max_concurrent_nodes": 5},
            "limits": {"max_llm_calls": 100},
            "caps": {"deny": ["secret"]},
        }
        pc = PipelineConfig.model_validate({**spec, "metadata": {"name": "test"}})
        assert pc.orchestrator is not None
        assert pc.orchestrator.max_concurrent_nodes == 5
        assert pc.limits is not None
        assert pc.limits.max_llm_calls == 100
        assert pc.caps is not None
        assert pc.caps.deny == ["secret"]

    def test_spec_without_new_fields(self) -> None:
        """Existing YAML without orchestrator/limits/caps should still work."""
        spec = {
            "nodes": [],
            "ports": {"llm": {"adapter": "mock"}},
        }
        pc = PipelineConfig.model_validate({**spec, "metadata": {"name": "test"}})
        assert pc.orchestrator is None
        assert pc.limits is None
        assert pc.caps is None
        assert pc.ports == {"llm": {"adapter": "mock"}}


class TestPipelineConfigRebuildSafety:
    """Verify PipelineConfig works with direct imports."""

    def test_direct_import_and_instantiate(self) -> None:
        """Importing PipelineConfig directly should work without
        going through hexdag.kernel (the rebuild is idempotent)."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig as DirectPC

        pc = DirectPC()
        assert pc.orchestrator is None
        assert pc.limits is None
        assert pc.caps is None

    def test_model_validate_with_orchestrator_dict(self) -> None:
        """model_validate should work with forward-ref types."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig as DirectPC

        pc = DirectPC.model_validate({
            "orchestrator": {"max_concurrent_nodes": 5},
        })
        assert pc.orchestrator is not None
        assert pc.orchestrator.max_concurrent_nodes == 5

    def test_model_validate_with_limits_dict(self) -> None:
        from hexdag.kernel.domain.pipeline_config import PipelineConfig as DirectPC

        pc = DirectPC.model_validate({
            "limits": {"max_llm_calls": 50},
        })
        assert pc.limits is not None
        assert pc.limits.max_llm_calls == 50

    def test_model_validate_with_caps_dict(self) -> None:
        from hexdag.kernel.domain.pipeline_config import PipelineConfig as DirectPC

        pc = DirectPC.model_validate({
            "caps": {"deny": ["secret"]},
        })
        assert pc.caps is not None
        assert pc.caps.deny == ["secret"]

    def test_rebuild_idempotent(self) -> None:
        """Calling _rebuild_pipeline_config multiple times should be safe."""
        from hexdag.kernel.domain.pipeline_config import _rebuild_pipeline_config

        # Call it multiple times — should not raise
        _rebuild_pipeline_config()
        _rebuild_pipeline_config()
        _rebuild_pipeline_config()
