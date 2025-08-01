"""Unit tests for validation types (ValidationResult and ValidationContext)."""

from hexai.validation.types import ValidationContext, ValidationResult


class TestValidationContext:
    """Test cases for ValidationContext."""

    def test_default_initialization(self):
        """Test ValidationContext with default values."""
        context = ValidationContext()

        assert context.node_name is None
        assert context.pipeline_name is None
        assert context.validation_stage == "unknown"
        assert context.metadata == {}

    def test_full_initialization(self):
        """Test ValidationContext with all parameters."""
        metadata = {"test": "value"}
        context = ValidationContext(
            node_name="test_node",
            pipeline_name="test_pipeline",
            validation_stage="input",
            metadata=metadata,
        )

        assert context.node_name == "test_node"
        assert context.pipeline_name == "test_pipeline"
        assert context.validation_stage == "input"
        assert context.metadata == metadata

    def test_with_stage(self):
        """Test creating new context with updated stage."""
        original = ValidationContext(
            node_name="test_node",
            pipeline_name="test_pipeline",
            validation_stage="input",
            metadata={"key": "value"},
        )

        new_context = original.with_stage("output")

        # Original should be unchanged
        assert original.validation_stage == "input"

        # New context should have updated stage but same other values
        assert new_context.node_name == "test_node"
        assert new_context.pipeline_name == "test_pipeline"
        assert new_context.validation_stage == "output"
        assert new_context.metadata == {"key": "value"}
        assert new_context.metadata is not original.metadata  # Should be a copy


class TestValidationResult:
    """Test cases for ValidationResult."""

    def test_direct_initialization_success(self):
        """Test direct initialization of successful result."""
        result = ValidationResult(is_valid=True, data="test_data", warnings=["warning1"])

        assert result.is_valid is True
        assert result.data == "test_data"
        assert result.errors == []
        assert result.warnings == ["warning1"]
        assert result.metadata == {}
        assert bool(result) is True

    def test_direct_initialization_failure(self):
        """Test direct initialization of failed result."""
        result = ValidationResult(
            is_valid=False, data="test_data", errors=["error1", "error2"], warnings=["warning1"]
        )

        assert result.is_valid is False
        assert result.data == "test_data"
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]
        assert bool(result) is False

    def test_success_factory_method(self):
        """Test ValidationResult.success() factory method."""
        result = ValidationResult.success("test_data")

        assert result.is_valid is True
        assert result.data == "test_data"
        assert result.errors == []
        assert result.warnings == []
        assert bool(result) is True

    def test_success_factory_with_warnings(self):
        """Test ValidationResult.success() with warnings."""
        warnings = ["warning1", "warning2"]
        result = ValidationResult.success("test_data", warnings)

        assert result.is_valid is True
        assert result.data == "test_data"
        assert result.errors == []
        assert result.warnings == warnings
        assert bool(result) is True

    def test_failure_factory_method(self):
        """Test ValidationResult.failure() factory method."""
        errors = ["error1", "error2"]
        result = ValidationResult.failure("test_data", errors)

        assert result.is_valid is False
        assert result.data == "test_data"
        assert result.errors == errors
        assert result.warnings == []
        assert bool(result) is False

    def test_failure_factory_with_warnings(self):
        """Test ValidationResult.failure() with warnings."""
        errors = ["error1"]
        warnings = ["warning1"]
        result = ValidationResult.failure("test_data", errors, warnings)

        assert result.is_valid is False
        assert result.data == "test_data"
        assert result.errors == errors
        assert result.warnings == warnings
        assert bool(result) is False

    def test_add_error(self):
        """Test adding errors to validation result."""
        result = ValidationResult.success("test_data")
        assert result.is_valid is True

        result.add_error("new error")

        assert result.is_valid is False
        assert "new error" in result.errors
        assert bool(result) is False

    def test_add_warning(self):
        """Test adding warnings to validation result."""
        result = ValidationResult.success("test_data")

        result.add_warning("new warning")

        assert result.is_valid is True  # Should still be valid
        assert "new warning" in result.warnings
        assert bool(result) is True

    def test_bool_conversion(self):
        """Test boolean conversion of ValidationResult."""
        success_result = ValidationResult.success("data")
        failure_result = ValidationResult.failure("data", ["error"])

        assert bool(success_result) is True
        assert bool(failure_result) is False

        # Test in conditional context
        if success_result:
            success_tested = True
        else:
            success_tested = False

        if failure_result:
            failure_tested = True
        else:
            failure_tested = False

        assert success_tested is True
        assert failure_tested is False
