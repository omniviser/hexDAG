"""Tests for sanitized types registry and built-in cleaners."""

from __future__ import annotations

import pytest

from hexdag.kernel.validation.sanitized_types import (
    COMMON_NULLS,
    _build_cleaner_from_config,
    _clean_currency,
    _clean_flexible_bool,
    _clean_nullable,
    _clean_score,
    get_available_types,
    get_type,
    register_type,
    register_type_from_config,
)

# ============================================================================
# Built-in cleaners
# ============================================================================


class TestCleanNullable:
    def test_none_passthrough(self) -> None:
        assert _clean_nullable(None) is None

    def test_null_strings(self) -> None:
        for s in ["N/A", "n/a", "none", "None", "NULL", "TBD", "unknown", "-", "--", ""]:
            assert _clean_nullable(s) is None, f"Expected None for {s!r}"

    def test_valid_strings_pass_through(self) -> None:
        assert _clean_nullable("hello") == "hello"
        assert _clean_nullable("42") == "42"

    def test_non_string_passthrough(self) -> None:
        assert _clean_nullable(42) == 42
        assert _clean_nullable(3.14) == 3.14


class TestCleanCurrency:
    def test_dollar_amount(self) -> None:
        assert _clean_currency("$1,200") == 1200.0

    def test_dollar_with_cents(self) -> None:
        assert _clean_currency("$1,200.50") == 1200.50

    def test_plain_number_string(self) -> None:
        assert _clean_currency("1200") == 1200.0

    def test_numeric_passthrough(self) -> None:
        assert _clean_currency(1200) == 1200.0
        assert _clean_currency(1200.50) == 1200.50

    def test_null_strings(self) -> None:
        assert _clean_currency("N/A") is None
        assert _clean_currency("TBD") is None
        assert _clean_currency(None) is None

    def test_euro_amount(self) -> None:
        assert _clean_currency("€1.200") == 1.200

    def test_negative(self) -> None:
        assert _clean_currency("-500") == -500.0

    def test_empty_after_strip(self) -> None:
        assert _clean_currency("$$$") is None

    def test_invalid_string(self) -> None:
        assert _clean_currency("abc") is None


class TestCleanFlexibleBool:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "yes", "Yes", "1", "y", "on"])
    def test_truthy(self, val: str) -> None:
        assert _clean_flexible_bool(val) is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "no", "No", "0", "n", "off"])
    def test_falsy(self, val: str) -> None:
        assert _clean_flexible_bool(val) is False

    def test_none(self) -> None:
        assert _clean_flexible_bool(None) is None

    def test_null_string(self) -> None:
        assert _clean_flexible_bool("N/A") is None

    def test_bool_passthrough(self) -> None:
        assert _clean_flexible_bool(True) is True
        assert _clean_flexible_bool(False) is False


class TestCleanScore:
    def test_normal_score(self) -> None:
        assert _clean_score(0.85) == 0.85

    def test_string_score(self) -> None:
        assert _clean_score("0.95") == 0.95

    def test_clamp_high(self) -> None:
        assert _clean_score(1.5) == 1.0

    def test_clamp_low(self) -> None:
        assert _clean_score(-0.3) == 0.0

    def test_none(self) -> None:
        assert _clean_score(None) is None

    def test_null_string(self) -> None:
        assert _clean_score("N/A") is None

    def test_invalid_string(self) -> None:
        assert _clean_score("abc") is None


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_builtin_types_registered(self) -> None:
        available = get_available_types()
        assert "currency" in available
        assert "flexible_bool" in available
        assert "score" in available
        assert "upper_str" in available
        assert "lower_str" in available
        assert "nullable_str" in available
        assert "trimmed_str" in available

    def test_get_type_returns_annotated(self) -> None:
        t = get_type("currency")
        assert t is not None

    def test_get_type_unknown_returns_none(self) -> None:
        assert get_type("nonexistent_type_xyz") is None

    def test_register_custom_type(self) -> None:
        register_type(
            "test_custom_1",
            str,
            lambda v: str(v).upper() if v else v,
            description="Test custom type",
        )
        assert get_type("test_custom_1") is not None
        assert "test_custom_1" in get_available_types()

    def test_descriptions_populated(self) -> None:
        available = get_available_types()
        for name, desc in available.items():
            assert desc, f"Type '{name}' has empty description"


# ============================================================================
# Config-driven registration
# ============================================================================


class TestRegisterFromConfig:
    def test_simple_pattern(self) -> None:
        register_type_from_config("test_mc", {"base": "str", "pattern": r"MC-?\d+"})
        t = get_type("test_mc")
        assert t is not None

    def test_multi_pattern(self) -> None:
        register_type_from_config(
            "test_carrier",
            {"base": "str", "pattern": [r"MC-?\d+", r"Motor Carrier:?\s*(\d+)"]},
        )
        t = get_type("test_carrier")
        assert t is not None

    def test_invalid_base(self) -> None:
        with pytest.raises(ValueError, match="invalid base"):
            register_type_from_config("test_bad", {"base": "complex"})

    def test_clamp_config(self) -> None:
        register_type_from_config("test_rating", {"base": "float", "clamp": [1.0, 5.0]})
        assert get_type("test_rating") is not None

    def test_custom_bool_values(self) -> None:
        register_type_from_config(
            "test_avail",
            {
                "base": "bool",
                "true_values": ["available", "open"],
                "false_values": ["booked", "taken"],
            },
        )
        assert get_type("test_avail") is not None


class TestBuildCleanerFromConfig:
    def test_pattern_extraction(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "pattern": r"MC-?\d+"})
        assert cleaner("Carrier MC-123456 verified") == "MC-123456"

    def test_multi_pattern_first_match(self) -> None:
        cleaner = _build_cleaner_from_config({
            "base": "str",
            "pattern": [r"MC-?\d+", r"Motor Carrier:?\s*(\d+)"],
        })
        assert cleaner("Carrier MC-123456") == "MC-123456"
        assert cleaner("Motor Carrier: 789") == "789"

    def test_pattern_no_match_returns_none(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "pattern": r"MC-?\d+"})
        assert cleaner("no match here") is None

    def test_strip_and_float(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "float", "strip": "$,"})
        assert cleaner("$1,200.50") == 1200.50

    def test_null_strings(self) -> None:
        cleaner = _build_cleaner_from_config({
            "base": "float",
            "nulls": ["N/A", "TBD"],
        })
        assert cleaner("N/A") is None
        assert cleaner("tbd") is None
        assert cleaner("42") == 42.0

    def test_common_nulls_shorthand(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "nulls": "common"})
        for null_str in ["N/A", "none", "TBD", "unknown"]:
            assert cleaner(null_str) is None
        assert cleaner("valid") == "valid"

    def test_clamp(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "float", "clamp": [0.0, 1.0]})
        assert cleaner("1.5") == 1.0
        assert cleaner("-0.3") == 0.0
        assert cleaner("0.7") == 0.7

    def test_upper(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "upper": True})
        assert cleaner("hello") == "HELLO"

    def test_lower(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "lower": True})
        assert cleaner("HELLO") == "hello"

    def test_max_length(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "max_length": 2})
        assert cleaner("CALIFORNIA") == "CA"

    def test_trim_default(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str"})
        assert cleaner("  hello  ") == "hello"

    def test_trim_disabled(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str", "trim": False})
        assert cleaner("  hello  ") == "  hello  "

    def test_default_value(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "float", "default": 0.0})
        assert cleaner(None) == 0.0

    def test_custom_bool_true_values(self) -> None:
        cleaner = _build_cleaner_from_config({
            "base": "bool",
            "true_values": ["available", "open"],
            "false_values": ["booked", "taken"],
        })
        assert cleaner("available") is True
        assert cleaner("booked") is False
        assert cleaner("yes") is True  # fallback to defaults

    def test_int_conversion(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "int"})
        assert cleaner("42") == 42
        assert cleaner("42.7") == 42

    def test_int_clamp(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "int", "clamp": [0, 100]})
        assert cleaner("150") == 100
        assert cleaner("-5") == 0

    def test_none_passthrough(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str"})
        assert cleaner(None) is None

    def test_composed_operations(self) -> None:
        """Test multiple cleaning operations chained together."""
        cleaner = _build_cleaner_from_config({
            "base": "float",
            "strip": "$,€",
            "nulls": ["N/A"],
            "clamp": [0, 100000],
        })
        assert cleaner("$1,200") == 1200.0
        assert cleaner("N/A") is None
        assert cleaner("$150,000") == 100000.0

    def test_pattern_with_capture_group(self) -> None:
        cleaner = _build_cleaner_from_config({
            "base": "str",
            "pattern": r"MC[- ]?(\d+)",
        })
        assert cleaner("MC-123456") == "123456"
        assert cleaner("MC 789") == "789"

    def test_empty_string_after_trim(self) -> None:
        cleaner = _build_cleaner_from_config({"base": "str"})
        assert cleaner("   ") is None


# ============================================================================
# Pydantic integration via create_pydantic_model
# ============================================================================


class TestPydanticIntegration:
    """Test that sanitized types work through create_pydantic_model."""

    def _factory(self):  # type: ignore[no-untyped-def]
        """Get a concrete factory instance for testing."""
        from hexdag.stdlib.nodes.llm_node import LLMNode

        return LLMNode()

    def test_currency_field(self) -> None:
        model = self._factory().create_pydantic_model("TestCurrency", {"rate": "currency"})
        assert model is not None
        result = model.model_validate({"rate": "$1,200"})
        assert result.rate == 1200.0  # type: ignore[attr-defined]

    def test_flexible_bool_field(self) -> None:
        model = self._factory().create_pydantic_model("TestBool", {"is_empty": "flexible_bool"})
        assert model is not None
        result = model.model_validate({"is_empty": "yes"})
        assert result.is_empty is True  # type: ignore[attr-defined]

    def test_score_field(self) -> None:
        model = self._factory().create_pydantic_model("TestScore", {"confidence": "score"})
        assert model is not None
        result = model.model_validate({"confidence": "1.5"})
        assert result.confidence == 1.0  # type: ignore[attr-defined]

    def test_upper_str_field(self) -> None:
        model = self._factory().create_pydantic_model("TestUpper", {"name": "upper_str"})
        assert model is not None
        result = model.model_validate({"name": "acme trucking"})
        assert result.name == "ACME TRUCKING"  # type: ignore[attr-defined]

    def test_nullable_str_field(self) -> None:
        model = self._factory().create_pydantic_model("TestNullable", {"origin": "nullable_str"})
        assert model is not None
        result = model.model_validate({"origin": "N/A"})
        assert result.origin is None  # type: ignore[attr-defined]

    def test_mixed_schema(self) -> None:
        """Test that plain types and sanitized types work together."""
        model = self._factory().create_pydantic_model(
            "TestMixed",
            {
                "rate": "currency",
                "name": "str",
                "is_active": "flexible_bool",
            },
        )
        assert model is not None
        result = model.model_validate({
            "rate": "$500",
            "name": "Test",
            "is_active": "yes",
        })
        assert result.rate == 500.0  # type: ignore[attr-defined]
        assert result.name == "Test"  # type: ignore[attr-defined]
        assert result.is_active is True  # type: ignore[attr-defined]

    def test_null_value_for_sanitized_type(self) -> None:
        """Sanitized types are nullable by default."""
        model = self._factory().create_pydantic_model("TestNullCurrency", {"rate": "currency"})
        assert model is not None
        result = model.model_validate({"rate": None})
        assert result.rate is None  # type: ignore[attr-defined]

    def test_freight_end_to_end(self) -> None:
        """Full freight extraction scenario."""
        # Register custom type for this test
        register_type_from_config("test_mc_e2e", {"base": "str", "pattern": r"MC-?\d+"})

        model = self._factory().create_pydantic_model(
            "FreightExtraction",
            {
                "rate": "currency",
                "mc_number": "test_mc_e2e",
                "is_empty": "flexible_bool",
                "confidence": "score",
                "carrier_name": "upper_str",
                "origin": "nullable_str",
            },
        )
        assert model is not None

        # Simulate messy LLM output
        result = model.model_validate({
            "rate": "$1,200",
            "mc_number": "Carrier MC-123456 (verified)",
            "is_empty": "yes",
            "confidence": "0.95",
            "carrier_name": "acme trucking",
            "origin": "N/A",
        })

        assert result.rate == 1200.0  # type: ignore[attr-defined]
        assert result.mc_number == "MC-123456"  # type: ignore[attr-defined]
        assert result.is_empty is True  # type: ignore[attr-defined]
        assert result.confidence == 0.95  # type: ignore[attr-defined]
        assert result.carrier_name == "ACME TRUCKING"  # type: ignore[attr-defined]
        assert result.origin is None  # type: ignore[attr-defined]


# ============================================================================
# Schema conversion integration
# ============================================================================


class TestSchemaConversion:
    def test_sanitized_type_is_valid(self) -> None:
        from hexdag.kernel.utils.schema_conversion import is_valid_type_name

        assert is_valid_type_name("currency")
        assert is_valid_type_name("flexible_bool")
        assert is_valid_type_name("score")

    def test_sanitized_type_nullable_is_valid(self) -> None:
        from hexdag.kernel.utils.schema_conversion import is_valid_type_name

        assert is_valid_type_name("currency?")

    def test_unknown_type_still_invalid(self) -> None:
        from hexdag.kernel.utils.schema_conversion import is_valid_type_name

        assert not is_valid_type_name("nonexistent_xyz_type")

    def test_error_message_lists_sanitized_types(self) -> None:
        from hexdag.kernel.utils.schema_conversion import normalize_schema

        with pytest.raises(ValueError, match="Sanitized types:.*currency"):
            normalize_schema({"field": "nonexistent_xyz_type"})


# ============================================================================
# COMMON_NULLS constant
# ============================================================================


class TestCommonNulls:
    def test_contains_expected_values(self) -> None:
        expected = {"n/a", "na", "none", "null", "nil", "tbd", "unknown", "-", "--", ""}
        assert expected.issubset(COMMON_NULLS)

    def test_all_lowercase(self) -> None:
        for s in COMMON_NULLS:
            assert s == s.lower()
