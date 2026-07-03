"""Tests for reference_resolver — extracting node refs from mappings, expressions, templates."""

from hexdag.compiler.reference_resolver import (
    extract_input_refs_from_mapping,
    extract_jinja_head_names,
    extract_refs_from_expressions,
    extract_refs_from_mapping,
    extract_refs_from_spec,
    extract_refs_from_string,
    extract_refs_from_template,
)

KNOWN_NODES = frozenset({"analyzer", "fetcher", "product", "order", "cleanup"})


class TestExtractRefsFromMapping:
    """Tests for extract_refs_from_mapping."""

    def test_node_field_reference(self):
        mapping = {"result": "analyzer.output"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_multiple_node_refs(self):
        mapping = {
            "analysis": "analyzer.result",
            "data": "fetcher.response",
        }
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "fetcher"}

    def test_dollar_input_skipped(self):
        mapping = {
            "request_id": "$input.id",
            "analysis": "analyzer.result",
        }
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_pure_dollar_input_skipped(self):
        mapping = {"data": "$input.field"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_unknown_node_ignored(self):
        mapping = {"data": "unknown_node.field"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_simple_field_name_no_dot(self):
        mapping = {"data": "some_field"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_non_string_values_skipped(self):
        mapping = {"data": 42, "other": "analyzer.result"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_empty_mapping(self):
        refs = extract_refs_from_mapping({}, KNOWN_NODES)
        assert refs == set()


class TestExtractRefsFromExpressions:
    """Tests for extract_refs_from_expressions."""

    def test_node_field_in_expression(self):
        expressions = {"total": "product.price * order.quantity"}
        refs = extract_refs_from_expressions(expressions, KNOWN_NODES)
        assert refs == {"product", "order"}

    def test_builtin_names_skipped(self):
        expressions = {"count": "len(items)"}
        refs = extract_refs_from_expressions(expressions, KNOWN_NODES)
        assert refs == set()

    def test_mixed_refs_and_builtins(self):
        expressions = {"valid": "len(analyzer.items) > 0"}
        refs = extract_refs_from_expressions(expressions, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_no_refs(self):
        expressions = {"val": "42 + 10"}
        refs = extract_refs_from_expressions(expressions, KNOWN_NODES)
        assert refs == set()


class TestExtractRefsFromTemplate:
    """Tests for extract_refs_from_template."""

    def test_jinja_variable(self):
        template = "Analyze: {{analyzer.result}}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_jinja_with_spaces(self):
        template = "Data: {{ fetcher.response }}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == {"fetcher"}

    def test_multiple_jinja_refs(self):
        template = "Product: {{product.name}}, Order: {{order.id}}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == {"product", "order"}

    def test_no_node_refs(self):
        template = "Hello {{name}}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == set()

    def test_unknown_node_ignored(self):
        template = "Result: {{unknown.field}}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == set()


MACRO_INSTANCES = frozenset({"extract_rate"})


class TestMacroPrefixedRefs:
    """Tests for macro-prefixed node reference detection."""

    def test_macro_prefixed_ref_in_mapping(self):
        """Macro-generated node name in input_mapping infers macro instance as dep."""
        known = frozenset({"extract_rate", "other"})
        mapping = {"data": "extract_rate_result.output"}
        refs = extract_refs_from_mapping(mapping, known, MACRO_INSTANCES)
        assert refs == {"extract_rate"}

    def test_macro_prefixed_ref_in_expressions(self):
        """Macro-generated node name in expressions infers macro instance as dep."""
        known = frozenset({"extract_rate", "other"})
        expressions = {"total": "extract_rate_final.amount * 2"}
        refs = extract_refs_from_expressions(expressions, known, MACRO_INSTANCES)
        assert refs == {"extract_rate"}

    def test_macro_prefixed_ref_in_template(self):
        """Macro-generated node name in template infers macro instance as dep."""
        known = frozenset({"extract_rate", "other"})
        template = "Rate: {{extract_rate_result.rate}}"
        refs = extract_refs_from_template(template, known, MACRO_INSTANCES)
        assert refs == {"extract_rate"}

    def test_non_macro_prefix_ignored(self):
        """Unknown node name not matching a macro prefix is ignored."""
        known = frozenset({"extract_rate", "other"})
        mapping = {"data": "unknown_node.output"}
        refs = extract_refs_from_mapping(mapping, known, MACRO_INSTANCES)
        assert refs == set()

    def test_no_macro_instances_backward_compatible(self):
        """Omitting macro_instances works (backward compatible)."""
        known = frozenset({"analyzer"})
        mapping = {"data": "analyzer.output"}
        refs = extract_refs_from_mapping(mapping, known)
        assert refs == {"analyzer"}

    def test_overlapping_macro_names_longest_wins(self):
        """When macros have overlapping prefixes, the longest match wins.

        Regression: frozenset iteration is arbitrary, so without sorting by
        length descending, 'extract' could match 'extract_rate_node' before
        the correct 'extract_rate' gets a chance.
        """
        macros = frozenset({"extract", "extract_rate"})
        known = frozenset({"extract", "extract_rate"})
        mapping = {"data": "extract_rate_node.output"}
        refs = extract_refs_from_mapping(mapping, known, macros)
        assert refs == {"extract_rate"}

    def test_overlapping_macro_names_bare_name(self):
        """Bare macro-prefixed name also matches longest prefix."""
        macros = frozenset({"extract", "extract_rate"})
        known = frozenset({"extract", "extract_rate"})
        refs = extract_refs_from_string("extract_rate_node", known, macros)
        assert refs == {"extract_rate"}


class TestExtractRefsFromString:
    """Tests for extract_refs_from_string (used by composite node scanning)."""

    def test_node_field_reference(self):
        refs = extract_refs_from_string("checker.done == False", KNOWN_NODES)
        assert refs == set()  # checker not in KNOWN_NODES

    def test_known_node_reference(self):
        refs = extract_refs_from_string("analyzer.done == True", KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_macro_prefixed_reference(self):
        known = frozenset({"extract_rate"})
        refs = extract_refs_from_string("extract_rate_result.flag == True", known, MACRO_INSTANCES)
        assert refs == {"extract_rate"}


class TestBareNodeNames:
    """Tests for bare node name detection (no dot) in reference extraction."""

    def test_bare_known_node_in_mapping(self):
        """Bare node name in input_mapping resolves as a dependency."""
        mapping = {"data": "analyzer"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_bare_macro_prefixed_name_in_mapping(self):
        """Bare macro-generated node name infers the macro instance as dep."""
        known = frozenset({"extract_rate"})
        mapping = {"data": "extract_rate_result"}
        refs = extract_refs_from_mapping(mapping, known, MACRO_INSTANCES)
        assert refs == {"extract_rate"}

    def test_bare_unknown_name_ignored(self):
        """Bare name not matching any known node or macro is ignored."""
        mapping = {"data": "unknown_thing"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_bare_known_node_in_string(self):
        """Bare node name in a string ref (e.g., when clause)."""
        refs = extract_refs_from_string("analyzer", KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_bare_builtin_name_not_treated_as_node(self):
        """Builtin names like 'len' should not be treated as node refs."""
        refs = extract_refs_from_string("len", KNOWN_NODES)
        assert refs == set()


class TestListValuesInMapping:
    """Tests for list-valued input_mapping entries."""

    def test_list_of_node_refs(self):
        """List values in input_mapping should be scanned for node refs."""
        mapping = {"data": ["analyzer.result", "fetcher.response"]}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "fetcher"}

    def test_mixed_list_and_string(self):
        """Mix of string and list values."""
        mapping = {
            "single": "analyzer.output",
            "multi": ["fetcher.data", "order.id"],
        }
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "fetcher", "order"}

    def test_list_with_non_string_items_skipped(self):
        """Non-string items within a list are safely skipped."""
        mapping = {"data": ["analyzer.result", 42, None, True]}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_numeric_value_skipped(self):
        """Numeric values in input_mapping are silently skipped."""
        mapping = {"count": 42}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_empty_list_returns_no_refs(self):
        """Empty list produces no refs."""
        mapping = {"data": []}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()


class TestCtxReservedPrefix:
    """ctx references must not be treated as node dependencies."""

    def test_ctx_field_not_a_dependency(self):
        """ctx.run_id should not extract as a node reference."""
        mapping = {"tag": "ctx.run_id"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_ctx_in_expression(self):
        """ctx.pipeline_name in expression is not a dep."""
        exprs = {"check": "ctx.pipeline_name == 'test'"}
        refs = extract_refs_from_expressions(exprs, KNOWN_NODES)
        assert refs == set()

    def test_ctx_in_template(self):
        """{{ctx.run_id}} in template is not a dep."""
        template = "Run {{ctx.run_id}} for {{analyzer.result}}"
        refs = extract_refs_from_template(template, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_ctx_in_when_string(self):
        """ctx.pipeline_name in a when clause string is not a dep."""
        refs = extract_refs_from_string("ctx.pipeline_name == 'test'", KNOWN_NODES)
        assert refs == set()

    def test_ctx_mixed_with_node_refs(self):
        """ctx refs coexist with real node refs."""
        mapping = {
            "tag": "ctx.run_id",
            "data": "analyzer.output",
            "input_val": "$input.field",
        }
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}


class TestExpressionValuesInMapping:
    """Tests for expression-valued input_mapping entries (not just plain refs)."""

    def test_arithmetic_expression_extracts_dotted_refs(self):
        """Arithmetic expression with node.field refs extracts both nodes."""
        mapping = {"total": "product.price * order.quantity"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"product", "order"}

    def test_function_call_expression_extracts_refs(self):
        """Function call expression extracts node refs."""
        mapping = {"count": "len(analyzer.items)"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_comparison_expression_extracts_refs(self):
        """Comparison expression extracts node refs."""
        mapping = {"is_valid": "analyzer.score > 0.5"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_ternary_expression_extracts_refs(self):
        """Ternary expression extracts refs from both branches."""
        mapping = {"result": "analyzer.output if order.valid else 'N/A'"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "order"}

    def test_bare_node_in_arithmetic_expression(self):
        """Bare node name in arithmetic expression is detected."""
        mapping = {"total": "analyzer + 1"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_bare_node_in_function_call(self):
        """Bare node name as function argument is detected."""
        mapping = {"count": "len(analyzer)"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer"}

    def test_mixed_dotted_and_bare_in_expression(self):
        """Mix of dotted and bare refs in one expression."""
        mapping = {"val": "analyzer + fetcher.count"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "fetcher"}

    def test_builtins_not_extracted(self):
        """Builtin function names are not treated as node refs."""
        mapping = {"val": "len(items) + max(values)"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == set()

    def test_coalesce_expression_extracts_refs(self):
        """coalesce() with multiple node refs extracts all."""
        mapping = {"val": "coalesce(analyzer.score, fetcher.default_score)"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "fetcher"}

    def test_boolean_expression_extracts_refs(self):
        """Boolean expression extracts refs."""
        mapping = {"check": "analyzer.done and order.valid"}
        refs = extract_refs_from_mapping(mapping, KNOWN_NODES)
        assert refs == {"analyzer", "order"}


class TestExtractInputRefsFromMapping:
    """Tests for extract_input_refs_from_mapping — collecting $input.X field names."""

    def test_direct_input_ref(self):
        """Direct $input.field reference extracts the field name."""
        mapping = {"conversation_id": "$input.conversation_id"}
        assert extract_input_refs_from_mapping(mapping) == {"conversation_id"}

    def test_multiple_input_refs(self):
        """Multiple $input references across different values."""
        mapping = {
            "conv_id": "$input.conversation_id",
            "load": "$input.load_id",
        }
        assert extract_input_refs_from_mapping(mapping) == {"conversation_id", "load_id"}

    def test_embedded_input_ref_in_expression(self):
        """$input.field embedded in a coalesce expression."""
        mapping = {"rate": "coalesce($input.rate, 0)"}
        assert extract_input_refs_from_mapping(mapping) == {"rate"}

    def test_multiple_input_refs_in_single_value(self):
        """Multiple $input refs in one expression value."""
        mapping = {"combined": "$input.rate + $input.margin"}
        assert extract_input_refs_from_mapping(mapping) == {"rate", "margin"}

    def test_no_input_refs(self):
        """Node references (not $input) are not extracted."""
        mapping = {"data": "analyzer.output", "other": "fetcher.response"}
        assert extract_input_refs_from_mapping(mapping) == set()

    def test_non_string_values_skipped(self):
        """Non-string values are safely skipped."""
        mapping = {"count": 42, "conv": "$input.conversation_id"}
        assert extract_input_refs_from_mapping(mapping) == {"conversation_id"}

    def test_empty_mapping(self):
        """Empty mapping returns empty set."""
        assert extract_input_refs_from_mapping({}) == set()

    def test_plain_dollar_input_no_field(self):
        """Bare $input without a field name extracts nothing."""
        mapping = {"all": "$input"}
        assert extract_input_refs_from_mapping(mapping) == set()

    def test_list_values_extract_input_refs(self):
        """$input.X refs inside list-valued entries are extracted."""
        mapping = {"items": ["$input.order_id", "$input.customer_id", "node.field"]}
        assert extract_input_refs_from_mapping(mapping) == {"order_id", "customer_id"}

    def test_mixed_list_and_string_values(self):
        """Both string and list values are processed."""
        mapping = {
            "single": "$input.name",
            "multi": ["$input.age", "literal"],
        }
        assert extract_input_refs_from_mapping(mapping) == {"name", "age"}

    def test_nested_input_ref_captures_first_segment(self):
        """$input.a.b.c captures only the top-level field 'a' (by design).

        The input_schema validator keys on the flat, top-level input
        field, so a nested path is validated by its first segment.
        Runtime resolution still walks the full depth.
        """
        mapping = {"x": "$input.a.b.c"}
        assert extract_input_refs_from_mapping(mapping) == {"a"}


class TestExtractRefsFromSpec:
    """Tests for extract_refs_from_spec — the deep-scan shared by builder and validator."""

    def test_human_message_top_level(self):
        """Canonical llm_node field is scanned (was a known inference gap)."""
        spec = {"human_message": "Analyze {{fetcher.load.description}}"}
        assert extract_refs_from_spec(spec, KNOWN_NODES) == {"fetcher"}

    def test_bare_whole_node_ref(self):
        """{{node}} without a field references the whole output (conversation form)."""
        spec = {"conversation": "{{analyzer}}"}
        assert extract_refs_from_spec(spec, KNOWN_NODES) == {"analyzer"}

    def test_custom_field_any_name(self):
        """Custom node spec fields get inference without any declaration."""
        spec = {"subject_template": "Order {{order.id}} shipped"}
        assert extract_refs_from_spec(spec, KNOWN_NODES) == {"order"}

    def test_deeply_nested_strings(self):
        """Refs are found in dicts and lists at any depth."""
        spec = {
            "template": {
                "messages": [
                    {"role": "user", "content": "Check {{product.sku}}"},
                ],
            },
            "options": {"footer": {"text": "by {{cleanup.report}}"}},
        }
        assert extract_refs_from_spec(spec, KNOWN_NODES) == {"product", "cleanup"}

    def test_unknown_names_ignored(self):
        """Aliases and namespaces never become edges."""
        spec = {
            "human_message": "{{email_subject}} {{input.carrier}} {{state.round}}",
        }
        assert extract_refs_from_spec(spec, KNOWN_NODES) == set()

    def test_expression_fields_still_scanned(self):
        """Bare expression grammar in framework fields keeps working."""
        spec = {
            "when": "analyzer.done == True",
            "input_mapping": {"data": "fetcher.result"},
            "expressions": {"total": "order.price * 2"},
        }
        assert extract_refs_from_spec(spec, KNOWN_NODES) == {"analyzer", "fetcher", "order"}

    def test_expression_grammar_not_applied_to_arbitrary_strings(self):
        """Bare node.field in prose (non-framework field) creates no edge."""
        spec = {"description": "reads from analyzer.output downstream"}
        assert extract_refs_from_spec(spec, KNOWN_NODES) == set()


class TestExtractJinjaHeadNames:
    """Tests for extract_jinja_head_names — feeds the validator typo lint."""

    def test_collects_dotted_and_bare(self):
        text = "{{analyzer.result}} and {{whole_node}} and {{ spaced.x }}"
        assert extract_jinja_head_names(text) == {"analyzer", "whole_node", "spaced"}

    def test_no_match_without_braces(self):
        assert extract_jinja_head_names("plain analyzer.result text") == set()
