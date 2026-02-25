"""Tests for reference_resolver — extracting node refs from mappings, expressions, templates."""

from hexdag.compiler.reference_resolver import (
    extract_refs_from_expressions,
    extract_refs_from_mapping,
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
