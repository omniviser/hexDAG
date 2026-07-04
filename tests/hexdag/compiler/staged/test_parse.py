"""Tests for the parse stage: values + source map + include expansion."""

import pytest

from hexdag.compiler.staged.parse import IncludeError, parse_source


def marker_line(content: str, marker: str = "# <-- HERE") -> int:
    """1-based line number of the marker comment (provenance assertions)."""
    for i, line in enumerate(content.splitlines(), start=1):
        if marker in line:
            return i
    raise AssertionError("marker not found")


class TestSourceMapCapture:
    def test_records_paths_with_lines(self):
        content = """\
kind: Pipeline
metadata:
  name: demo
spec:
  nodes:
    - kind: function_node   # <-- HERE
      metadata:
        name: parse
"""
        parsed = parse_source(content)
        loc = parsed.source_map.at(("spec", "nodes", 0, "kind"))
        assert loc is not None
        assert loc.line == marker_line(content)

    def test_longest_prefix_fallback(self):
        content = """\
kind: Pipeline
spec:
  nodes:
    - kind: function_node   # <-- HERE
"""
        parsed = parse_source(content)
        loc = parsed.source_map.at(("spec", "nodes", 0, "kind", "nonexistent", "deep"))
        assert loc is not None
        assert loc.line == marker_line(content)

    def test_multi_document_paths_are_index_prefixed(self):
        content = (
            "kind: Pipeline\nmetadata:\n  name: a\n---\nkind: Pipeline\nmetadata:\n  name: b\n"
        )
        parsed = parse_source(content)
        assert len(parsed.docs) == 2
        loc_b = parsed.source_map.at((1, "metadata", "name"))
        assert loc_b is not None
        assert loc_b.line == 7

    def test_values_match_safe_load(self):
        content = "a: 1\nb: [true, null, 2.5]\nc:\n  nested: text\n"
        parsed = parse_source(content)
        assert parsed.docs[0] == {"a": 1, "b": [True, None, 2.5], "c": {"nested": "text"}}


class TestIncludeExpansion:
    def test_dict_include(self, tmp_path):
        (tmp_path / "port.yaml").write_text("adapter: mock\nconfig: {}\n")
        content = 'spec:\n  ports:\n    llm:\n      "!include": ./port.yaml\n'
        parsed = parse_source(content, base_path=tmp_path)
        assert parsed.docs[0]["spec"]["ports"]["llm"] == {"adapter": "mock", "config": {}}

    def test_list_include_splices_fragment(self, tmp_path):
        fragment = """\
- kind: function_node
  metadata:
    name: from_fragment_one
- kind: function_node
  metadata:
    name: from_fragment_two   # <-- HERE
"""
        (tmp_path / "fragment.yaml").write_text(fragment)
        content = (
            "spec:\n"
            "  nodes:\n"
            "    - kind: function_node\n"
            "      metadata:\n"
            "        name: local\n"
            '    - "!include": ./fragment.yaml\n'
            "    - kind: function_node\n"
            "      metadata:\n"
            "        name: after\n"
        )
        parsed = parse_source(content, base_path=tmp_path)
        nodes = parsed.docs[0]["spec"]["nodes"]
        names = [n["metadata"]["name"] for n in nodes]
        assert names == ["local", "from_fragment_one", "from_fragment_two", "after"]

        # Origin of spliced elements is the FRAGMENT file and line
        loc = parsed.source_map.at(("spec", "nodes", 2, "metadata", "name"))
        assert loc is not None
        assert loc.file is not None and loc.file.endswith("fragment.yaml")
        assert loc.line == marker_line(fragment)

        # Elements after the splice keep entry-file provenance
        loc_after = parsed.source_map.at(("spec", "nodes", 3, "metadata", "name"))
        assert loc_after is not None
        assert loc_after.file is None or not loc_after.file.endswith("fragment.yaml")

    def test_anchor_include(self, tmp_path):
        (tmp_path / "anchors.yaml").write_text("first:\n  x: 1\nsecond:\n  y: 2\n")
        content = 'value:\n  "!include": ./anchors.yaml#second\n'
        parsed = parse_source(content, base_path=tmp_path)
        assert parsed.docs[0]["value"] == {"y": 2}

    def test_nested_includes_track_origin(self, tmp_path):
        (tmp_path / "inner.yaml").write_text("deep: value   # <-- HERE\n")
        (tmp_path / "outer.yaml").write_text('wrapped:\n  "!include": ./inner.yaml\n')
        content = 'root:\n  "!include": ./outer.yaml\n'
        parsed = parse_source(content, base_path=tmp_path)
        assert parsed.docs[0]["root"]["wrapped"]["deep"] == "value"
        loc = parsed.source_map.at(("root", "wrapped", "deep"))
        assert loc is not None and loc.file is not None
        assert loc.file.endswith("inner.yaml")
        assert loc.line == 1

    def test_missing_file_raises_with_location(self, tmp_path):
        content = 'spec:\n  nodes:\n    - "!include": ./missing.yaml\n'
        with pytest.raises(IncludeError) as excinfo:
            parse_source(content, base_path=tmp_path)
        assert "missing.yaml" in str(excinfo.value)
        assert excinfo.value.loc is not None
        assert excinfo.value.loc.line == 3

    def test_circular_include_detected(self, tmp_path):
        (tmp_path / "a.yaml").write_text('x:\n  "!include": ./b.yaml\n')
        (tmp_path / "b.yaml").write_text('y:\n  "!include": ./a.yaml\n')
        content = 'root:\n  "!include": ./a.yaml\n'
        with pytest.raises(IncludeError, match="Circular include"):
            parse_source(content, base_path=tmp_path)

    def test_absolute_path_rejected(self, tmp_path):
        content = 'root:\n  "!include": /etc/passwd\n'
        with pytest.raises(IncludeError, match="Absolute paths not allowed"):
            parse_source(content, base_path=tmp_path)

    def test_traversal_outside_root_rejected(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (tmp_path / "outside.yaml").write_text("secret: true\n")
        content = 'root:\n  "!include": ../outside.yaml\n'
        with pytest.raises(IncludeError, match="traverses outside project root"):
            parse_source(content, base_path=root)

    def test_root_list_fragment_parses(self, tmp_path):
        """Fragment files (root-level lists) parse without crashing."""
        content = "- kind: function_node\n  metadata:\n    name: solo\n"
        parsed = parse_source(content, base_path=tmp_path)
        assert isinstance(parsed.docs[0], list)
        assert parsed.docs[0][0]["metadata"]["name"] == "solo"


class TestParseCache:
    def test_cached_docs_are_isolated(self):
        from hexdag.compiler.staged.parse import parse_source_cached

        content = "kind: Pipeline\nspec:\n  nodes: []\n"
        first = parse_source_cached(content)
        first.docs[0]["spec"]["nodes"].append({"mutated": True})
        second = parse_source_cached(content)
        assert second.docs[0]["spec"]["nodes"] == []

    def test_include_sources_never_cached(self, tmp_path):
        from hexdag.compiler.staged.parse import parse_source_cached

        target = tmp_path / "inc.yaml"
        target.write_text("v: 1\n")
        content = 'root:\n  "!include": ./inc.yaml\n'
        first = parse_source_cached(content, base_path=tmp_path)
        assert first.docs[0]["root"]["v"] == 1
        target.write_text("v: 2\n")
        second = parse_source_cached(content, base_path=tmp_path)
        assert second.docs[0]["root"]["v"] == 2
