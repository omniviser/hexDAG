"""Tests for the compiler front door: hexdag.compiler.staged.compile().

One entry point for build/validate/lenient; validate mode never raises;
consumers (CLI/API/MCP/Studio) all sit on top of this surface.
"""

import pytest

from hexdag.compiler.staged import CompileResult, compile
from hexdag.kernel.exceptions import YamlPipelineBuilderError

VALID_PIPELINE = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: compile-api-test
spec:
  nodes:
    - kind: function_node
      metadata:
        name: parse
      spec:
        fn: json.loads

    - kind: function_node
      metadata:
        name: report
      spec:
        fn: json.dumps
      dependencies: [parse]
"""

INVALID_PIPELINE = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: broken
spec:
  nodes:
    - kind: function_node
      metadata:
        name: consumer
      spec:
        fn: json.loads
        dependencies: [does_not_exist]
"""

BROKEN_YAML = "invalid: yaml: :"


class TestValidateMode:
    """mode='validate' runs stages up to validation and never raises."""

    def test_valid_pipeline_ok(self):
        result = compile(VALID_PIPELINE, mode="validate")
        assert result.ok
        assert result.graph is None  # validate mode never lowers
        assert result.document is not None
        assert result.node_names == ["parse", "report"]

    def test_invalid_pipeline_returns_diagnostics(self):
        result = compile(INVALID_PIPELINE, mode="validate")
        assert not result.ok
        assert any("does_not_exist" in e for e in result.errors)

    def test_broken_yaml_never_raises(self):
        result = compile(BROKEN_YAML, mode="validate")
        assert not result.ok
        assert result.error_type == "YAMLError"
        assert any("YAML syntax error" in e for e in result.errors)

    def test_missing_include_never_raises(self, tmp_path):
        content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: broken-include
spec:
  nodes:
    - "!include": ./missing.yaml
"""
        result = compile(content, mode="validate", base_path=tmp_path)
        assert not result.ok
        assert any("missing.yaml" in e for e in result.errors)

    def test_path_source_reads_file_and_defaults_base_path(self, tmp_path):
        included = tmp_path / "fragment.yaml"
        included.write_text(
            "- kind: function_node\n"
            "  metadata:\n"
            "    name: from_fragment\n"
            "  spec:\n"
            "    fn: json.loads\n"
        )
        pipeline = tmp_path / "pipeline.yaml"
        pipeline.write_text(
            "apiVersion: hexdag/v1\n"
            "kind: Pipeline\n"
            "metadata:\n"
            "  name: with-include\n"
            "spec:\n"
            "  nodes:\n"
            '    - "!include": ./fragment.yaml\n'
        )
        result = compile(pipeline, mode="validate")
        assert result.ok
        assert result.node_names == ["from_fragment"]

    def test_severity_mapping(self):
        # Redundant explicit dependencies produce a warning-class finding today;
        # errors and warnings must land in the right severity buckets.
        result = compile(INVALID_PIPELINE, mode="validate")
        assert all(d.severity == "error" for d in result.diagnostics if d.message in result.errors)


class TestBuildMode:
    """mode='build' matches YamlPipelineBuilder behavior."""

    def test_build_returns_graph_and_config(self):
        result = compile(VALID_PIPELINE, mode="build")
        assert result.ok
        assert result.graph is not None
        assert len(result.graph) == 2
        assert result.config is not None
        assert result.config.metadata.get("name") == "compile-api-test"

    def test_build_parity_with_builder(self):
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        graph, config = YamlPipelineBuilder().build_from_yaml_string(VALID_PIPELINE)
        result = compile(VALID_PIPELINE, mode="build")
        assert sorted(n.name for n in result.graph.values()) == sorted(
            n.name for n in graph.values()
        )
        assert result.config.metadata == config.metadata

    def test_build_raises_by_default(self):
        with pytest.raises(YamlPipelineBuilderError):
            compile(INVALID_PIPELINE, mode="build")

    def test_build_raise_on_error_false_returns_diagnostics(self):
        result = compile(INVALID_PIPELINE, mode="build", raise_on_error=False)
        assert not result.ok
        assert result.error_type == "YamlPipelineBuilderError"
        assert result.graph is None


class TestLenientMode:
    """Structure-only validation for contexts without files or env vars."""

    def test_include_stripped_with_warning(self):
        content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: lenient-test
spec:
  nodes:
    - "!include": ./somewhere.yaml
"""
        result = compile(content, lenient=True)
        assert result.ok
        assert any("somewhere.yaml" in w for w in result.warnings)

    def test_env_vars_not_required(self):
        content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: lenient-env
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        api_key: ${DEFINITELY_NOT_SET_VAR}
  nodes: []
"""
        result = compile(content, lenient=True)
        assert result.ok

    def test_non_dict_yaml_rejected(self):
        result = compile("- just\n- a\n- list\n", lenient=True)
        assert not result.ok
        assert result.error_type == "ParseError"


class TestCompileResult:
    def test_ok_reflects_error_severity_only(self):
        result = CompileResult()
        assert result.ok
        result = compile(VALID_PIPELINE, mode="validate")
        # warnings alone don't flip ok
        assert result.ok or result.errors


class TestApiLayerAdapter:
    """hexdag.api.validation is a thin JSON adapter over compile()."""

    def test_valid_keeps_old_keys_and_adds_diagnostics(self):
        from hexdag.api.validation import validate

        result = validate(VALID_PIPELINE)
        assert result["valid"] is True
        assert result["message"] == "Pipeline is valid"
        assert result["node_count"] == 2
        assert set(result["nodes"]) == {"parse", "report"}
        assert "diagnostics" in result

    def test_invalid_full_mode_error_keys(self):
        from hexdag.api.validation import validate

        result = validate(INVALID_PIPELINE)
        assert result["valid"] is False
        assert "error" in result
        assert result["error_type"] == "YamlPipelineBuilderError"

    def test_lenient_yaml_error_type(self):
        from hexdag.api.validation import validate

        result = validate(BROKEN_YAML, lenient=True)
        assert result["valid"] is False
        assert "YAML" in result["error_type"]

    def test_diagnostics_shape(self):
        from hexdag.api.validation import validate

        result = validate(INVALID_PIPELINE, lenient=True)
        assert result["diagnostics"], "expected structured diagnostics"
        diag = result["diagnostics"][0]
        assert {"code", "severity", "message", "line", "column"} <= diag.keys()


class TestFragmentMode:
    """Fragments (root-level node lists) validate standalone."""

    def test_node_list_fragment_validates(self):
        content = """\
- kind: function_node
  metadata:
    name: frag_node
  spec:
    fn: json.loads
"""
        result = compile(content, fragment=True)
        assert result.ok
        assert result.node_names == ["frag_node"]

    def test_external_refs_become_warnings(self):
        content = """\
- kind: function_node
  metadata:
    name: frag_node
  spec:
    fn: json.loads
    dependencies: [defined_in_including_pipeline]
"""
        result = compile(content, fragment=True)
        assert result.ok, result.errors
        assert any("external to this fragment" in w for w in result.warnings)

    def test_internal_errors_stay_errors(self):
        content = """\
- kind: function_node
  metadata:
    name: frag_node
"""
        result = compile(content, fragment=True)
        # function_node without fn is a genuine fragment-internal error
        assert not result.ok

    def test_mapping_fragment_is_info_only(self):
        result = compile("llm:\n  adapter: mock\n", fragment=True)
        assert result.ok
        assert any("Mapping fragment" in s for s in result.suggestions)

    def test_full_manifest_validates_normally(self):
        content = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: not-a-fragment
spec:
  nodes: []
"""
        result = compile(content, fragment=True)
        assert result.ok


class TestLocatedDiagnostics:
    """Parse/include failures carry file:line locations."""

    def test_include_error_location(self, tmp_path):
        pipeline = tmp_path / "p.yaml"
        pipeline.write_text(
            "apiVersion: hexdag/v1\n"
            "kind: Pipeline\n"
            "metadata:\n"
            "  name: x\n"
            "spec:\n"
            "  nodes:\n"
            '    - "!include": ./missing.yaml\n'
        )
        result = compile(pipeline, mode="validate")
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.loc is not None
        assert diag.loc.line == 7

    def test_yaml_syntax_error_location(self):
        result = compile("kind: Pipeline\nmetadata:\n  bad: [unclosed\n", mode="validate")
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.loc is not None
        assert diag.loc.line is not None


class TestValidateGreenImpliesBuildGreen:
    """The one-validation invariant: what validates, builds.

    Checks migrated from build-time raises must be flagged at validate
    time (no exception), and pipelines that validate green must build.
    """

    RESERVED_NAME = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: reserved
spec:
  nodes:
    - kind: function_node
      metadata:
        name: input
      spec:
        fn: json.loads
"""

    BROKEN_SWITCH = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: broken-switch
spec:
  nodes:
    - kind: data_node
      metadata:
        name: seed
      spec:
        output: {x: 1}
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        route_downstream: true
        branches:
          - condition: "seed.x == 1"
            action: does_not_exist
"""

    MACRO_OVERLAP = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: macro-overlap
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: agent
      spec:
        macro: hexdag.stdlib.macros.ReasoningAgentMacro
        config: {question: a}
        inputs: {question: b}
"""

    @pytest.mark.parametrize(
        ("yaml_content", "marker"),
        [
            (RESERVED_NAME, "reserved expression namespaces"),
            (BROKEN_SWITCH, "Route target 'does_not_exist'"),
            (MACRO_OVERLAP, "both 'config' and 'inputs'"),
        ],
    )
    def test_migrated_build_errors_flagged_at_validate(self, yaml_content, marker):
        result = compile(yaml_content, mode="validate")
        assert not result.ok
        assert any(marker in e for e in result.errors), result.errors

    @pytest.mark.parametrize("yaml_content", [RESERVED_NAME, BROKEN_SWITCH, MACRO_OVERLAP])
    def test_migrated_build_errors_never_raise_in_validate(self, yaml_content):
        compile(yaml_content, mode="validate")  # must not raise

    def test_valid_corpus_validate_green_implies_build_green(self):
        corpus = [
            VALID_PIPELINE,
            """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: switch-ok
spec:
  nodes:
    - kind: data_node
      metadata:
        name: seed
      spec:
        output: {x: 1}
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        route_downstream: true
        branches:
          - condition: "seed.x == 1"
            action: handle
    - kind: function_node
      metadata:
        name: handle
      spec:
        fn: json.dumps
        wait_for: [router]
""",
        ]
        for content in corpus:
            validated = compile(content, mode="validate")
            assert validated.ok, validated.errors
            built = compile(content, mode="build")  # must not raise
            assert built.graph is not None
