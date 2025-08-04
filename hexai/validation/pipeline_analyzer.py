"""Pipeline Type Safety Analyzer - Static type checking for pipelines at compile-time.

This module provides comprehensive type safety analysis for pipelines without requiring
runtime execution. It uses the validation framework to verify type compatibility across
all nodes and connections.
"""

from dataclasses import dataclass
from enum import Enum

from hexai.core.domain.dag import DirectedGraph
from hexai.validation import ValidationStrategy, can_convert_schema


class TypeSafetyLevel(Enum):
    """Type safety levels for pipeline analysis."""

    PERFECT = "perfect"  # All types specified and compatible
    GOOD = "good"  # Most types specified, minor issues
    PARTIAL = "partial"  # Some types missing but compatible where specified
    UNSAFE = "unsafe"  # Type mismatches or critical issues


@dataclass
class TypeCompatibilityIssue:
    """Represents a type compatibility issue in the pipeline."""

    severity: str  # "error", "warning", "info"
    source_node: str
    target_node: str
    source_type: type | None
    target_type: type | None
    message: str
    suggestion: str | None = None


@dataclass
class NodeTypeInfo:
    """Type information for a single node."""

    node_name: str
    input_type: type | None
    output_type: type | None
    has_input_mapping: bool
    dependencies: list[str]
    type_specified: bool  # Whether types are explicitly specified


@dataclass
class PipelineTypeAnalysis:
    """Complete type analysis result for a pipeline."""

    pipeline_name: str
    safety_level: TypeSafetyLevel
    total_nodes: int
    typed_nodes: int
    issues: list[TypeCompatibilityIssue]
    node_info: list[NodeTypeInfo]
    type_coverage: float  # Percentage of nodes with explicit types
    is_type_safe: bool

    @property
    def summary(self) -> str:
        """Human-readable summary of the analysis."""
        return (
            f"Pipeline '{self.pipeline_name}': {self.safety_level.value.upper()} "
            f"({self.typed_nodes}/{self.total_nodes} nodes typed, "
            f"{len([i for i in self.issues if i.severity == 'error'])} errors)"
        )


class PipelineTypeAnalyzer:
    """Static type safety analyzer for pipelines."""

    def __init__(self, validation_strategy: ValidationStrategy = ValidationStrategy.COERCE):
        """Initialize the analyzer.

        Args:
        ----
        validation_strategy: Strategy to use for type compatibility checking
        """
        self.validation_strategy = validation_strategy

    def analyze_pipeline(
        self, graph: DirectedGraph, pipeline_name: str = "unnamed"
    ) -> PipelineTypeAnalysis:
        """Perform comprehensive type safety analysis on a pipeline.

        Args
        ----
        graph: The DirectedGraph to analyze
        pipeline_name: Name of the pipeline for reporting

        Returns
        -------
        Complete type analysis results
        """
        # Collect node type information
        node_info = self._collect_node_type_info(graph)

        # Analyze type compatibility between connected nodes
        issues = self._analyze_type_compatibility(graph, node_info)

        # Calculate type coverage and safety metrics
        typed_nodes = sum(1 for node in node_info if node.type_specified)
        type_coverage = (typed_nodes / len(node_info)) * 100 if node_info else 0

        # Determine overall safety level
        safety_level = self._calculate_safety_level(issues, type_coverage)

        # Determine if pipeline is type safe
        error_count = len([i for i in issues if i.severity == "error"])
        is_type_safe = error_count == 0

        return PipelineTypeAnalysis(
            pipeline_name=pipeline_name,
            safety_level=safety_level,
            total_nodes=len(node_info),
            typed_nodes=typed_nodes,
            issues=issues,
            node_info=node_info,
            type_coverage=type_coverage,
            is_type_safe=is_type_safe,
        )

    def _collect_node_type_info(self, graph: DirectedGraph) -> list[NodeTypeInfo]:
        """Collect type information for all nodes in the graph."""
        node_info = []

        for node_name, node_spec in graph.nodes.items():
            has_input_mapping = "input_mapping" in node_spec.params
            type_specified = node_spec.in_type is not None or node_spec.out_type is not None

            node_info.append(
                NodeTypeInfo(
                    node_name=node_name,
                    input_type=node_spec.in_type,
                    output_type=node_spec.out_type,
                    has_input_mapping=has_input_mapping,
                    dependencies=list(node_spec.deps),
                    type_specified=type_specified,
                )
            )

        return node_info

    def _analyze_type_compatibility(
        self, graph: DirectedGraph, node_info: list[NodeTypeInfo]
    ) -> list[TypeCompatibilityIssue]:
        """Analyze type compatibility between all connected nodes."""
        issues = []
        node_info_map = {node.node_name: node for node in node_info}

        for node_name, node_spec in graph.nodes.items():
            node = node_info_map[node_name]

            # Skip nodes with input mapping - they handle their own conversion
            if node.has_input_mapping:
                issues.append(
                    TypeCompatibilityIssue(
                        severity="info",
                        source_node="<various>",
                        target_node=node_name,
                        source_type=None,
                        target_type=node.input_type,
                        message=f"Node '{node_name}' uses input mapping for type conversion",
                        suggestion="Ensure input mapping correctly transforms data types",
                    )
                )
                continue

            # Check each dependency connection
            for dep_name in node_spec.deps:
                dep_node = node_info_map[dep_name]

                # Check if both nodes have type specifications
                if node.input_type is None:
                    issues.append(
                        TypeCompatibilityIssue(
                            severity="warning",
                            source_node=dep_name,
                            target_node=node_name,
                            source_type=dep_node.output_type,
                            target_type=None,
                            message=f"Node '{node_name}' has no input type specified",
                            suggestion="Add in_type specification for better type safety",
                        )
                    )
                    continue

                if dep_node.output_type is None:
                    issues.append(
                        TypeCompatibilityIssue(
                            severity="warning",
                            source_node=dep_name,
                            target_node=node_name,
                            source_type=None,
                            target_type=node.input_type,
                            message=f"Node '{dep_name}' has no output type specified",
                            suggestion="Add out_type specification for better type safety",
                        )
                    )
                    continue

                # Check type compatibility using validation framework
                if not can_convert_schema(dep_node.output_type, node.input_type):
                    issues.append(
                        TypeCompatibilityIssue(
                            severity="error",
                            source_node=dep_name,
                            target_node=node_name,
                            source_type=dep_node.output_type,
                            target_type=node.input_type,
                            message=(
                                f"Type mismatch: '{dep_name}' outputs "
                                f"{dep_node.output_type.__name__} but '{node_name}' expects "
                                f"{node.input_type.__name__}"
                            ),
                            suggestion=(
                                "Add a converter or use input_mapping to transform the data, "
                                "or register a custom converter for this type pair"
                            ),
                        )
                    )
                else:
                    # Types are compatible - add info message if conversion is needed
                    if dep_node.output_type != node.input_type:
                        issues.append(
                            TypeCompatibilityIssue(
                                severity="info",
                                source_node=dep_name,
                                target_node=node_name,
                                source_type=dep_node.output_type,
                                target_type=node.input_type,
                                message=(
                                    f"Automatic conversion: '{dep_name}' outputs "
                                    f"{dep_node.output_type.__name__} → "
                                    f"'{node_name}' expects {node.input_type.__name__}"
                                ),
                                suggestion=(
                                    "Conversion will happen automatically via validation framework"
                                ),
                            )
                        )

        return issues

    def _calculate_safety_level(
        self, issues: list[TypeCompatibilityIssue], type_coverage: float
    ) -> TypeSafetyLevel:
        """Calculate overall type safety level based on issues and coverage."""
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])

        if error_count > 0:
            return TypeSafetyLevel.UNSAFE
        elif warning_count == 0 and type_coverage >= 90:
            return TypeSafetyLevel.PERFECT
        elif warning_count <= 2 and type_coverage >= 70:
            return TypeSafetyLevel.GOOD
        else:
            return TypeSafetyLevel.PARTIAL

    def validate_pipeline_types(self, graph: DirectedGraph, pipeline_name: str = "unnamed") -> bool:
        """Quick type safety validation - returns True if pipeline is type safe.

        Args
        ----
        graph: The DirectedGraph to validate
        pipeline_name: Name of the pipeline for error reporting

        Returns
        -------
        True if pipeline is type safe, False otherwise
        """
        analysis = self.analyze_pipeline(graph, pipeline_name)
        return analysis.is_type_safe


# Convenience functions
def analyze_pipeline_types(
    graph: DirectedGraph,
    pipeline_name: str = "unnamed",
    validation_strategy: ValidationStrategy = ValidationStrategy.COERCE,
) -> PipelineTypeAnalysis:
    """Analyze pipeline type safety using the validation framework.

    Args
    ----
    graph: The DirectedGraph to analyze
    pipeline_name: Name of the pipeline
    validation_strategy: Strategy for type compatibility checking

    Returns
    -------
    Complete type analysis results
    """
    analyzer = PipelineTypeAnalyzer(validation_strategy)
    return analyzer.analyze_pipeline(graph, pipeline_name)


def is_pipeline_type_safe(
    graph: DirectedGraph,
    pipeline_name: str = "unnamed",
    validation_strategy: ValidationStrategy = ValidationStrategy.COERCE,
) -> bool:
    """Quick check if pipeline is type safe.

    Args
    ----
    graph: The DirectedGraph to check
    pipeline_name: Name of the pipeline
    validation_strategy: Strategy for type compatibility checking

    Returns
    -------
    True if pipeline is type safe, False otherwise
    """
    analyzer = PipelineTypeAnalyzer(validation_strategy)
    return analyzer.validate_pipeline_types(graph, pipeline_name)
