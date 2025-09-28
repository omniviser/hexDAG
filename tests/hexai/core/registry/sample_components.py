"""Sample components for testing the bootstrap system."""

import sys
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from hexai.core.registry.decorators import adapter, node, tool
from hexai.core.registry.discovery import discover_components
from hexai.core.registry.models import (
    ClassComponent,
    ComponentMetadata,
    ComponentType,
)


# Define a test port using Protocol
@runtime_checkable
class TestPort(Protocol):
    """A test port interface."""

    @abstractmethod
    def process(self, data: str) -> str:
        """Process data."""
        ...


@node(name="sample_node", description="A sample node for testing")
class SampleNode:
    """Sample node implementation."""

    def execute(self, data):
        return data


@tool(name="sample_tool", description="A sample tool")
class SampleTool:
    """Sample tool implementation."""

    def run(self):
        return "tool result"


@adapter(implements_port="test_port", name="sample_adapter")
class SampleAdapter:
    """Sample adapter that implements TestPort."""

    def process(self, data: str) -> str:
        """Process data implementation."""
        return f"processed: {data}"


def register_components(registry, namespace):
    """Custom registration hook to register the test port first."""

    # Register the test port that our adapter implements
    port_meta = ComponentMetadata(
        name="test_port",
        component_type=ComponentType.PORT,
        component=ClassComponent(value=TestPort),
        namespace=namespace,
    )

    # Manually register the port
    if namespace not in registry._components:
        registry._components[namespace] = {}
    registry._components[namespace]["test_port"] = port_meta

    # Discover and register the decorated components from this module
    module = sys.modules[__name__]
    components = discover_components(module)

    count = 0
    for _, component in components:
        if not hasattr(component, "_hexdag_type"):
            continue

        # Extract attributes from component
        component_type = component._hexdag_type
        component_name = component._hexdag_name
        component_subtype = getattr(component, "_hexdag_subtype", None)
        component_description = getattr(component, "_hexdag_description", "")

        try:
            registry.register(
                name=component_name,
                component=component,
                component_type=component_type,
                namespace=namespace,
                privileged=(namespace == "core"),
                subtype=component_subtype,
                description=component_description,
            )
            count += 1
        except Exception:
            # Skip components that fail to register
            pass

    return count
