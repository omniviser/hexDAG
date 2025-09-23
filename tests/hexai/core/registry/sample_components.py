"""Sample components for testing the bootstrap system."""

import sys

from hexai.core.registry.decorators import adapter, node, tool
from hexai.core.registry.discovery import discover_components
from hexai.core.registry.models import (
    ClassComponent,
    ComponentMetadata,
    ComponentType,
    PortMetadata,
)


# Define a test port that the adapter will implement
class TestPort:
    """A test port interface."""

    def process(self, data: str) -> str:
        """Process data."""
        raise NotImplementedError


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
        port_metadata=PortMetadata(
            protocol_class=TestPort,
            required_methods=["process"],
            optional_methods=[],
        ),
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
        if not hasattr(component, "__hexdag_metadata__"):
            continue
        metadata = component.__hexdag_metadata__

        try:
            registry.register(
                name=metadata.name,
                component=component,
                component_type=metadata.type,
                namespace=namespace,
                privileged=(namespace == "core"),
                subtype=metadata.subtype,
                description=metadata.description,
                adapter_metadata=metadata.adapter_metadata
                if hasattr(metadata, "adapter_metadata")
                else None,
            )
            count += 1
        except Exception:
            # Skip components that fail to register
            pass

    return count
