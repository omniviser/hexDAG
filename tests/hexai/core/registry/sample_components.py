"""Sample components for testing the bootstrap system."""

from hexai.core.registry.decorators import adapter, node, tool


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


@adapter(name="sample_adapter")
class SampleAdapter:
    """Sample adapter implementation."""

    pass


# Don't define register_components - let the default discovery work
# If we wanted custom logic, we could do:
# def register_components(registry, namespace):
#     """Custom registration hook."""
#     # Custom logic here
#     pass
