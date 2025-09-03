"""Example demonstrating the new convention-over-configuration registry system."""

from hexai.core.registry import node, registry, tool


# Example 1: Simple component with decorator - auto-registered on import
@node(tags={"example", "processing"})
class DataProcessorNode:
    """Processes incoming data streams."""

    async def execute(self, data: dict) -> dict:
        """Process the data."""
        return {"processed": data}


# Example 2: Tool component with minimal configuration
@tool()
class WebScraperTool:
    """Scrapes web pages for content."""

    def scrape(self, url: str) -> str:
        """Scrape content from URL."""
        return f"Content from {url}"


# Example 3: Component with explicit configuration
@node(
    name="custom_analyzer",
    version="1.0.0",
    tags={"nlp", "analysis"},
    dependencies={"tokenizer", "sentiment_model"},
)
class TextAnalyzer:
    """Analyzes text for various NLP tasks."""

    async def execute(self, text: str) -> dict:
        """Analyze the text."""
        return {"sentiment": "positive", "entities": []}


def demo_autodiscovery():
    """Demonstrate automatic component discovery."""

    print("=== Automatic Component Discovery Demo ===\n")

    # Register namespace for examples
    registry.register_namespace("examples")

    # First, auto-discover components from this module
    from hexai.core.registry.autodiscover import autodiscover_components

    count = autodiscover_components(registry, package="__main__", namespace="examples")
    print(f"Auto-discovered {count} components from decorators\n")

    # List registered components
    print("Registered components:")
    components = registry.list_components(component_type="node", namespace="examples")
    for name in components:
        metadata = registry.get_metadata(name, component_type="node", namespace="examples")
        print(f"  - {name} (node): {metadata.description}")

    components = registry.list_components(component_type="tool", namespace="examples")
    for name in components:
        metadata = registry.get_metadata(name, component_type="tool", namespace="examples")
        print(f"  - {name} (tool): {metadata.description}")

    # Try to get a component
    try:
        processor_class = registry.get(
            "data_processor_node", component_type="node", namespace="examples"
        )
        print(f"\nRetrieved component class: {processor_class}")
    except KeyError as e:
        print(f"\nNote: {e}")


def demo_plugin_creation():
    """Demonstrate plugin scaffolding."""

    print("\n=== Plugin Scaffolding Demo ===\n")

    # This would create a complete plugin structure
    print("To create a new plugin, run:")
    print("  path = create_plugin_scaffold('my_awesome_plugin', './plugins')")
    print("\nThis creates:")
    print("  - plugins/my_awesome_plugin/")
    print("    - __init__.py (with register function)")
    print("    - components/")
    print("      - nodes/my_awesome_plugin_node.py")
    print("      - tools/my_awesome_plugin_tool.py")
    print("    - pyproject.toml (with entry points)")
    print("    - README.md")


def demo_convention_discovery():
    """Demonstrate convention-based discovery."""

    print("\n=== Convention-Based Discovery Demo ===\n")

    print("Components are discovered from conventional paths:")
    print("  - nodes/        → automatically registered as nodes")
    print("  - tools/        → automatically registered as tools")
    print("  - adapters/     → automatically registered as adapters")
    print("  - plugins/      → automatically loaded as plugins")
    print("\nNo configuration needed - just follow the conventions!")


if __name__ == "__main__":
    # Run the demos
    demo_autodiscovery()
    demo_plugin_creation()
    demo_convention_discovery()

    print("\n=== Benefits of Convention over Configuration ===\n")
    print("1. Zero boilerplate - just decorate your classes")
    print("2. Automatic discovery - no manual registration needed")
    print("3. Consistent structure - everyone follows the same patterns")
    print("4. Plugin-friendly - easy to extend with new components")
    print("5. Type-safe - full type hints and validation")
