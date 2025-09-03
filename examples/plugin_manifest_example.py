"""Example showing how plugins use entry points as their manifest.

Entry points serve as the plugin manifest system - no manual configuration needed!
When users install a plugin via pip, the entry point is automatically registered.
"""

# =============================================================================
# PLUGIN STRUCTURE EXAMPLE
# =============================================================================

"""
my_nlp_plugin/
├── pyproject.toml          # Plugin manifest via entry points
├── src/
│   └── my_nlp_plugin/
│       ├── __init__.py     # Plugin initialization
│       ├── nodes/          # Node components
│       │   ├── __init__.py
│       │   └── sentiment.py
│       └── tools/          # Tool components
│           ├── __init__.py
│           └── tokenizer.py
"""

# =============================================================================
# 1. PLUGIN MANIFEST (pyproject.toml)
# =============================================================================

PLUGIN_MANIFEST = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "hexai-nlp-plugin"
version = "1.0.0"
description = "NLP components for hexDAG"
dependencies = [
    "hexai>=0.1.0",
]

# THIS IS THE MANIFEST! Entry points declare the plugin to hexDAG
[project.entry-points."hexai.plugins"]
nlp = "my_nlp_plugin:register"  # Calls register() function on load
"""

# =============================================================================
# 2. PLUGIN INITIALIZATION (my_nlp_plugin/__init__.py)
# =============================================================================


def example_plugin_init():
    """Example of a plugin's __init__.py file."""
    code = '''
"""NLP Plugin for hexDAG."""

def register():
    """Entry point called by hexDAG to register this plugin.

    This function is called when hexDAG discovers the plugin via entry points.
    It should import all modules containing decorated components.
    """
    # Import modules with decorated components
    # The decorators will auto-register with the registry
    from . import nodes
    from . import tools

    # Optional: Log that plugin is loaded
    import logging
    logger = logging.getLogger(__name__)
    logger.info("NLP plugin registered successfully")
'''
    return code


# =============================================================================
# 3. PLUGIN COMPONENTS (my_nlp_plugin/nodes/sentiment.py)
# =============================================================================


def example_plugin_component():
    """Example of a plugin component using decorators."""
    code = '''
"""Sentiment analysis node for NLP plugin."""

from hexai.core.registry import node

@node(
    namespace='nlp',  # Plugin namespace
    tags={'nlp', 'sentiment', 'analysis'},
    version='1.0.0'
)
class SentimentAnalyzer:
    """Analyzes sentiment of text."""

    def __init__(self):
        """Initialize sentiment analyzer."""
        # Load model, etc.
        pass

    async def execute(self, text: str) -> dict:
        """Analyze sentiment of text.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        dict
            Sentiment scores
        """
        # Simplified example
        return {
            'sentiment': 'positive',
            'confidence': 0.95,
            'scores': {
                'positive': 0.95,
                'negative': 0.03,
                'neutral': 0.02
            }
        }
'''
    return code


# =============================================================================
# 4. HOW IT WORKS
# =============================================================================


def demonstrate_plugin_discovery():
    """Show how hexDAG discovers plugins via entry points."""

    print("=" * 70)
    print("PLUGIN MANIFEST SYSTEM VIA ENTRY POINTS")
    print("=" * 70)

    print("\n1. USER INSTALLS PLUGIN:")
    print("   $ pip install hexai-nlp-plugin")
    print("   → Entry point is registered in site-packages")

    print("\n2. HEXDAG STARTS UP:")
    print("   → Registry initializes")
    print("   → Calls discover_plugins()")
    print("   → Finds 'hexai.plugins' entry points")

    print("\n3. PLUGIN IS LOADED:")
    print("   → Entry point 'nlp = my_nlp_plugin:register' is found")
    print("   → Calls my_nlp_plugin.register()")
    print("   → register() imports modules with @node, @tool decorators")
    print("   → Decorators auto-register components with registry")

    print("\n4. COMPONENTS ARE AVAILABLE:")
    print("   → registry.get('sentiment_analyzer', namespace='nlp')")
    print("   → No manual configuration needed!")

    print("\n" + "=" * 70)
    print("BENEFITS:")
    print("=" * 70)
    print("✓ Zero configuration - just pip install")
    print("✓ Standard Python pattern (used by pytest, click, flask)")
    print("✓ Automatic discovery at startup")
    print("✓ Plugin isolation via namespaces")
    print("✓ Version management via pip")


# =============================================================================
# 5. COMPARISON WITH OTHER APPROACHES
# =============================================================================


def compare_manifest_approaches():
    """Compare entry points with other manifest approaches."""

    print("\n" + "=" * 70)
    print("MANIFEST APPROACH COMPARISON")
    print("=" * 70)

    approaches = {
        "Entry Points (Our Choice)": {
            "pros": [
                "Standard Python mechanism",
                "Automatic with pip install",
                "No manual configuration",
                "Used by major frameworks",
            ],
            "cons": ["Requires proper packaging", "Not visible without tooling"],
            "example": "pyproject.toml with entry_points",
        },
        "Django INSTALLED_APPS": {
            "pros": ["Explicit list in settings", "Easy to see what's loaded", "Order control"],
            "cons": [
                "Manual configuration required",
                "User must edit settings",
                "Can forget to add",
            ],
            "example": "INSTALLED_APPS = ['my_plugin']",
        },
        "Plugin Directory Scanning": {
            "pros": ["Just drop files in folder", "No packaging needed"],
            "cons": [
                "No dependency management",
                "No versioning",
                "Security concerns",
                "Performance overhead",
            ],
            "example": "plugins/*.py auto-loaded",
        },
        "JSON/YAML Manifests": {
            "pros": ["Human readable", "Language agnostic"],
            "cons": ["Extra files to maintain", "No code completion", "Can get out of sync"],
            "example": "plugin.yaml with metadata",
        },
    }

    for approach, details in approaches.items():
        print(f"\n{approach}:")
        print("  Pros:")
        for pro in details["pros"]:
            print(f"    + {pro}")
        print("  Cons:")
        for con in details["cons"]:
            print(f"    - {con}")
        print(f"  Example: {details['example']}")


# =============================================================================
# RUN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    demonstrate_plugin_discovery()
    compare_manifest_approaches()

    print("\n" + "=" * 70)
    print("IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    print("\nThe registry now supports:")
    print("1. ✓ Core components (hardcoded list in _load_core_components)")
    print("2. ✓ Plugins (discovered via entry points - the manifest!)")
    print("3. ✓ User components (direct decorator usage)")
    print("\nNo INSTALLED_COMPONENTS setting needed - entry points ARE the manifest!")
