"""Preprocessing plugins for the YAML pipeline builder.

These plugins transform YAML configuration before graph building:
- IncludePreprocessPlugin: Resolve !include directives
- EnvironmentVariablePlugin: Resolve ${VAR} and ${VAR:default}
- TemplatePlugin: Jinja2 template rendering
"""

from hexdag.compiler.preprocessing.env_vars import EnvironmentVariablePlugin
from hexdag.compiler.preprocessing.include import IncludePreprocessPlugin
from hexdag.compiler.preprocessing.template import TemplatePlugin

__all__ = [
    "EnvironmentVariablePlugin",
    "IncludePreprocessPlugin",
    "TemplatePlugin",
]
