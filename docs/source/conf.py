# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))


project = "hexdag"
copyright = "2025, hexDAG Team"
author = "hexDAG Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]


templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Napoleon settings - disable type info in docstrings since we use type hints
napoleon_use_param = True
napoleon_use_rtype = False  # Don't show return type in docstring
napoleon_use_ivar = True
napoleon_use_keyword = True

# Autodoc typehints settings
autodoc_typehints = "description"  # Show type hints in description
autodoc_typehints_description_target = "documented"
always_document_param_types = True
typehints_fully_qualified = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
