#!/usr/bin/env python3
"""Entry point for hexAI CLI tools."""

from hexai.utils.features import FeatureManager, FeatureMissingError

try:
    FeatureManager.require_feature("cli")
except FeatureMissingError as e:
    raise FeatureMissingError(
        f"{e.message}\n\nTo enable CLI support, install:\n    pip install hexdag[cli]\n"
    ) from e

if __name__ == "__main__":
    # Import only when run as main to avoid circular import warnings
    from hexai.cli.simple_pipeline_cli import main

    main()
