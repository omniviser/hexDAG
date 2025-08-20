import pytest

from hexai.utils.features import FeatureManager, FeatureMissingError


def test_cli_feature_missing(monkeypatch):
    # Simulate missing package "click" for the CLI feature
    monkeypatch.setattr(FeatureManager, "_has_package", lambda pkg: False)

    # Expect FeatureMissingError when requiring CLI feature
    with pytest.raises(FeatureMissingError) as e:
        FeatureManager.require_feature("cli")

    # Error message must hint how to install extras
    assert "pip install hexdag[cli]" in str(e.value)
