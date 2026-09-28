import json

import pytest
from pydantic import ValidationError

from mce.config.settings import EngineSettings, MCEConfig, ProviderSettings, SnapshotSettings
from mce.reproducibility import config_fingerprint
from mce.serialization import canonical_json_bytes, canonical_json_str


def test_config_fingerprint_stable_and_relevant_changes() -> None:
    first = MCEConfig(_env_file=None)
    same = MCEConfig(_env_file=None)
    changed = MCEConfig(_env_file=None, snapshot=SnapshotSettings(cadence_ns=1))
    logging_only = MCEConfig(_env_file=None, engine=EngineSettings(log_level="DEBUG"))
    assert config_fingerprint(first) == config_fingerprint(same)
    assert config_fingerprint(first) != config_fingerprint(changed)
    assert config_fingerprint(first) == config_fingerprint(logging_only)
    assert len(config_fingerprint(first)) == 64


def test_fingerprint_ignores_input_mapping_order() -> None:
    first = MCEConfig.model_validate(
        {"provider": {"source": "file"}, "snapshot": {"cadence_ns": 1}}
    )
    second = MCEConfig.model_validate(
        {"snapshot": {"cadence_ns": 1}, "provider": {"source": "file"}}
    )
    assert config_fingerprint(first) == config_fingerprint(second)


def test_environment_nested_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCE_PROVIDER__SOURCE", "file")
    monkeypatch.setenv("MCE_SNAPSHOT__CADENCE_NS", "1000")
    settings = MCEConfig(_env_file=None)
    assert settings.provider.source == "file"
    assert settings.snapshot.cadence_ns == 1000


def test_config_validation_and_source_identifier() -> None:
    with pytest.raises(ValidationError):
        SnapshotSettings(cadence_ns=0)
    with pytest.raises(ValidationError):
        ProviderSettings(source="Vendor With Spaces")


def test_canonical_json_sorts_maps_sets_and_rejects_nonfinite() -> None:
    a = {"z": {"b", "a"}, "a": {"y": 2, "x": 1}}
    b = {"a": {"x": 1, "y": 2}, "z": {"a", "b"}}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert canonical_json_str(a) == '{"a":{"x":1,"y":2},"z":["a","b"]}'
    assert json.loads(canonical_json_str(a))["z"] == ["a", "b"]
    with pytest.raises(ValueError):
        canonical_json_str({"invalid": float("nan")})
    with pytest.raises(TypeError):
        canonical_json_str({1: "invalid"})
