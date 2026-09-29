"""Deterministic hash of result-relevant, non-secret configuration."""

from hashlib import sha256

from mce.config.settings import MCEConfig
from mce.serialization.canonical import canonical_json_bytes


def config_fingerprint(config: MCEConfig) -> str:
    """Hash an explicit allowlist; operational logging settings are excluded."""
    payload = {
        "market": config.market,
        "provider": config.provider,
        "snapshot": config.snapshot,
        "ingestion": config.ingestion,
        "freshness": config.freshness,
        "replay": config.replay,
        "features": config.features,
        "statistical": config.statistical,
        "reproducibility": config.reproducibility,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()
