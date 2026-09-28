"""Canonical observation ingestion and normalization."""

from mce.ingestion.normalize import normalize_observation
from mce.ingestion.service import IngestionResult, IngestionService

__all__ = ["IngestionResult", "IngestionService", "normalize_observation"]
