# Production-style research readiness

MCE 1.0.0 completes the planned twenty-phase research roadmap and offers a stable schema-1.0 snapshot, deterministic historical replay, typed provider boundaries, a local service/CLI, stage diagnostics, and reproducibility lineage. It is a hardened context-service foundation, not a live trading system.

Limitations: no live vendor adapter is bundled; options and macro quality depend on supplied sources; statistical states require a separate trained-artifact lifecycle and explicit availability; the local WSGI server has no authentication or TLS; the in-memory replay store is bounded and does not provide persistence, distributed computation, or high-frequency tick storage. Large-scale deployment needs independent load testing, operational telemetry, access control, and provider-specific integration. No trading recommendation, portfolio, broker, execution, alpha, or forecasting layer is included.

## Release checklist

1. Run full pytest and verify at least 95% coverage.
2. Run Ruff lint/format, strict mypy, and `git diff --check`.
3. Review schema-1.0 JSON round-trip and engine version 1.0.0.
4. Run future-data, future-membership, macro-revision, options-surface, HMM-artifact, replay, and stale-field tests.
5. Run `.venv/bin/python -m benchmarks.full_pipeline`; record workload, total and per-stage times, and optional memory separately.
6. Review full diff/status, credential and TODO scans, API error safety, and dependency list.
7. Review architecture, service, replay, operations, and downstream-consumer documentation.

Post-1.0 work may add real vendor transports, persistent model registries, service deployment, and independent downstream systems. Those are outside MCE-20.
