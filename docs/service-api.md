# Local service/API

The optional stdlib WSGI adapter wraps `MarketContextService`, a framework-neutral class. It serves a **preconfigured in-memory historical observation batch**; requests cannot name a filesystem path or load a provider. Start it with explicit configuration and input files:

```sh
.venv/bin/python -m mce serve --config config.json --input observations.json --host 127.0.0.1 --port 8000
```

| Route | Purpose |
| --- | --- |
| `GET /health` | Process liveness only; returns `{"status":"ok"}`. |
| `GET /ready` | Critical market-source readiness plus separate optional provider statuses. Missing optional providers do not make the process unready. |
| `GET /version` | Engine and snapshot/statistical/options/macro schema versions and supported feature families. |
| `GET /api/context/schema` | Stable snapshot field names and schema version. |
| `POST /api/context/snapshot` | JSON `{"as_of_ns":1735828260000000000,"schema_version":"1.0","include_diagnostics":false}`. Returns snapshot, lineage, and fingerprint. |
| `POST /api/context/replay` | JSON `{"start_ns":...,"end_ns":...,"cadence":"completed_bar","max_snapshots":100}`. Returns bounded snapshots and fingerprints. Other cadence values: `n_minute` with `cadence_minutes`, `session_close`, or `timestamps` with sorted `timestamps`. |

`as_of_ns` and replay clocks are strict nonnegative UTC nanoseconds. Unknown fields, duplicate JSON keys, invalid clocks, and incompatible schema versions are rejected. Requests are limited to 64 KiB; service replay is limited to 100 snapshots and ten calendar years; responses are limited to 8 MiB. JSON is canonical, finite, and sorted. There are no NumPy objects, pickle payloads, or arbitrary file operations at the HTTP boundary.

Errors have `{"error":{"code":"INVALID_REQUEST","message":"..."}}`. Stable codes include `INVALID_AS_OF`, `INSUFFICIENT_MARKET_DATA`, `PROVIDER_UNAVAILABLE`, `SCHEMA_MISMATCH`, `MODEL_ARTIFACT_UNAVAILABLE`, `REPLAY_RANGE_INVALID`, and `INTERNAL_ERROR`. Validation details, stack traces, local paths, environment values, and source payloads are not returned. A 503 readiness result reports unavailable critical market data; `/health` remains a liveness check. Service instance counters are available through `service.metrics.snapshot()` for local observability. The stdlib WSGI server is intentionally small and synchronous.

No request ID, stage duration, provider health check, or logging field enters the quantitative fingerprint. For a richer deployment, embed the framework-neutral service behind an independently managed HTTP server with its own authentication, TLS, rate limits, and network policy.
