# Market Context Engine

Market Context Engine (MCE) is a standalone Python research system for descriptive U.S. equity-market context. **MCE-1 through MCE-20 are complete; engine version is 1.0.0.** It produces typed, schema-1.0 `MarketContextSnapshot` results from causally available observations. It does not forecast returns, recommend trades, manage portfolios, or execute orders.

MCE covers XNYS sessions and causal ingestion/replay; trend, realized volatility, point-in-time breadth and sectors, correlation, dispersion, liquidity/stress, deterministic regimes and transition analytics; statistical/HMM research, factors, networks, seasonality, options and macro context; and an explainable ensemble. The MCE-20 pipeline runs existing engines in a validated dependency order, shares same-clock results, and emits separate lineage, diagnostics, and a deterministic snapshot fingerprint. Optional sources may be withheld without carrying old feature fields forward.

## Install and check

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/pytest --cov=mce --cov-report=term
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

No credentials are needed. Python 3.11 or newer is required. Runtime dependencies are Pydantic, pydantic-settings, exchange-calendars, and NumPy.

## Explicit-input quickstart

Save `config.json`:

```json
{"market":{"universe":{"broad_market":["SPY"]}}}
```

Save `observations.json` (one completed 2025-01-02 regular-session minute):

```json
{"observations":[{"kind":"bar","symbol":"SPY","source":"synthetic","event_timestamp_ns":1735828260000000000,"effective_timestamp_ns":1735828260000000000,"bar_start_ns":1735828200000000000,"bar_end_ns":1735828260000000000,"open":100,"high":101,"low":99,"close":100,"volume":1000}]}
```

```sh
.venv/bin/python -m mce validate-config --config config.json
.venv/bin/python -m mce snapshot --config config.json --input observations.json --as-of-ns 1735828260000000000 --output snapshot.json
.venv/bin/python -m mce replay --config config.json --input observations.json --start-ns 1735828260000000000 --end-ns 1735828260000000000 --cadence completed_bar --output replay.jsonl
.venv/bin/python -m mce serve --config config.json --input observations.json --host 127.0.0.1 --port 8000
```

The single-bar example validates the plumbing; many quantitative scores need longer history and multiple constituents. The CLI refuses to overwrite output unless `--force` is given. The local service exposes `/health`, `/ready`, `/version`, `/api/context/schema`, `/api/context/snapshot`, and `/api/context/replay` over preconfigured observations. It is a synchronous local research adapter without bundled authentication or TLS.

## Contracts and limits

All quantitative clocks are explicit UTC nanoseconds. Only completed regular-session bars available by `T` enter a snapshot at `T`; point-in-time membership, macro vintages, option availability, and filtered HMM artifact availability remain causal. A fresh snapshot is built for every call. Runtime timing and service metadata stay outside its fingerprint. Consumers such as AME read the snapshot or API contract and own any separate decision and execution logic.

See [system architecture](docs/system-architecture.md), [service API](docs/service-api.md), [historical replay](docs/replay.md), [operations](docs/operations.md), [production-style readiness](docs/production-readiness.md), [feature catalog](docs/features.md), [ensemble formulas](docs/ensemble-context-engine.md), and [development](docs/development.md). The [changelog](CHANGELOG.md) records the 1.0.0 boundary. Real vendor integrations, persistent storage, distributed service deployment, and further large-scale load testing remain future infrastructure work.
