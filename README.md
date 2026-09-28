# Market Context Engine

Market Context Engine (MCE) is a standalone research analytics project for broad U.S. equity-market context. It is designed to produce a typed, versioned `MarketContextSnapshot` that downstream research systems can consume. MCE is research analytics software. It does not execute trades.

**Current phase:** MCE-1, foundation and public contracts (`0.1.0`). There are no data feeds, calculations, service endpoints, or trading functions yet.

## Design

Provider adapters will translate source-specific data into canonical bars, quotes, and trades. Later phases will add U.S. exchange-calendar handling, feature calculation, regime assessment, and snapshot production. The snapshot is the integration boundary for Axiom Microstructure Engine (AME), which handles its own microstructure and execution research. A separate options engine may later supply options-derived context; MCE does not price options.

MCE uses UTC nanosecond timestamps internally and reserves U.S. Eastern Time for exchange-session interpretation and display. Every snapshot at time `T` must depend only on information effectively available by `T`. Unavailable analytics are `null`, never fabricated zeros.

See [architecture](docs/architecture.md), [data model](docs/data-model.md), [time semantics](docs/time-semantics.md), and [development](docs/development.md).

## Install and verify

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/pytest --cov=mce --cov-report=term-missing
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

Python 3.11 or newer is required. There are no API keys or credentials in MCE-1. Configuration can be supplied through nested environment variables such as `MCE_PROVIDER__SOURCE=file`.

## Roadmap boundary

MCE-1 defines models, configuration, serialization, reproducibility, and provider protocols. Later phases will implement exchange calendars, historical/live provider adapters, freshness policies, features, regimes, explainability calculations, and a local service. MCE does not provide brokerage integration, order routing, position management, stock selection, options valuation, or real-money trading.
