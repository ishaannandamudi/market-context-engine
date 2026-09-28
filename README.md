# Market Context Engine

Market Context Engine (MCE) is a standalone research analytics project for broad U.S. equity-market context. It is designed to produce a typed, versioned `MarketContextSnapshot` that downstream research systems can consume. MCE is research analytics software. It does not execute trades.

**Current phase:** MCE-2, U.S. market data and causal historical ingestion (`0.2.0`). MCE can now classify NYSE trading sessions, normalize and ingest canonical bars/quotes/trades, replay fixed historical data with an explicit clock, query a bounded causal store, and evaluate source freshness and coarse data quality. It still computes no market features or regimes.

## Design

Provider adapters translate source-specific data into canonical bars, quotes, and trades. MCE-2 includes a deterministic in-memory historical reference provider and an XNYS exchange-calendar adapter. Later phases will add feature calculation, regime assessment, and snapshot production. The snapshot is the integration boundary for Axiom Microstructure Engine (AME), which handles its own microstructure and execution research. A separate options engine may later supply options-derived context; MCE does not price options.

MCE uses UTC nanosecond timestamps internally and reserves U.S. Eastern Time for exchange-session interpretation and display. Every snapshot at time `T` must depend only on information effectively available by `T`. Unavailable analytics are `null`, never fabricated zeros.

See [architecture](docs/architecture.md), [data model](docs/data-model.md), [time semantics](docs/time-semantics.md), [calendar](docs/market-calendar.md), [ingestion](docs/ingestion.md), [replay](docs/replay.md), [freshness and quality](docs/freshness-quality.md), and [development](docs/development.md).

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

Python 3.11 or newer is required. No API keys or credentials are needed. Configuration can be supplied through nested environment variables such as `MCE_PROVIDER__SOURCE=file` or `MCE_REPLAY__MAX_OBSERVATIONS=50000`.

## Roadmap boundary

MCE-2 adds a historical calendar/ingestion/replay foundation to the MCE-1 contracts. MCE-3 will add reusable feature windows and begin actual quantitative features. Real vendor adapters, Parquet I/O, late-arrival live-feed handling, confidence modeling, regimes, persistence, and a local service remain later work. MCE does not provide brokerage integration, order routing, position management, stock selection, options valuation, or real-money trading.
