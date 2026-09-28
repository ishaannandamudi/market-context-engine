# Market Context Engine

Market Context Engine (MCE) is a standalone research analytics project for broad U.S. equity-market context. It is designed to produce a typed, versioned `MarketContextSnapshot` that downstream research systems can consume. MCE is research analytics software. It does not execute trades.

**Current phase:** MCE-4, advanced realized volatility (`0.4.0`). MCE classifies NYSE trading sessions, ingests canonical observations, replays fixed historical data with an explicit clock, and computes completed-bar trend and realized-volatility context. It describes observed volatility; it does not forecast future volatility or classify regimes.

## Design

Provider adapters translate source-specific data into canonical bars, quotes, and trades. MCE includes a deterministic in-memory historical reference provider and an XNYS exchange-calendar adapter. Trend and realized-volatility engines read the causal store only; their typed results can be projected into existing snapshot fields. Later phases will add other features and regime assessment. The snapshot is the integration boundary for Axiom Microstructure Engine (AME), which handles its own microstructure and execution research. A separate options engine may later supply options-derived context; MCE does not price options.

MCE uses UTC nanosecond timestamps internally and reserves U.S. Eastern Time for exchange-session interpretation and display. Every snapshot at time `T` must depend only on information effectively available by `T`. Unavailable analytics are `null`, never fabricated zeros.

See [architecture](docs/architecture.md), [data model](docs/data-model.md), [time semantics](docs/time-semantics.md), [calendar](docs/market-calendar.md), [ingestion](docs/ingestion.md), [replay](docs/replay.md), [feature windows](docs/features.md), [trend formulas](docs/trend-engine.md), [volatility formulas](docs/volatility-engine.md), [freshness and quality](docs/freshness-quality.md), and [development](docs/development.md).

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

MCE-4 adds close-to-close, range, Yang–Zhang, intraday realized and bipower variation, semivariance, and causal volatility context to the MCE-3 trend foundation. Market breadth and regime classification remain later work. Real vendor adapters, Parquet I/O, late-arrival live-feed handling, confidence modeling, persistent storage, and a local service also remain later work. MCE does not provide volatility forecasts, brokerage integration, order routing, position management, stock selection, options valuation, or real-money trading.
