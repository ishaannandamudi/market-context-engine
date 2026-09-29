# Market Context Engine

Market Context Engine (MCE) is a standalone research analytics project for broad U.S. equity-market context. It is designed to produce a typed, versioned `MarketContextSnapshot` that downstream research systems can consume. MCE is research analytics software. It does not execute trades.

**Current phase:** MCE-10, deterministic market regimes (`0.10.0`). MCE classifies NYSE trading sessions, replays canonical observations with an explicit clock, and computes trend, realized volatility, market breadth, point-in-time sector context, continuous dependence evidence, return dispersion, and continuous liquidity/stress context. It combines those observed features into explainable regime dimensions and a headline state. Regime classification is descriptive, not a forecast.

## Design

Provider adapters translate source-specific data into canonical bars, quotes, and trades. MCE includes deterministic in-memory historical data and membership providers and an XNYS exchange-calendar adapter. Trend, realized-volatility, breadth, sector, correlation, dispersion, and liquidity/stress engines read the causal store; their typed results can be projected into existing snapshot fields. The regime engine consumes same-clock upstream evidence and explicit prior regime history without recomputing feature engines or keeping hidden state. The snapshot is the integration boundary for Axiom Microstructure Engine (AME), which handles its own microstructure and execution research. A separate options engine may later supply options-derived context; MCE does not price options.

MCE uses UTC nanosecond timestamps internally and reserves U.S. Eastern Time for exchange-session interpretation and display. Every snapshot at time `T` must depend only on information effectively available by `T`. Unavailable analytics are `null`, never fabricated zeros.

See [architecture](docs/architecture.md), [data model](docs/data-model.md), [time semantics](docs/time-semantics.md), [calendar](docs/market-calendar.md), [ingestion](docs/ingestion.md), [replay](docs/replay.md), [feature windows](docs/features.md), [trend formulas](docs/trend-engine.md), [volatility formulas](docs/volatility-engine.md), [breadth formulas](docs/breadth-engine.md), [sector formulas](docs/sector-engine.md), [correlation formulas](docs/correlation-engine.md), [dispersion formulas](docs/dispersion-engine.md), [liquidity and stress formulas](docs/liquidity-stress-engine.md), [regime rules](docs/regime-engine.md), [freshness and quality](docs/freshness-quality.md), and [development](docs/development.md).

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

MCE-10 adds deterministic regime dimensions, a compact headline classification, ordered rule traces, explicit conflict records, evidence strength, replayable hysteresis, and typed transitions. MCE-11 may add descriptive transition analytics from observed regime history. Real vendor adapters, historical constituent feeds, corporate-action ingestion, Parquet I/O, persistent storage, and a local service also remain later work. MCE does not provide forecasts, brokerage integration, order routing, position management, stock selection, options valuation, or real-money trading.
