# Market Context Engine

Market Context Engine (MCE) is a standalone research analytics project for broad U.S. equity-market context. It is designed to produce a typed, versioned `MarketContextSnapshot` that downstream research systems can consume. MCE is research analytics software. It does not execute trades.

**Current phase:** MCE-9, advanced liquidity and market stress (`0.9.0`). MCE classifies NYSE trading sessions, replays canonical observations with an explicit clock, and computes trend, realized volatility, market breadth, point-in-time sector context, continuous dependence evidence, return dispersion, and continuous liquidity/stress context. MCE does not forecast or classify regimes.

## Design

Provider adapters translate source-specific data into canonical bars, quotes, and trades. MCE includes deterministic in-memory historical data and membership providers and an XNYS exchange-calendar adapter. Trend, realized-volatility, breadth, sector, correlation, dispersion, and liquidity/stress engines read the causal store; their typed results can be projected into existing snapshot fields. The stress engine consumes optional same-clock outputs from prior engines without recomputing them. Later phases will add regime assessment. The snapshot is the integration boundary for Axiom Microstructure Engine (AME), which handles its own microstructure and execution research. A separate options engine may later supply options-derived context; MCE does not price options.

MCE uses UTC nanosecond timestamps internally and reserves U.S. Eastern Time for exchange-session interpretation and display. Every snapshot at time `T` must depend only on information effectively available by `T`. Unavailable analytics are `null`, never fabricated zeros.

See [architecture](docs/architecture.md), [data model](docs/data-model.md), [time semantics](docs/time-semantics.md), [calendar](docs/market-calendar.md), [ingestion](docs/ingestion.md), [replay](docs/replay.md), [feature windows](docs/features.md), [trend formulas](docs/trend-engine.md), [volatility formulas](docs/volatility-engine.md), [breadth formulas](docs/breadth-engine.md), [sector formulas](docs/sector-engine.md), [correlation formulas](docs/correlation-engine.md), [dispersion formulas](docs/dispersion-engine.md), [liquidity and stress formulas](docs/liquidity-stress-engine.md), [freshness and quality](docs/freshness-quality.md), and [development](docs/development.md).

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

MCE-9 adds causal per-member and broad liquidity evidence, comparable-time relative volume, Amihud and quoted-spread measurements, point-in-time gap breadth, and an explainable continuous stress composite. Regime classification remains MCE-10 work. Real vendor adapters, historical constituent feeds, corporate-action ingestion, Parquet I/O, persistent storage, and a local service also remain later work. MCE does not provide forecasts, brokerage integration, order routing, position management, stock selection, options valuation, or real-money trading.
