# U.S. market calendar

MCE-2 uses the maintained [`exchange-calendars` XNYS calendar](https://github.com/gerrymanoim/exchange_calendars) directly. It supplies session dates, UTC regular opens/closes, exchange holidays, and early closes. It supports Python 3.11. [`pandas-market-calendars`](https://github.com/rsheftel/pandas_market_calendars) also supports these cases but depends on `exchange-calendars` itself; using it would add another calendar layer. The direct choice still pulls pandas, NumPy, and the calendar's other dependencies. No holiday table or regular-hours schedule is hard-coded in MCE.

`MarketSessionService` has an explicit configured calendar range, default 2000-01-01 through 2035-12-31. This avoids the library's moving default range relative to today's date. XNYS is a broad U.S. equity convention, not a claim that every U.S. option or venue shares the same sessions. Dates outside the configured range classify as `UNKNOWN`; direct trading-day and bounds queries raise `CalendarRangeError`.

The exchange library defines **regular** sessions. MCE separately defines configurable extended-hours interpretation in `America/New_York`, default 04:00 through the actual exchange open and actual exchange close through 20:00. These are research classification windows, not venue guarantees. On a trading date:

- `PRE_MARKET` is `[04:00 local, regular_open)`.
- `REGULAR` is `[regular_open, regular_close)`.
- `AFTER_HOURS` is `[regular_close, 20:00 local)`.
- All other instants are `CLOSED`. Holidays and weekends are `CLOSED` throughout.

Thus exact regular open is regular; exact regular close is after hours. On an early-close date, after hours begins at the actual early close. The local New York date determines the trading date, so a UTC midnight instant may still belong to the previous New York date. The service converts from UTC with `zoneinfo`; EST, EDT, and DST transitions never depend on machine timezone. A `CLOSED` instant on a trading date still reports that trading date; a holiday/weekend reports none.

`previous_trading_day` and `next_trading_day` are strictly before/after the given date. `regular_session_bounds` requires a session. `snapshot_times(day, cadence_ns)` returns regular-session completion instants `open + cadence` through close inclusive when the cadence lands on close. It does not create incomplete trailing intervals. A snapshot at exact close can see the just-completed final bar even though the session classifier labels the instant `AFTER_HOURS`.
