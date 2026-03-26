# Operating Thresholds

This document defines the active operator policy used by the current system.

## Alert states

- `SAFE`
- `WATCH`
- `ALERT`

The displayed `risk` is persistence-stabilized. The displayed `RAW` risk is the underlying continuous signal before persistence.

## Enter/exit policy

Values come from `config.py`:

- `RISK_WATCH_ENTER = 0.42`
- `RISK_WATCH_EXIT = 0.32`
- `RISK_ALERT_ENTER = 0.64`
- `RISK_ALERT_EXIT = 0.52`
- `RISK_WATCH_ENTER_SECONDS = 0.7`
- `RISK_ALERT_ENTER_SECONDS = 1.2`
- `RISK_EXIT_SECONDS = 1.2`

Interpretation:

1. `SAFE -> WATCH` if raw risk is above watch-enter for at least watch-enter seconds.
2. `WATCH -> ALERT` if raw risk is above alert-enter for at least alert-enter seconds.
3. `WATCH -> SAFE` if raw risk is below watch-exit for at least exit seconds.
4. `ALERT -> WATCH` if raw risk is below alert-exit for at least exit seconds.

Fast distress path (early escalation):

- `RISK_FAST_WATCH = 0.50`
- `RISK_FAST_ALERT = 0.68`
- `RISK_FAST_WATCH_SECONDS = 0.35`
- `RISK_FAST_ALERT_SECONDS = 0.25`

## Operator actions

Values come from `config.py`:

- `ACTION_SAFE = "MONITOR"`
- `ACTION_WATCH = "APPROACH + OBSERVE"`
- `ACTION_ALERT = "RESCUE NOW"`

In practice:

1. `SAFE`: Keep observing and continue patrol/scan.
2. `WATCH`: Move closer, maintain camera lock, prepare intervention.
3. `ALERT`: Initiate rescue protocol immediately.

## Benchmark rule

Offline benchmark currently treats a video as positive when at least one `ALERT` state appears.
This can be adjusted in `config.py` via `BENCHMARK_ALERT_STATE`.
