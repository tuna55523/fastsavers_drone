# Controls Guide

Get battery:
python -c "from system.drone.drone_controller import DroneController as D; d=D(); print('Battery:', d.get_battery(), '%'); d.close()"

This project has two operation modes:

- `MANUAL` (operator RC control)
- `AUTO` (vision-driven search/track/rescue behavior)

## Main Window Controls

- `U`: Toggle `AUTO` / `MANUAL`
- `T`: Takeoff
- `L`: Land
- `ESC`: Exit app

## Manual Movement Controls

- `W`: Forward
- `S`: Backward
- `A`: Left
- `D`: Right
- `Q`: Rotate left (yaw -)
- `E`: Rotate right (yaw +)
- `C`: Up
- `Z` or `X`: Down

Manual speed is controlled by `MANUAL_SPEED` in `config.py`.

## Auto Mode Behavior

When `AUTO` is enabled:

1. Vision pipeline continues detection + pose + risk analysis.
2. State machine selects:
   - `SEARCH`: Drone performs smooth yaw sweep search.
   - `TRACK` / `RESCUE`: Drone auto-follows current target with smoothed yaw/forward control.
3. HUD overlays remain active (risk, target, policy, FPS, battery).

## Real Drone Notes

`RUN_MODE = "real"` uses a thread-based RC sender (in `system/movement/advanced_flight.py`) for smoother motion.
Key points:

- High-rate RC dispatch (`REAL_RC_HZ`)
- Command timeout hover (`REAL_RC_CMD_TIMEOUT_SEC`)
- Slew-limited command changes (reduces jerk)
- Link fail streak tracking + optional failsafe land
- Stream stale detection + stream auto-recovery

All related tuning parameters are in `config.py` under:

- `REAL_*`
- `AUTO_TRACK_*`
- `SEARCH_SWEEP_*`
