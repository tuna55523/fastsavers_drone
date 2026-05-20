# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous rescue drone system for detecting drowning victims in water. Runs on DJI Tello hardware with a YOLOv11-based vision pipeline, multi-person tracking, risk analysis, and a state machine for autonomous search/track/rescue behavior.

## Run Commands

```bash
# Run with real DJI Tello drone (set RUN_MODE = "real" in config.py)
python drone_main.py

# Run in simulation mode (set RUN_MODE = "sim" in config.py, uses videos/video8.mp4)
python drone_main.py
```

No build step — pure Python. No test suite exists yet.

## Configuration

All tunable parameters live in [config.py](config.py):
- `RUN_MODE`: `"real"` (Tello) or `"sim"` (local video)
- `CONF_THRESHOLD`: YOLO detection confidence (default 0.28)
- Risk thresholds: `WATCH_ENTRY_THRESHOLD`, `ALERT_ENTRY_THRESHOLD`, etc.
- Flight parameters: `MAX_SPEED`, `BATTERY_POLL_INTERVAL`
- Model paths: `MODEL_PATH` (detection), `POSE_MODEL_PATH` (pose)

## Architecture

### Entry Point
[drone_main.py](drone_main.py) — initializes all subsystems, runs the ~30fps main loop, renders HUD overlay, handles keyboard input.

### Keyboard Controls
- Manual flight: `W/S` (forward/back), `A/D` (left/right), `Q/E` (rotate), `C/Z` (up/down)
- `U` — toggle auto/manual mode
- `T/L` — takeoff/land

### Core Subsystems

**Vision** — [system/vision/detect_track.py](system/vision/detect_track.py)
- YOLOv11 detection + pose estimation (17 COCO keypoints)
- Falls back to Haar cascade face detection when YOLO misses

**Identity & Risk** — [system/identity_manager.py](system/identity_manager.py) (~975 lines, most complex file)
- Multi-person tracking via IoU + spatial distance + velocity prediction
- `compute_risk_v2()` — primary risk scorer with weighted signal combination:
  - Inactivity, panic motion, struggle-without-progress, visual flutter, pose flailing
  - Acute distress fast path (0.25–0.35s escalation vs 0.7–1.2s normal)
  - Rise-rate tracking to detect rapid score escalation
- Risk persistence state machine: SAFE → WATCH (0.42–0.50) → ALERT (0.64–0.68)

**Drone Control**
- [system/drone/drone_controller.py](system/drone/drone_controller.py) — real Tello interface
- [system/drone/sim_drone_controller.py](system/drone/sim_drone_controller.py) — video playback simulation
- [system/movement/advanced_flight.py](system/movement/advanced_flight.py) — high-rate RC sender, slew limiting, link failure tracking

**Intelligence**
- [system/intelligence/state_machine.py](system/intelligence/state_machine.py) — SEARCH / TRACK / RESCUE states
- [system/intelligence/search_behavior.py](system/intelligence/search_behavior.py) — yaw sweep pattern during SEARCH
- [system/intelligence/risk_engine.py](system/intelligence/risk_engine.py) — supplementary time-based risk

**Safety** — [system/safety/safety_manager.py](system/safety/safety_manager.py)
- Monitors battery; auto-lands at 15%

**Water Filtering** — [system/water_filter.py](system/water_filter.py)
- Filters detections outside water region to reduce false positives

### Sub-project: `bilimşenliğidrone/`
A parallel, simpler implementation focused on person-follow and acrobatics (flips, rolls). Has its own config, safety, and control modules. Contains planning docs (`00_master_roadmap.md`, `01-09_plan_*.md`) describing a 6-phase development roadmap.

### Sub-project: `android_drone_station/`
Android app for remote monitoring/control of the drone station.

## Risk Scoring Quick Reference

`compute_risk_v2()` builds on a 5% baseline and adds/subtracts per signal. Key weights:
- +58% visual flutter (splash/turbulence)
- +54% panic motion pattern
- +48% struggle without progress
- +42% inactivity
- +40% low-speed panic
- +34% pose flailing
- −28% smooth swimming (negative risk)

See [OPERATING_THRESHOLDS.md](OPERATING_THRESHOLDS.md) for full state machine timing documentation.
