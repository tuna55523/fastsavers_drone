import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BENCHMARK_ALERT_STATE,
    BENCHMARK_LABELS_PATH,
    BENCHMARK_OUTPUT_DIR,
    BENCHMARK_RESIZE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from system.vision.detect_track import DetectTrackSystem  # noqa: E402


POSITIVE_LABELS = {"drowning", "panic", "distress", "rescue"}
NEGATIVE_LABELS = {"normal", "swimming", "safe", "neutral"}
STATE_RANK = {"NONE": 0, "SAFE": 1, "WATCH": 2, "ALERT": 3}


def normalize_label(label):
    if label is None:
        return None
    v = str(label).strip().lower()
    if v in POSITIVE_LABELS:
        return "positive"
    if v in NEGATIVE_LABELS:
        return "negative"
    return None


def read_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return data["videos"]

    raise ValueError("labels file must be a list or {\"videos\": [...]} format")


def run_video(vision, video_path, name, label, frames_writer, resize):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "name": name,
            "path": str(video_path),
            "label": label or "",
            "status": "open_failed",
            "frames": 0,
            "fps": 0.0,
            "source_fps": 0.0,
            "proc_fps": 0.0,
            "mean_frame_ms": 0.0,
            "wall_duration_sec": 0.0,
            "duration_sec": 0.0,
            "max_risk": 0.0,
            "max_raw_risk": 0.0,
            "max_state": "NONE",
            "first_watch_sec": "",
            "first_alert_sec": "",
            "predicted_alert": "negative",
            "predicted_watch": "negative",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    frame_idx = 0
    max_risk = 0.0
    max_raw_risk = 0.0
    max_state = "NONE"
    first_watch = None
    first_alert = None
    process_time_acc = 0.0
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if resize:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

        sec = frame_idx / fps
        frame_start = time.perf_counter()
        _, target, _ = vision.run(frame, frame_ts=sec)
        process_time_acc += (time.perf_counter() - frame_start)

        if target is None:
            risk = 0.0
            raw_risk = 0.0
            alert_state = "NONE"
            target_id = -1
        else:
            risk = float(target.get("risk", 0.0))
            raw_risk = float(target.get("raw_risk", risk))
            alert_state = str(target.get("alert_state", "SAFE"))
            target_id = int(target.get("id", -1))

        max_risk = max(max_risk, risk)
        max_raw_risk = max(max_raw_risk, raw_risk)
        if STATE_RANK.get(alert_state, 0) > STATE_RANK.get(max_state, 0):
            max_state = alert_state

        if first_watch is None and alert_state in {"WATCH", "ALERT"}:
            first_watch = sec
        if first_alert is None and alert_state == BENCHMARK_ALERT_STATE:
            first_alert = sec

        frames_writer.writerow([
            name,
            str(video_path),
            label or "",
            frame_idx,
            f"{sec:.3f}",
            target_id,
            alert_state,
            f"{risk:.4f}",
            f"{raw_risk:.4f}",
        ])

        frame_idx += 1

    cap.release()
    wall_duration = max(1e-6, time.perf_counter() - wall_start)
    proc_fps = float(frame_idx / wall_duration if frame_idx > 0 else 0.0)
    mean_frame_ms = float((process_time_acc / frame_idx) * 1000.0 if frame_idx > 0 else 0.0)

    predicted_alert = "positive" if first_alert is not None else "negative"
    predicted_watch = "positive" if first_watch is not None else "negative"

    return {
        "name": name,
        "path": str(video_path),
        "label": label or "",
        "status": "ok",
        "frames": frame_idx,
        "fps": float(fps),
        "source_fps": float(fps),
        "proc_fps": proc_fps,
        "mean_frame_ms": mean_frame_ms,
        "wall_duration_sec": float(wall_duration),
        "duration_sec": float(frame_idx / fps if fps > 0 else 0.0),
        "max_risk": float(max_risk),
        "max_raw_risk": float(max_raw_risk),
        "max_state": max_state,
        "first_watch_sec": "" if first_watch is None else f"{first_watch:.3f}",
        "first_alert_sec": "" if first_alert is None else f"{first_alert:.3f}",
        "predicted_alert": predicted_alert,
        "predicted_watch": predicted_watch,
    }


def compute_metrics(summaries, prediction_key):
    tp = fp = tn = fn = 0

    for row in summaries:
        gt = normalize_label(row["label"])
        if gt is None or row["status"] != "ok":
            continue

        pred = row[prediction_key]
        if gt == "positive" and pred == "positive":
            tp += 1
        elif gt == "positive" and pred == "negative":
            fn += 1
        elif gt == "negative" and pred == "positive":
            fp += 1
        elif gt == "negative" and pred == "negative":
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline benchmark for drowning detection.")
    parser.add_argument("--labels", default=BENCHMARK_LABELS_PATH, help="Path to labels JSON.")
    parser.add_argument("--output-dir", default=BENCHMARK_OUTPUT_DIR, help="Output directory for CSV/JSON reports.")
    parser.add_argument("--no-resize", action="store_true", help="Disable benchmark frame resize.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_labels(labels_path)
    if not rows:
        raise ValueError("labels list is empty")

    frames_csv_path = output_dir / "benchmark_frames.csv"
    summary_csv_path = output_dir / "benchmark_summary.csv"
    report_json_path = output_dir / "benchmark_report.json"

    resize = BENCHMARK_RESIZE and not args.no_resize

    vision = DetectTrackSystem()
    summaries = []

    with open(frames_csv_path, "w", newline="", encoding="utf-8") as frames_file:
        frames_writer = csv.writer(frames_file)
        frames_writer.writerow([
            "video_name",
            "video_path",
            "label",
            "frame_idx",
            "sec",
            "target_id",
            "alert_state",
            "risk",
            "raw_risk",
        ])

        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                continue

            vision.reset_tracking()

            video_path = Path(item.get("path", ""))
            if not video_path.is_absolute():
                video_path = ROOT / video_path

            name = item.get("name") or video_path.stem or f"video_{i}"
            label = item.get("label")

            print(f"[BENCH] {name} -> {video_path}")
            summary = run_video(
                vision=vision,
                video_path=video_path,
                name=name,
                label=label,
                frames_writer=frames_writer,
                resize=resize,
            )
            summaries.append(summary)

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "name",
                "path",
                "label",
                "status",
                "frames",
                "fps",
                "source_fps",
                "proc_fps",
                "mean_frame_ms",
                "wall_duration_sec",
                "duration_sec",
                "max_risk",
                "max_raw_risk",
                "max_state",
                "first_watch_sec",
                "first_alert_sec",
                "predicted_alert",
                "predicted_watch",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    metrics_alert = compute_metrics(summaries, "predicted_alert")
    metrics_watch = compute_metrics(summaries, "predicted_watch")
    report = {
        "labels_file": str(labels_path),
        "output_dir": str(output_dir),
        "resize": resize,
        "alert_state_for_positive": BENCHMARK_ALERT_STATE,
        "metrics_alert": metrics_alert,
        "metrics_watch": metrics_watch,
        "videos": summaries,
    }

    with open(report_json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print("[BENCH] Finished")
    print(f"[BENCH] Frames:  {frames_csv_path}")
    print(f"[BENCH] Summary: {summary_csv_path}")
    print(f"[BENCH] Report:  {report_json_path}")
    print(f"[BENCH] Metrics ALERT: {metrics_alert}")
    print(f"[BENCH] Metrics WATCH: {metrics_watch}")


if __name__ == "__main__":
    main()
