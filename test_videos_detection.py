import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FRAME_HEIGHT, FRAME_WIDTH
from system.vision.detect_track import DetectTrackSystem

VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = VIDEOS_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DISPLAY_NAME = "Drone Video Test"
PROCESS_EVERY_N = 2


def list_videos(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def draw_header(frame, video_name: str, frame_no: int, total_people: int, danger_count: int, target_text: str, fps_text: str):
    cv2.rectangle(frame, (10, 10), (780, 92), (20, 20, 20), -1)
    cv2.putText(frame, f"VIDEO: {video_name}", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"FRAME: {frame_no}", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (120, 220, 255), 2)
    cv2.putText(frame, f"INSAN: {total_people}  TEHLIKE: {danger_count}", (190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (120, 255, 120), 2)
    cv2.putText(frame, target_text, (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 210, 120), 2)
    cv2.putText(frame, fps_text, (610, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (120, 255, 255), 2)


def main():
    videos = list_videos(VIDEOS_DIR)
    if not videos:
        print(f"[HATA] Video bulunamadi: {VIDEOS_DIR}")
        return

    print("[INFO] Bulunan videolar:")
    for idx, video in enumerate(videos, start=1):
        print(f"  {idx}. {video.name}")

    detector = DetectTrackSystem()
    cv2.namedWindow(DISPLAY_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    print("\n[INFO] Kontroller:")
    print("  q = cikis")
    print("  n = sonraki video")
    print("  r = videoyu bastan oynat")
    print("  s = islenmis videoyu kaydet")
    print("  bosluk = dur / devam et")

    video_idx = 0
    paused = False
    fps_smooth = 0.0
    last_infer = {
        "target": None,
        "persons": [],
        "display": None,
        "infer_ms": 0.0,
    }

    while video_idx < len(videos):
        video_path = videos[video_idx]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[HATA] Acilamadi: {video_path.name}")
            video_idx += 1
            continue

        detector.reset_tracking()
        frame_no = 0
        writer = None
        save_enabled = False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = FRAME_WIDTH
        height = FRAME_HEIGHT

        print(f"\n[INFO] Oynatiliyor: {video_path.name}")

        while True:
            loop_t0 = time.perf_counter()
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print(f"[INFO] Bitti: {video_path.name}")
                    break
                frame_no += 1
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
                rerun_infer = (frame_no % PROCESS_EVERY_N) == 1 or last_infer["display"] is None

                if rerun_infer:
                    ts = time.time()
                    infer_t0 = time.perf_counter()
                    annotated, target, persons, _ = detector.run(frame.copy(), frame_ts=ts)
                    infer_ms = (time.perf_counter() - infer_t0) * 1000.0
                    last_infer["target"] = target
                    last_infer["persons"] = persons
                    last_infer["display"] = annotated
                    last_infer["infer_ms"] = infer_ms
                else:
                    annotated = last_infer["display"].copy()
                    target = last_infer["target"]
                    persons = list(last_infer["persons"])
                    infer_ms = float(last_infer["infer_ms"])

                danger_count = sum(1 for p in persons if p.get("is_danger", False))
                if target is not None:
                    target_text = (
                        f"HEDEF ID:{target.get('id')}  DURUM:{target.get('status_label', 'SAFE')}  "
                        f"RISK:{float(target.get('risk', 0.0)):.2f}"
                    )
                else:
                    target_text = "HEDEF: YOK"

                fps_now = 1.0 / max(1e-6, time.perf_counter() - loop_t0)
                fps_smooth = fps_now if fps_smooth <= 0.0 else (0.88 * fps_smooth + 0.12 * fps_now)
                fps_text = f"FPS:{fps_smooth:4.1f}  INF:{infer_ms:4.0f}ms"

                draw_header(
                    annotated,
                    video_path.name,
                    frame_no,
                    len(persons),
                    danger_count,
                    target_text,
                    fps_text,
                )

                if target is not None:
                    x1, y1, x2, y2 = target["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(
                        annotated,
                        "AKTIF HEDEF",
                        (x1, max(20, y1 - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )

                if save_enabled and writer is not None:
                    writer.write(annotated)

                display = annotated
                last_infer["display"] = annotated.copy()
            else:
                display = display if 'display' in locals() else None
                if display is None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    display = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

            if display is not None:
                cv2.imshow(DISPLAY_NAME, display)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord('q'):
                cap.release()
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
                return
            if key == ord('n'):
                break
            if key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detector.reset_tracking()
                frame_no = 0
                paused = False
                continue
            if key == ord(' '):
                paused = not paused
            if key == ord('s') and not save_enabled:
                out_path = OUTPUT_DIR / f"processed_{video_path.stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                save_enabled = True
                print(f"[INFO] Kayit basladi: {out_path}")

        cap.release()
        if writer is not None:
            writer.release()
        video_idx += 1

    cv2.destroyAllWindows()
    print("[INFO] Tum videolar tamamlandi.")


if __name__ == "__main__":
    main()
