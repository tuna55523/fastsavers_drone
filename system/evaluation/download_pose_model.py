import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def _resolve_downloaded_checkpoint(model, model_id):
    candidates = [
        getattr(model, "ckpt_path", None),
        getattr(model, "pt_path", None),
        model_id,
    ]

    model_obj = getattr(model, "model", None)
    if model_obj is not None:
        candidates.extend(
            [
                getattr(model_obj, "pt_path", None),
                getattr(model_obj, "yaml_file", None),
            ]
        )

    for cand in candidates:
        if not cand:
            continue
        p = Path(str(cand))
        if p.exists() and p.is_file():
            return p

    return None


def download_pose_model(model_id, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[POSE] Preparing model: {model_id}")
    model = YOLO(model_id)

    src = _resolve_downloaded_checkpoint(model, model_id)
    if src is None:
        raise FileNotFoundError(
            f"Downloaded checkpoint not found after loading '{model_id}'."
        )

    if src.resolve() != output_path.resolve():
        shutil.copy2(src, output_path)
        print(f"[POSE] Copied model to: {output_path}")
    else:
        print(f"[POSE] Model already at: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download and place a pose model.")
    parser.add_argument("--model-id", default="yolo11s-pose.pt", help="Ultralytics model id.")
    parser.add_argument(
        "--output",
        default="models/yolo11s-pose.pt",
        help="Target path where the model file will be placed.",
    )
    args = parser.parse_args()

    out = download_pose_model(args.model_id, args.output)
    print(f"[POSE] Ready: {out}")


if __name__ == "__main__":
    main()
