import argparse
from pathlib import Path

from ultralytics import YOLO


def export_tflite(weights: Path, imgsz: int, int8: bool, device: str):
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    result = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=int8,
        half=False,
        dynamic=False,
        simplify=True,
        device=device,
        nms=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to Android-friendly TFLite")
    parser.add_argument(
        "--weights",
        default="bilimşenliğidrone/models/person_tracking_best_v8n_768.pt",
        help="Path to .pt weights",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for export")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--device", default="cpu", help="Export device")
    args = parser.parse_args()

    out = export_tflite(Path(args.weights), args.imgsz, args.int8, args.device)
    print(f"Export completed: {out}")


if __name__ == "__main__":
    main()
