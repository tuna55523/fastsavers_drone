from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


SOURCES = [
    "VisDrone.v1i.yolov8",
    "Human Detection Through Drone.v1i.yolov8",
    "INRIA Person detection dataset.v1i.yolov8",
]

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def ensure_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def normalize_label_file(src_label: Path, dst_label: Path) -> Tuple[int, int]:
    """
    Returns:
      (kept_lines, dropped_lines)
    """
    kept: List[str] = []
    dropped = 0
    raw = src_label.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    for line in raw:
        parts = line.strip().split()
        if len(parts) < 5:
            dropped += 1
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            dropped += 1
            continue
        # Force single-class schema: class 0 = person
        kept.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    dst_label.write_text("\n".join(kept), encoding="utf-8")
    return len(kept), dropped


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def merge(base_dir: Path, out_dir: Path) -> Dict[str, Dict[str, int]]:
    ensure_dirs(out_dir)
    summary: Dict[str, Dict[str, int]] = {
        split: {
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "dropped_label_lines": 0,
        }
        for split in SPLITS
    }
    source_stats: Dict[str, Dict[str, int]] = {}

    for src_name in SOURCES:
        src_root = base_dir / src_name
        if not src_root.exists():
            raise FileNotFoundError(f"Source dataset missing: {src_root}")

        src_slug = slugify(src_name)
        source_stats[src_name] = {"images": 0, "labels": 0, "boxes": 0}

        for split in SPLITS:
            img_dir = src_root / split / "images"
            lbl_dir = src_root / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(f"Missing split folders in {src_name} -> {split}")

            images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

            for img in images:
                new_stem = f"{src_slug}__{img.stem}"
                dst_img = unique_path(out_dir / split / "images" / f"{new_stem}{img.suffix.lower()}")
                # Keep label stem identical to image stem
                dst_lbl = out_dir / split / "labels" / f"{dst_img.stem}.txt"

                src_lbl = lbl_dir / f"{img.stem}.txt"
                if not src_lbl.exists():
                    # keep image with empty label file
                    shutil.copy2(img, dst_img)
                    dst_lbl.write_text("", encoding="utf-8")
                    summary[split]["images"] += 1
                    summary[split]["labels"] += 1
                    source_stats[src_name]["images"] += 1
                    source_stats[src_name]["labels"] += 1
                    continue

                shutil.copy2(img, dst_img)
                kept, dropped = normalize_label_file(src_lbl, dst_lbl)

                summary[split]["images"] += 1
                summary[split]["labels"] += 1
                summary[split]["boxes"] += kept
                summary[split]["dropped_label_lines"] += dropped

                source_stats[src_name]["images"] += 1
                source_stats[src_name]["labels"] += 1
                source_stats[src_name]["boxes"] += kept

    # Write YOLO data.yaml
    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # Write machine-readable summary
    report = {
        "sources": SOURCES,
        "source_stats": source_stats,
        "split_summary": summary,
        "total_images": sum(summary[s]["images"] for s in SPLITS),
        "total_labels": sum(summary[s]["labels"] for s in SPLITS),
        "total_boxes": sum(summary[s]["boxes"] for s in SPLITS),
    }
    (out_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write human-readable summary
    lines = [
        "# Merged Dataset Report",
        "",
        f"Output: {out_dir}",
        "",
        "## Sources",
    ]
    for src in SOURCES:
        st = source_stats[src]
        lines.append(f"- {src}: images={st['images']} labels={st['labels']} boxes={st['boxes']}")

    lines.extend(["", "## Split Summary"])
    for split in SPLITS:
        st = summary[split]
        lines.append(
            f"- {split}: images={st['images']} labels={st['labels']} boxes={st['boxes']} dropped_lines={st['dropped_label_lines']}"
        )

    lines.extend(
        [
            "",
            "## Total",
            f"- images={report['total_images']}",
            f"- labels={report['total_labels']}",
            f"- boxes={report['total_boxes']}",
            "",
            "Class schema normalized to: class 0 = person",
        ]
    )
    (out_dir / "merge_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    # script location: bilimşenliğidrone/scripts
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "data" / "merged_person3"
    if output_root.exists():
        shutil.rmtree(output_root)
    merge(project_root, output_root)
    print(f"[OK] merged dataset created: {output_root}")
