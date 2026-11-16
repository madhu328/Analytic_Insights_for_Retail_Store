"""Audit a directory of images and YOLO annotations before training.

Usage:
    python -m training.audit_dataset --images datasets/raw/images --labels datasets/raw/labels
    python -m training.audit_dataset --images datasets/raw/images --labels datasets/raw/labels --allow-missing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit(images: Path, labels: Path, allow_missing: bool = False) -> dict:
    if not images.is_dir():
        raise ValueError(f"Image directory does not exist: {images}")
    if not labels.is_dir():
        raise ValueError(f"Label directory does not exist: {labels}")

    image_paths = sorted(p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No supported images found in {images}")
    by_name: dict[str, Path] = {}
    errors: list[str] = []
    n_boxes, n_negative, n_missing = 0, 0, 0
    for path in image_paths:
        if path.stem in by_name:
            errors.append(f"Duplicate image stem {path.stem}: {by_name[path.stem]} and {path}")
        by_name[path.stem] = path
        label = labels / f"{path.stem}.txt"
        if not label.is_file():
            n_missing += 1
            if not allow_missing:
                errors.append(f"Missing annotation for {path.name}: {label}")
            continue
        rows = label.read_text(encoding="utf-8").splitlines()
        nonblank = [line for line in rows if line.strip()]
        if not nonblank:
            n_negative += 1
        for n, row in enumerate(nonblank, 1):
            fields = row.split()
            if len(fields) != 5:
                errors.append(f"{label}:{n}: expected class cx cy width height")
                continue
            try:
                category = int(fields[0])
                x, y, w, h = map(float, fields[1:])
            except ValueError:
                errors.append(f"{label}:{n}: non-numeric YOLO annotation")
                continue
            if category != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1) or x - w/2 < -1e-6 or x + w/2 > 1+1e-6 or y - h/2 < -1e-6 or y + h/2 > 1+1e-6:
                errors.append(f"{label}:{n}: invalid person box or class")
            else:
                n_boxes += 1
    for label in labels.rglob("*.txt"):
        if label.stem not in by_name:
            errors.append(f"Orphan annotation: {label}")
    report = {
        "images": len(image_paths), "persons": n_boxes,
        "verified_negative_images": n_negative, "missing_labels": n_missing,
        "source_videos": len({p.stem.split("__")[0] for p in image_paths}),
        "error_count": len(errors), "errors": errors[:100],
        "valid": not errors,
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--allow-missing", action="store_true", help="Only for intentionally unlabelled images; not for supervised training")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    result = audit(args.images, args.labels, args.allow_missing)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
