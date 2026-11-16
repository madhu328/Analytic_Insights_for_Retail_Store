"""Inspect incoming retail dataset ZIPs without extracting large video files.

Examples:
    python -m training.inspect_archives "data set store.zip" "data set file.zip" "videos.zip" "videos dataset.zip"
    python -m training.inspect_archives datasets/incoming/*.zip --json reports/archive_inventory.json

This report identifies candidates for video processing, YOLO annotation training,
and POS/retail analytics. It does NOT infer labels just from filenames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".dav", ".h264", ".webm"}
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLES = {".csv", ".xlsx", ".xls", ".parquet"}
LABELS = {".txt", ".xml", ".json"}
SKIP = {"__MACOSX", ".DS_Store"}


def inspect_archive(path: Path) -> dict:
    result = {"archive": str(path), "valid_zip": False}
    if not path.is_file():
        return {**result, "error": "file not found"}
    try:
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.infolist()
                       if not m.is_dir() and not any(part in SKIP for part in Path(m.filename).parts)]
            types = Counter(Path(m.filename).suffix.lower() for m in members)
            examples: dict[str, list[str]] = {"videos": [], "images": [], "tables": [], "possible_annotations": []}
            for m in members:
                ext = Path(m.filename).suffix.lower()
                if ext in VIDEOS:
                    category = "videos"
                elif ext in IMAGES:
                    category = "images"
                elif ext in TABLES:
                    category = "tables"
                elif ext in LABELS:
                    category = "possible_annotations"
                else:
                    continue
                if len(examples[category]) < 8:
                    examples[category].append(m.filename)
            return {
                **result, "valid_zip": True,
                "compressed_bytes": path.stat().st_size,
                "uncompressed_bytes": sum(m.file_size for m in members),
                "file_count": len(members),
                "extensions": dict(sorted(types.items(), key=lambda x: (-x[1], x[0]))),
                "video_count": sum(types[x] for x in VIDEOS),
                "image_count": sum(types[x] for x in IMAGES),
                "table_count": sum(types[x] for x in TABLES),
                "possible_annotation_count": sum(types[x] for x in LABELS),
                "examples": examples,
                "note": "TXT/JSON/XML counts do not establish valid person bounding-box labels."
            }
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return {**result, "error": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="Optional output JSON file")
    args = parser.parse_args(argv)
    reports = [inspect_archive(path) for path in args.archives]
    rendered = json.dumps(reports, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if not all(r["valid_zip"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
