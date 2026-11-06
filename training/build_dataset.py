"""Step 3 - assemble a YOLO dataset with a leakage-free, video-level split.

  python -m training.build_dataset --images datasets/raw/images --labels datasets/raw/labels \
      --out datasets/store_person --val 0.15 --test 0.15

Frames are grouped by source video (prefix before ``__``) and whole videos are assigned to
train / val / test, so test accuracy reflects footage the model has never seen.
Images without a label file are kept as negatives (empty label) - important for teaching
the model that posters and mannequins are *not* people.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def group_of(path: Path) -> str:
    return path.stem.split("__")[0]


def split_groups(groups: list[str], val: float, test: float, seed: int = 42) -> dict[str, str]:
    g = sorted(groups)
    random.Random(seed).shuffle(g)
    n = len(g)
    n_test = max(1, round(n * test)) if n >= 3 and test > 0 else 0
    n_val = max(1, round(n * val)) if n - n_test >= 2 and val > 0 else 0
    out = {}
    for i, name in enumerate(g):
        out[name] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def build(images: Path, labels: Path, out: Path, val: float = 0.15, test: float = 0.15,
          seed: int = 42, copy: bool = True) -> dict:
    imgs = sorted(p for p in images.rglob("*") if p.suffix.lower() in IMG_EXT)
    by_group = defaultdict(list)
    for p in imgs:
        by_group[group_of(p)].append(p)
    assign = split_groups(list(by_group), val, test, seed)
    stats = {s: {"images": 0, "boxes": 0, "negatives": 0, "videos": 0} for s in ("train", "val", "test")}
    for grp, files in by_group.items():
        split = assign[grp]
        stats[split]["videos"] += 1
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            dst = out / "images" / split / img.name
            if copy:
                shutil.copy2(img, dst)
            elif not dst.exists():
                dst.symlink_to(img.resolve())
            lab = labels / (img.stem + ".txt")
            text = lab.read_text() if lab.exists() else ""
            (out / "labels" / split / (img.stem + ".txt")).write_text(text)
            nb = sum(1 for line in text.splitlines() if line.strip())
            stats[split]["images"] += 1
            stats[split]["boxes"] += nb
            stats[split]["negatives"] += nb == 0
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: person\n")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="datasets/raw/images")
    ap.add_argument("--labels", default="datasets/raw/labels")
    ap.add_argument("--out", default="datasets/store_person")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args(argv)
    stats = build(Path(args.images), Path(args.labels), Path(args.out), args.val, args.test, args.seed,
                  copy=not args.symlink)
    for k, v in stats.items():
        print(f"{k:5s}: {v}")
    print(f"dataset -> {args.out}/data.yaml")


if __name__ == "__main__":
    main()
