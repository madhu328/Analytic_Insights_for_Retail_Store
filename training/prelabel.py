"""Step 2 - pre-label frames with a large teacher model, then correct them by hand.

  python -m training.prelabel datasets/raw/images --out datasets/raw/labels --model yolo11x.pt

Writes YOLO txt labels (class 0 = person, normalised cx cy w h). Import images + labels
into CVAT / Label Studio / Roboflow ("YOLO 1.1" format), fix missed and false boxes
(occluded shoppers, mirror reflections, posters, mannequins), export back as YOLO.
Hand-correcting teacher labels is ~5x faster than drawing every box and is the single
biggest lever for >95% precision/recall on your own cameras.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("--out", default="datasets/raw/labels")
    ap.add_argument("--model", default="yolo11x.pt", help="teacher: the largest model you can run")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for r in model.predict(source=args.images, classes=[0], conf=args.conf, imgsz=args.imgsz,
                           device=args.device, stream=True, verbose=False):
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in r.boxes.xywhn.tolist()]
        (out / (Path(r.path).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    print(f"pre-labelled {n_img} images, {n_box} person boxes -> {out}. Now review them in CVAT/Label Studio.")


if __name__ == "__main__":
    main()
