"""Step 4 - fine-tune YOLO11m on your store frames and report test accuracy.

  python -m training.train_detector --data datasets/store_person/data.yaml --epochs 120 --imgsz 960

Choices that matter for CCTV:
* start from COCO-pretrained ``yolo11m.pt`` (transfer learning; person is a COCO class)
* ``single_cls`` person-only head; ``imgsz`` 960+ because shoppers far from the camera are small
* colour/brightness augmentation for store lighting, no vertical flips / rotation (CCTV is upright)
* early stopping on val mAP; final numbers on the *held-out test videos*

Writes ``reports/detector_metrics.json`` (shown on the dashboard's Model accuracy tab) and
copies the best weights to ``models/store_yolo11m.pt``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = 0.95


def summarise(metrics, split: str, data: str, weights: str) -> dict:
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "split": split, "data": data,
        "weights": weights, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4), "target": TARGET,
    }
    out["passed"] = {k: out[k] >= TARGET for k in ("precision", "recall", "f1", "map50")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/store_person/data.yaml")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto (60%% GPU memory)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="store_yolo11m")
    ap.add_argument("--eval-only", help="skip training; evaluate these weights")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    if args.eval_only:
        weights = args.eval_only
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=int(args.batch) if args.batch >= 1 else args.batch,
            device=args.device, patience=args.patience, project=str(ROOT / "runs"), name=args.name,
            exist_ok=True, single_cls=True, cos_lr=True, close_mosaic=15, seed=42,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=0.0, flipud=0.0, fliplr=0.5,
            scale=0.5, translate=0.1, mosaic=1.0, mixup=0.1, plots=True)
        weights = str(ROOT / "runs" / args.name / "weights" / "best.pt")
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy2(weights, ROOT / "models" / "store_yolo11m.pt")
        weights = str(ROOT / "models" / "store_yolo11m.pt")

    best = YOLO(weights)
    report = {}
    for split in ("val", "test"):
        try:
            m = best.val(data=args.data, split=split, imgsz=args.imgsz, device=args.device, plots=False,
                         verbose=False)
            report[split] = summarise(m, split, args.data, weights)
        except Exception as exc:  # test split may be empty for tiny datasets
            report[split] = {"error": str(exc)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "detector_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
