"""Step 5 - run the pipeline over every video listed in a manifest and (optionally) ingest.

  python -m training.batch_process training/videos_manifest.csv --model models/store_yolo11m.pt \
      --post http://localhost:8000

Manifest CSV columns: video,store_id,camera,start_time   (camera = CAM1 | CAM2 | CAM3 | CAM5)
CAM3 clips of a store are processed first so their Re-ID gallery is shared by that store.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ORDER = {"CAM3": 0, "CAM1": 1, "CAM2": 2, "CAM5": 3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--out-dir", default="data/pipeline_outputs")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--post", help="Ingest API base URL")
    args = ap.parse_args(argv)

    from pipeline.bridge import post_batches
    from pipeline.run_pipeline import process_video

    with open(args.manifest, newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: (r["store_id"], ORDER.get(r["camera"].upper(), 9)))
    for r in rows:
        store, cam = r["store_id"], r["camera"].upper()
        out = Path(args.out_dir) / store / f"{Path(r['video']).stem}_{cam.lower()}_events.jsonl"
        gallery = str(Path(args.out_dir) / store / "reid_gallery.json")
        events = process_video(r["video"], store, cam, r["start_time"], str(out), model=args.model,
                               fps=args.fps, conf=args.conf, imgsz=args.imgsz, device=args.device,
                               reid_gallery=gallery)
        print(f"{r['video']} [{store}/{cam}] -> {len(events)} events")
        if args.post and events:
            print("   ingest:", post_batches(args.post, "/events/ingest", events))


if __name__ == "__main__":
    main()
