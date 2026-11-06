"""Step 1 - sample diverse frames from the store videos for annotation.

  python -m training.extract_frames videos/ --out datasets/raw/images --every 1.0

* one frame every ``--every`` seconds, skipping near-duplicates (static scenes add no signal)
* file names are ``<video-stem>__<frame>.jpg`` so the dataset can later be split *by video*
  (frames of one clip never leak between train and test, which would inflate accuracy)
"""
from __future__ import annotations

import argparse
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".dav", ".h264"}


def extract(video: Path, out: Path, every_s: float = 1.0, diff_thresh: float = 4.0, max_frames: int = 0) -> int:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * every_s))
    out.mkdir(parents=True, exist_ok=True)
    prev, idx, saved = None, 0, 0
    stem = video.parent.name + "_" + video.stem if video.parent.name else video.stem
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev is None or float(np.mean(np.abs(small - prev))) >= diff_thresh:
                    cv2.imwrite(str(out / f"{stem}__{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    prev = small
                    if max_frames and saved >= max_frames:
                        break
        idx += 1
    cap.release()
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", help="video file or directory (searched recursively)")
    ap.add_argument("--out", default="datasets/raw/images")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--diff", type=float, default=4.0, help="min mean pixel change to keep a frame")
    ap.add_argument("--max-per-video", type=int, default=0)
    args = ap.parse_args(argv)
    src = Path(args.videos)
    vids = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    total = 0
    for v in vids:
        n = extract(v, Path(args.out), args.every, args.diff, args.max_per_video)
        total += n
        print(f"{v}: {n} frames")
    print(f"total {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
