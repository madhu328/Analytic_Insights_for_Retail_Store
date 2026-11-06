"""Bridge: push pipeline JSONL events and POS CSVs to the Ingest API.

  python -m pipeline.bridge events data/synthetic/store_1/*.jsonl --api http://localhost:8000
  python -m pipeline.bridge pos data/pos/purpple_pos_transactions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


def read_jsonl(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as fh:
            out.extend(json.loads(line) for line in fh if line.strip())
    return out


def read_pos_csv(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                r["basket_value_inr"] = float(r["basket_value_inr"])
                out.append(r)
    return out


def post_batches(api: str, route: str, items: list[dict], batch: int = 500, retries: int = 3) -> dict:
    tot = {"received": 0, "inserted": 0, "duplicates": 0, "rejected": 0}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        for attempt in range(retries):
            try:
                r = requests.post(f"{api.rstrip('/')}{route}", json=chunk, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        res = r.json()
        for k in tot:
            tot[k] += res.get(k, 0)
        for err in res.get("errors", [])[:5]:
            print(f"  rejected #{i + err['index']}: {err['error']}", file=sys.stderr)
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["events", "pos"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args(argv)
    files = [str(p) for x in args.paths for p in (sorted(Path(x).rglob("*.jsonl")) if Path(x).is_dir() else [Path(x)])]
    if args.kind == "events":
        res = post_batches(args.api, "/events/ingest", read_jsonl(files))
    else:
        res = post_batches(args.api, "/pos/ingest", read_pos_csv(files))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
