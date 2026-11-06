"""Load the bundled dataset into the database (directly, or through the REST API).

  python scripts/seed_demo.py                 # direct DB write (DATABASE_URL or data/store_intelligence.db)
  python scripts/seed_demo.py --api http://localhost:8000
  python scripts/seed_demo.py --reset         # drop & recreate the SQLite file first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.bridge import post_batches, read_jsonl, read_pos_csv  # noqa: E402

EVENT_SOURCES = [
    "data/synthetic/store_1/synthetic_events_store_1.jsonl",
    "data/synthetic/store_2/synthetic_events_store_2.jsonl",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/pipeline_outputs").rglob("*.jsonl")),
]
POS_SOURCES = [
    "data/synthetic/store_1/synthetic_pos_store_1.csv",
    "data/synthetic/store_2/synthetic_pos_store_2.csv",
    "data/pos/purpple_pos_transactions.csv",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", help="POST through the API instead of writing the DB directly")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args(argv)
    events = read_jsonl(ROOT / p for p in EVENT_SOURCES)
    pos = read_pos_csv(ROOT / p for p in POS_SOURCES)
    if args.api:
        print("events", post_batches(args.api, "/events/ingest", events))
        print("pos   ", post_batches(args.api, "/pos/ingest", pos))
        return
    from app import db
    from app.config import settings
    from app.schemas import Event, PosTransaction, validate_batch

    if args.reset and settings.database_url.startswith("sqlite:///"):
        f = Path(settings.database_url.removeprefix("sqlite:///"))
        for suffix in ("", "-wal", "-shm"):
            Path(str(f) + suffix).unlink(missing_ok=True)
    eng = db.get_engine()
    ev_ok, ev_err = validate_batch(Event, events, "event_id")
    pos_ok, pos_err = validate_batch(PosTransaction, pos, "transaction_id")
    print("events: inserted=%d duplicates=%d rejected=%d" % (*db.insert_events(eng, ev_ok), len(ev_err)))
    print("pos:    inserted=%d duplicates=%d rejected=%d" % (*db.insert_pos(eng, pos_ok), len(pos_err)))
    for s in db.store_summaries(eng):
        print(f"  {s['store_id']}: {s['events']} events, {s['transactions']} txns, dates={s['dates']}")


if __name__ == "__main__":
    main()
