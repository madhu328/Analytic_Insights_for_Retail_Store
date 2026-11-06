"""Step 6 - measure end-to-end accuracy: pipeline events vs. hand-labelled ground truth.

  python -m training.evaluate_events --pred data/pipeline_outputs/store_1 \
      --truth training/ground_truth/store_1.csv --out reports/event_accuracy.json

Ground truth CSV columns (one row per real event; label with ``training/label_events_app.py``):
  store_id,camera_id,event_type,timestamp,zone_id,dwell_ms
Predicted and true events are matched one-to-one within a time tolerance per event type
(same type, same camera and same zone). Visitor ids are *not* required to match, so the
score measures what the business uses: did the system see the entry / zone visit / queue
join at the right time and place.

Reported:
* per event type: TP / FP / FN, precision, recall, F1, mean timing error
* overall micro precision / recall / F1 ("event accuracy")
* counting accuracy = 1 - |pred - true| / true for entries, exits, zone visits, queue joins
* dwell accuracy = share of matched ZONE_DWELL within ±20% of true dwell, plus MAE
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TOLERANCE_S = {"ENTRY": 2.0, "EXIT": 2.0, "REENTRY": 2.0, "ZONE_ENTER": 3.0, "ZONE_EXIT": 3.0,
               "ZONE_DWELL": 5.0, "BILLING_QUEUE_JOIN": 5.0, "BILLING_QUEUE_ABANDON": 5.0}
TARGET = 0.95


def _ts(v) -> datetime:
    if isinstance(v, datetime):
        dt = v
    else:
        dt = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def load_pred(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        pp = Path(p)
        files = sorted(pp.rglob("*.jsonl")) if pp.is_dir() else [pp]
        for f in files:
            out += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return [e for e in out if not e.get("is_staff")]


def load_truth(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["dwell_ms"] = int(float(r.get("dwell_ms") or 0))
        r["zone_id"] = (r.get("zone_id") or "").strip().upper() or None
    return rows


def _key(e: dict) -> tuple:
    zone = (e.get("zone_id") or "").upper() or None
    if e["event_type"] in ("ENTRY", "EXIT", "REENTRY"):
        zone = None
    return e["event_type"], e.get("camera_id"), zone


def match(pred: list[dict], truth: list[dict], tolerance: dict | None = None):
    """Greedy one-to-one matching by smallest time error inside the tolerance window."""
    tol = {**TOLERANCE_S, **(tolerance or {})}
    pg, tg = defaultdict(list), defaultdict(list)
    for i, e in enumerate(pred):
        pg[_key(e)].append(i)
    for j, e in enumerate(truth):
        tg[_key(e)].append(j)
    pairs = []
    for k in set(pg) | set(tg):
        t_lim = tol.get(k[0], 3.0)
        cand = []
        for i in pg.get(k, []):
            ti = _ts(pred[i]["timestamp"])
            for j in tg.get(k, []):
                d = abs((ti - _ts(truth[j]["timestamp"])).total_seconds())
                if d <= t_lim:
                    cand.append((d, i, j))
        cand.sort()
        used_p, used_t = set(), set()
        for d, i, j in cand:
            if i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                pairs.append((i, j, d))
    return pairs


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def evaluate(pred: list[dict], truth: list[dict], tolerance: dict | None = None) -> dict:
    pairs = match(pred, truth, tolerance)
    mp = {i: (j, d) for i, j, d in pairs}
    mt = {j for _, j, _ in pairs}
    types = sorted({e["event_type"] for e in pred} | {e["event_type"] for e in truth})
    per = {}
    TP = FP = FN = 0
    for t in types:
        tp = sum(1 for i, (j, d) in mp.items() if pred[i]["event_type"] == t)
        fp = sum(1 for i, e in enumerate(pred) if e["event_type"] == t and i not in mp)
        fn = sum(1 for j, e in enumerate(truth) if e["event_type"] == t and j not in mt)
        errs = [d for i, (j, d) in mp.items() if pred[i]["event_type"] == t]
        p, r, f = _prf(tp, fp, fn)
        per[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
                  "true_count": tp + fn, "pred_count": tp + fp,
                  "mean_time_error_s": round(sum(errs) / len(errs), 2) if errs else None}
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    p, r, f = _prf(TP, FP, FN)

    def count_acc(types_):
        g = sum(1 for e in truth if e["event_type"] in types_)
        q = sum(1 for e in pred if e["event_type"] in types_)
        return round(max(0.0, 1 - abs(q - g) / g), 4) if g else None

    dwell = [(pred[i]["dwell_ms"], truth[j]["dwell_ms"]) for i, j, _ in pairs
             if pred[i]["event_type"] == "ZONE_DWELL" and truth[j]["dwell_ms"] > 0]
    within = [abs(a - b) <= 0.2 * b for a, b in dwell]
    zone_counts = defaultdict(lambda: [0, 0])
    for e in truth:
        if e["event_type"] == "ZONE_ENTER":
            zone_counts[e["zone_id"]][0] += 1
    for e in pred:
        if e["event_type"] == "ZONE_ENTER":
            zone_counts[(e.get("zone_id") or "").upper()][1] += 1
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET, "n_pred": len(pred), "n_truth": len(truth),
        "overall": {"precision": p, "recall": r, "f1": f, "tp": TP, "fp": FP, "fn": FN},
        "per_event_type": per,
        "counting_accuracy": {
            "entries": count_acc({"ENTRY", "REENTRY"}), "exits": count_acc({"EXIT"}),
            "zone_visits": count_acc({"ZONE_ENTER"}), "queue_joins": count_acc({"BILLING_QUEUE_JOIN"}),
        },
        "zone_visit_counts": {z: {"true": c[0], "pred": c[1]} for z, c in sorted(zone_counts.items()) if z},
        "dwell": {"matched": len(dwell),
                  "within_20pct": round(sum(within) / len(within), 4) if within else None,
                  "mae_s": round(sum(abs(a - b) for a, b in dwell) / len(dwell) / 1000, 2) if dwell else None},
        "tolerance_s": {**TOLERANCE_S, **(tolerance or {})},
    }
    report["passed"] = {"event_f1": f >= TARGET,
                        **{f"count_{k}": (v is None or v >= TARGET) for k, v in report["counting_accuracy"].items()}}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, help="JSONL files or directories")
    ap.add_argument("--truth", required=True, help="ground-truth CSV")
    ap.add_argument("--out", default="reports/event_accuracy.json")
    ap.add_argument("--camera", help="evaluate one camera_id only")
    args = ap.parse_args(argv)
    pred, truth = load_pred(args.pred), load_truth(args.truth)
    if args.camera:
        pred = [e for e in pred if e["camera_id"] == args.camera]
        truth = [e for e in truth if e["camera_id"] == args.camera]
    rep = evaluate(pred, truth)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    o = rep["overall"]
    print(f"event precision {o['precision']:.3f}  recall {o['recall']:.3f}  F1 {o['f1']:.3f}  "
          f"(target {TARGET})  -> {args.out}")
    for t, v in rep["per_event_type"].items():
        print(f"  {t:24s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(true {v['true_count']}, pred {v['pred_count']})")


if __name__ == "__main__":
    main()
