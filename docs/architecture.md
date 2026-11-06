# Architecture & algorithms

## Event schema (pipeline → API)

```json
{"event_id": "uuid", "store_id": "STORE_BLR_002", "camera_id": "CAM_SHELF_01",
 "visitor_id": "VIS_00012", "event_type": "ZONE_DWELL", "timestamp": "2026-04-10T20:11:35.400Z",
 "zone_id": "NEUTROGENA", "dwell_ms": 30000, "is_staff": false, "confidence": 0.85,
 "metadata": {"queue_depth": null, "sku_zone": "NEUTROGENA", "session_seq": 3}}
```

`event_type` is one of `ENTRY, EXIT, REENTRY, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON`. Zone events require `zone_id`, `confidence` must be in [0, 1], and timestamps are normalised to UTC.

## Detection / label pipeline (per processed frame, ~5 fps)

1. **Video ingest:** OpenCV reads the MP4 for each camera and samples it to the target fps.
2. **YOLO11m:** person detections (class 0, conf ≥ 0.35).
3. **ByteTrack:** stable track IDs. Two-stage association uses the low-confidence boxes as well, which recovers occluded shoppers.
4. **CAM3 only:** OSNet embedding of each track's largest crop, matched against the `ReIDManager` gallery by cosine similarity (≥ 0.80, EMA update). `SessionStore` keeps each visitor's inside/outside state, so a known visitor who has exited and comes back produces `REENTRY`.
5. **Camera logic:** CAM1/2 zone polygons and dwell, CAM3 entry-line half-plane, CAM5 queue and payment zones.
6. **EventEmitter:** writes JSONL. It buffers events so that tracks later classed as staff are flagged before writing.
7. **Bridge:** POSTs the events to `/events/ingest`, after which POS matching and analytics happen at query time.

Anchor points: shelf cameras use the bbox centre, because the polygons are drawn on the shelf face. Floor cameras (doorway, queue) use the feet point (bottom-centre).

## Session builder
Events are grouped per visitor and sorted by `(timestamp, session_seq, type order)`.
- `ENTRY`/`REENTRY` opens a session. A second opener without an EXIT supersedes the open session.
- `EXIT` closes the session.
- `ZONE_ENTER/DWELL/EXIT` produce zone visits. Dwell comes from `ZONE_DWELL.dwell_ms` when present, otherwise from exit − enter.
- `BILLING_*` and the `BILLING`/`PAYMENT` zones count as billing activity.
- Events with no open session are counted as orphans and never silently attached.
- Staff events and events below `MIN_EVENT_CONFIDENCE` are excluded and reported.

## POS correlation
POS rows carry no identity. Transactions are processed chronologically, and each one takes the **earliest unmatched** billing signal in `[t − 5 min, t + 1 min]`, which is how a FIFO queue behaves. A queue join followed by an abandon is not a candidate. Confidence = `(1 − 0.5·|gap|/window) × min event confidence`.

## Metrics
- conversion rate = converted unique visitors / unique visitors
- revenue = all POS in the window; attributed revenue = matched POS only
- queue: joins, abandons, abandonment %, average and max depth, average join→pay wait

## Heatmap
engagement = 100 × (0.6 · visits/max_visits + 0.4 · avg_dwell/max_avg_dwell)

Zone class:
- hot = top-third traffic and avg dwell ≥ the median zone dwell
- cold = bottom-third traffic or zero visits
- warm = everything else

## Anomalies
| Type | Rule |
|---|---|
| FOOTFALL_SPIKE / DROP | robust z (median/MAD) of hourly entries beyond ±3 |
| QUEUE_CONGESTION | hourly max depth ≥ `QUEUE_DEPTH_ALERT`; critical at 1.5× |
| QUEUE_ABANDONMENT | > 30% of joins abandoned (≥ 3 joins) |
| CONVERSION_DROP | hourly conversion < 50% of the store average (≥ 5 entries) |
| UNMATCHED_POS | POS with no billing activity; severity grows with the share |
| DEAD_ZONE | configured product zone with 0 visits |
| LONG_SESSION | duration > Q3 + 3·IQR (possibly unflagged staff or loitering) |
| ORPHAN_EVENTS / UNCLOSED_SESSIONS | entrance-camera coverage problems |
| AFTER_HOURS | visits outside `open_hours` |
| CAMERA_SILENCE | one camera emits nothing for > 60 min while the others are active |

## Journeys (Chen et al., 2026, adapted)
PCI = L_actual / L_shortest-tour-through-same-zones + 0.1 · extra turns > 45°. The segments (efficient / browsing / roaming), the hesitation tiers and the shelf quadrant follow the paper.

## Forecast & fusion (Deepa G et al., 2026)
Damped Holt (α = 0.5, β = 0.2, φ = 0.85) on hourly entries, stepping only through open hours. An hour-of-day profile is added once more than one day exists. The fusion regression is revenue_h ~ visitors_h + avg_dwell_h + queue_joins_h (standardised OLS, R² reported). It is indicative only on a single day of data.
