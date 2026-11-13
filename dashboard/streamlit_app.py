"""Store Intelligence dashboard (Streamlit). Talks to the FastAPI service over HTTP only.

  API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py --server.port 8501
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Store Intelligence", page_icon="🛍️", layout="wide")


def _is_dark() -> bool:
    """Server-configured dark theme, or the viewer's active theme (Streamlit >= 1.46)."""
    if (st.get_option("theme.base") or "") == "dark":
        return True
    try:
        return (st.context.theme.type or "light") == "dark"
    except Exception:
        return False


DARK = _is_dark()
# Validated reference palette (dataviz skill), with separate steps for light and dark surfaces.
if DARK:
    SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
    SEQ_BLUE = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]  # low recedes into the dark surface
    MUTED = "#898781"
else:
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
    MUTED = "#898781"
GRID, AXIS = "rgba(137,135,129,0.22)", "rgba(137,135,129,0.45)"  # recede on light and dark alike
STATUS = {"critical": "#d03b3b", "warning": "#fab219", "info": SERIES[0], "good": "#0ca30c"}
ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
LEVEL_ICON = {"critical": "🚨", "warning": "⚠️", "info": "💡", "good": "✅"}
ZONE_CLASS_COLOR = {"hot": SERIES[1], "warm": SERIES[3], "cold": SERIES[0]}
QUADRANT_COLOR = {"star": SERIES[0], "problem": SERIES[1], "potential": SERIES[2], "obsolete": MUTED}


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None),
                      font=dict(size=13))  # text colours follow the Streamlit theme
    fig.update_xaxes(showgrid=False, linecolor=AXIS, title_font_color=MUTED, tickfont_color=MUTED)
    if fig.layout.xaxis.title.text in ("hour", None):  # time axes are self-explanatory
        fig.update_xaxes(title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, title_font_color=MUTED, tickfont_color=MUTED)
    return fig


@st.cache_data(ttl=30, show_spinner=False)
def api(path: str, **params):
    r = requests.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    r.raise_for_status()
    return r.json()


def inr(x: float) -> str:
    return f"₹{x:,.2f}"


def inr_short(x: float) -> str:
    """Indian short scale for stat tiles: ₹92.8K, ₹1.2L, ₹3.4Cr."""
    for div, suf in ((1e7, "Cr"), (1e5, "L"), (1e3, "K")):
        if abs(x) >= div:
            return f"₹{x / div:.1f}{suf}"
    return f"₹{x:,.0f}"


def table(df: pd.DataFrame, name: str) -> None:
    """Dataframe + CSV download (every table in the dashboard is exportable)."""
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(f"⬇ {name}.csv", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv",
                       key=f"dl_{name}")


def pct_bar(label: str, value, target: float = 0.95):
    """Progress-style row for an accuracy figure against the 95% target."""
    if value is None:
        st.markdown(f"**{label}** · –")
        return
    ok = value >= target
    st.markdown(f"{'✅' if ok else '⚠️'} **{label}** · {100 * value:.1f}% "
                f"<span style='color:{MUTED}'>(target {100 * target:.0f}%)</span>", unsafe_allow_html=True)
    st.progress(min(1.0, float(value)))


# ---------------------------------------------------------------- sidebar / filters
try:
    stores = api("/stores")
except requests.RequestException as exc:
    st.error(f"Cannot reach the API at {API} — start it with `uvicorn app.main:app --port 8000`.\n\n{exc}")
    st.stop()
if not stores:
    st.warning("No data yet. Seed it with `python scripts/seed_demo.py`.")
    st.stop()

st.sidebar.title("🛍️ Store Intelligence")
labels = {s["store_id"]: s.get("name", s["store_id"]) for s in stores}
store_id = st.sidebar.selectbox("Store", list(labels), format_func=labels.get)
st.sidebar.caption(f"ID: `{store_id}`")
store = next(s for s in stores if s["store_id"] == store_id)
dates = store["dates"] or [None]
day = st.sidebar.selectbox("Date", dates[::-1],
                           format_func=lambda d: f"{pd.Timestamp(d):%a, %d %b %Y}" if d else "All")
with st.sidebar.expander("POS match window"):
    wb = st.slider("Minutes before POS", 1, 15, 5)
    wa = st.slider("Grace after POS", 0, 5, 1)
if st.sidebar.button("↻ Refresh"):
    st.cache_data.clear()
with st.sidebar.expander("📖 How to read this dashboard"):
    st.markdown("""
- **Visitors**: unique people who entered (a person who leaves and comes back counts once).
- **Sessions**: individual visits, re-entries included.
- **Converted**: visitors whose billing-counter activity matched a POS bill within 5 minutes.
- **Conversion rate**: converted ÷ visitors.
- **Attributed revenue**: the part of POS revenue linked to a tracked visitor.
- **Dwell**: time spent in front of a shelf zone.
- **Queue abandonment**: shoppers who joined the billing queue and left without paying.
- **Hot / warm / cold zone**: high traffic with long dwell / medium / low or no traffic.
- **PCI (path complexity)**: 1 = direct path; higher = more wandering.
- **Hesitation**: lingering > 60 s or returning to the same shelf; a cue for staff help.
- **Data health**: how far the camera and POS feeds can be trusted for the period.
""")
with st.sidebar.expander("⚙️ About"):
    st.caption(f"API: {API}")
    st.caption("Analytics are rebuilt from raw events on every request.")
P = {"store_id": store_id, "date": day, "window_before_min": wb, "window_after_min": wa}

m = api("/metrics", **P)
summ = api("/insights/summary", **P)
oh = store.get("open_hours") or {}
st.title(store.get("name", store_id))
st.caption(f"{pd.Timestamp(day):%A, %d %B %Y} · open {oh.get('open', '?')}–{oh.get('close', '?')} · "
           f"{m['data_quality']['total_events']:,} camera events · {m['transactions']} POS bills" if day else store_id)

tabs = st.tabs(["Overview", "Zones & heatmap", "Journeys", "Queue & anomalies", "Conversions",
                "Forecast", "Sessions", "Store comparison", "Model accuracy"])


def benchmark() -> dict:
    """Average of every *other* store-day with footfall, used as the KPI comparison baseline."""
    try:
        rows = [r for r in api("/compare") if r["visitors"] and not (r["store_id"] == store_id and r["date"] == day)]
    except requests.RequestException:
        rows = []
    if not rows:
        return {}
    keys = ("visitors", "conversion_rate", "revenue_inr", "avg_basket_inr", "avg_session_duration_s",
            "avg_zone_dwell_s", "queue_abandonment_rate")
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys} | {"n": len(rows)}


def delta(cur, key, bench, fmt="{:+.0f}", pct=False):
    if not bench or key not in bench or not bench[key]:
        return None
    d = cur - bench[key]
    return (fmt.format(d) + (" pts" if pct else "")) + " vs avg"


def insight_card(i: dict) -> None:
    with st.container(border=True):
        st.markdown(f"{LEVEL_ICON[i['level']]} **{i['title']}**  \n{i['detail']}"
                    + (f"  \n➡️ *{i['action']}*" if i.get("action") else ""))


# ---------------------------------------------------------------- overview
with tabs[0]:
    # -- key insights: the story first, numbers second
    st.subheader("Key insights")
    ins = summ["insights"]
    cols = st.columns(2)
    for k, i in enumerate(ins[:6]):
        with cols[k % 2]:
            insight_card(i)
    if len(ins) > 6:
        with st.expander(f"Show {len(ins) - 6} more insights"):
            cols = st.columns(2)
            for k, i in enumerate(ins[6:]):
                with cols[k % 2]:
                    insight_card(i)

    # -- KPIs with context
    bench = benchmark()
    st.subheader("Today's numbers")
    if bench:
        st.caption(f"Arrows compare with the average of {bench['n']} other store-day(s) in the system.")
    c = st.columns(6)
    c[0].metric("Visitors", m["visitors"], delta(m["visitors"], "visitors", bench),
                help="Unique people who entered. Someone who leaves and comes back counts once.")
    c[1].metric("Sessions", m["sessions"], f"{m['reentry_sessions']} re-entries", delta_color="off",
                help="Individual visits, including re-entries.")
    c[2].metric("Converted", m["converted_visitors"],
                help="Visitors whose billing activity matched a POS bill within the match window.")
    c[3].metric("Conversion rate", f"{m['conversion_rate']}%",
                delta(m["conversion_rate"], "conversion_rate", bench, "{:+.1f}", pct=True),
                help="Converted visitors ÷ visitors.")
    c[4].metric("Revenue", inr_short(m["revenue_inr"]),
                f"{100 * m['attributed_revenue_inr'] / (m['revenue_inr'] or 1):.0f}% linked to visitors",
                delta_color="off",
                help=f"{inr(m['revenue_inr'])} total POS revenue; {inr(m['attributed_revenue_inr'])} "
                     "is linked to tracked visitors.")
    c[5].metric("Avg basket", inr_short(m["avg_basket_inr"]),
                delta(m["avg_basket_inr"], "avg_basket_inr", bench, "₹{:+,.0f}"),
                help=f"Average bill value: {inr(m['avg_basket_inr'])}")
    c = st.columns(6)
    c[0].metric("Avg visit", f"{m['avg_session_duration_s'] / 60:.1f} min",
                delta(m["avg_session_duration_s"] / 60, "avg_visit_min",
                      {"avg_visit_min": bench.get("avg_session_duration_s", 0) / 60} if bench else {}, "{:+.1f} min"),
                help="Average time from entry to exit.")
    c[1].metric("Avg zone dwell", f"{m['avg_zone_dwell_s']:.0f} s",
                delta(m["avg_zone_dwell_s"], "avg_zone_dwell_s", bench, "{:+.0f} s"),
                help="Average time a shopper spends in front of one shelf zone.")
    c[2].metric("Zones / visit", m["avg_zones_per_session"], help="Distinct shelf zones visited per visit.")
    c[3].metric("Queue joins", m["queue"]["joins"], help="Times a shopper joined the billing queue.")
    c[4].metric("Queue abandonment", f"{m['queue']['abandonment_rate']}%",
                delta(m["queue"]["abandonment_rate"], "queue_abandonment_rate", bench, "{:+.1f}", pct=True),
                delta_color="inverse", help="Share of queue joins that left without paying. Lower is better.")
    c[5].metric("Avg wait to pay", f"{m['queue']['avg_wait_to_pay_s'] / 60:.1f} min",
                help="Average time from joining the queue to the matched POS bill.")

    # -- funnel + hourly traffic
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Conversion funnel**")
        level = st.radio("Count", ["visitor", "session"], horizontal=True, label_visibility="collapsed",
                         format_func={"visitor": "Unique visitors", "session": "All visits"}.get)
        f = pd.DataFrame(api("/funnel", **P, level=level)["stages"])
        fig = go.Figure(go.Funnel(y=f["stage"], x=f["count"], textinfo="value+percent initial",
                                  marker=dict(color=SERIES[0]), connector=dict(fillcolor=GRID, line=dict(width=0)),
                                  textfont=dict(color="#ffffff"), customdata=f["description"],
                                  hovertemplate="%{y}: %{x}<br>%{customdata}<extra></extra>"))
        st.plotly_chart(style(fig, 320).update_layout(margin=dict(t=8)), use_container_width=True)
        lost = [f"**−{r.drop_off_from_prev}** {f.stage[i - 1]}→{r.stage}" for i, r in f.iterrows()
                if i and r.drop_off_from_prev]
        if lost:
            st.caption("Shoppers lost at each step: " + " · ".join(lost))
    ts = pd.DataFrame(api("/timeseries", **P)["hourly"])
    with right:
        st.markdown("**Shoppers entering per hour**")
        if not ts.empty:
            ts["hour"] = pd.to_datetime(ts["hour"])
            pk = ts.loc[ts["entries"].idxmax()]
            fig = px.line(ts, x="hour", y="entries", markers=True, color_discrete_sequence=[SERIES[0]],
                          labels={"entries": "shoppers"})
            fig.update_traces(line_width=2, marker_size=8)
            fig.add_annotation(x=pk["hour"], y=pk["entries"], text=f"Peak {pk['hour']:%H:00} · {int(pk['entries'])}",
                               showarrow=True, arrowcolor=MUTED, yshift=6)
            st.plotly_chart(style(fig, 320).update_layout(margin=dict(t=8)), use_container_width=True)
            st.caption("Use the peak to plan staff breaks and extra billing counters.")
    if not ts.empty:
        st.markdown("**POS revenue per hour (₹)**")
        fig = px.bar(ts, x="hour", y="revenue_inr", color_discrete_sequence=[SERIES[2]],
                     labels={"revenue_inr": "₹"}, hover_data={"transactions": True, "conversions": True})
        st.plotly_chart(style(fig, 260).update_layout(margin=dict(t=8)), use_container_width=True)

    # -- data health instead of raw JSON
    h = summ["health"]
    st.subheader("Data health")
    hc = st.columns([1, 3])
    hc[0].metric("Health score", f"{h['score']}/100", h["grade"],
                 delta_color="normal" if h["score"] >= 85 else "off" if h["score"] >= 60 else "inverse",
                 help="How far this period's camera and POS data can be trusted.")
    with hc[1]:
        for chk in h["checks"]:
            st.markdown(f"{'✅' if chk['ok'] else '⚠️'} **{chk['check']}** · {chk['detail']}")
    with st.expander("Technical details"):
        st.json(m["data_quality"])

# ---------------------------------------------------------------- zones
with tabs[1]:
    h = api("/heatmap", **P)
    z = pd.DataFrame(h["zones"])
    s = h["summary"]
    c = st.columns(4)
    c[0].metric("Hot zones", s["hot"], f"{s['hot_zone_traffic_share_pct']}% of traffic", delta_color="off")
    c[1].metric("Warm zones", s["warm"])
    c[2].metric("Cold zones", s["cold"])
    c[3].metric("Zones tracked", s["zones"])
    if not z.empty:
        fig = px.bar(z.sort_values("engagement_score"), x="engagement_score", y="zone_id", orientation="h",
                     color="zone_class", color_discrete_map=ZONE_CLASS_COLOR,
                     category_orders={"zone_class": ["hot", "warm", "cold"]},
                     hover_data=["visits", "unique_visitors", "avg_dwell_s", "traffic_share_pct"],
                     title="Zone engagement score (0–100)", labels={"engagement_score": "score", "zone_id": ""})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style(fig, max(340, 26 * len(z))), use_container_width=True)
        mx = h["hour_zone_matrix"]
        if mx["hours"]:
            fig = go.Figure(go.Heatmap(z=mx["values"], x=[f"{x:02d}:00" for x in mx["hours"]], y=mx["zones"],
                                       colorscale=[[i / 4, c] for i, c in enumerate(SEQ_BLUE)],
                                       hovertemplate="%{y} @ %{x}: %{z} visits<extra></extra>", xgap=2, ygap=2))
            st.plotly_chart(style(fig.update_layout(title="Zone visits by hour"), max(320, 24 * len(mx["zones"]))),
                            use_container_width=True)
        table(z, f"zones_{store_id}_{day}")
    q = api("/insights/shelf", **P)
    if q["zones"]:
        qd = pd.DataFrame(q["zones"])
        fig = px.scatter(qd, x="attention_pct", y="conversion_pct", color="quadrant", text="zone_id",
                         symbol="quadrant", color_discrete_map=QUADRANT_COLOR, hover_data=["viewers", "buyers", "strategy"],
                         title="Shelf attention × conversion quadrant",
                         labels={"attention_pct": "Attention (% of visits)", "conversion_pct": "Conversion (% of viewers)"})
        fig.update_traces(marker_size=11, textposition="top center", textfont_color=MUTED)
        fig.add_vline(x=q["thresholds"]["attention_pct"], line_dash="dot", line_color=AXIS)
        fig.add_hline(y=q["thresholds"]["conversion_pct"], line_dash="dot", line_color=AXIS)
        st.plotly_chart(style(fig, 420), use_container_width=True)
        table(qd[["zone_id", "quadrant", "attention_pct", "conversion_pct", "viewers", "buyers", "strategy"]],
              f"shelf_quadrant_{store_id}_{day}")

# ---------------------------------------------------------------- journeys
with tabs[2]:
    j = api("/insights/journeys", **P)
    c = st.columns(3)
    c[0].metric("Mean PCI", j["pci"]["mean"] if j["pci"]["mean"] is not None else "–")
    c[1].metric("PCI std-dev", j["pci"]["std"] if j["pci"]["std"] is not None else "–")
    c[2].metric("Median PCI", j["pci"]["median"] if j["pci"]["median"] is not None else "–")
    st.caption("Path Complexity Index = actual path / shortest tour through the same zones + 0.1 × extra turns. "
               "Efficient < 1.5 ≤ browsing ≤ 2.5 < roaming (Chen et al., 2026).")
    seg = pd.DataFrame(j["segments"])
    l, r = st.columns(2)
    with l:
        fig = px.bar(seg, x="segment", y="sessions", title="Shopper segments by PCI", text="share_pct",
                     color_discrete_sequence=[SERIES[0]])
        fig.update_traces(texttemplate="%{text}%")
        st.plotly_chart(style(fig), use_container_width=True)
    with r:
        fig = px.bar(seg, x="segment", y="conversion_rate", title="Conversion rate by segment (%)",
                     color_discrete_sequence=[SERIES[2]])
        st.plotly_chart(style(fig), use_container_width=True)
    if j["top_flows"]:
        st.subheader("Top zone-to-zone flows")
        table(pd.DataFrame(j["top_flows"]), f"zone_flows_{store_id}_{day}")
    hz = api("/insights/hesitation", **P)
    st.subheader("Service touch-points (hesitation signals)")
    st.caption(f"Dwell timeout > {hz['thresholds']['dwell_timeout_s']:.0f}s or zone revisited "
               f"≥ {hz['thresholds']['loiter_revisits']}×. Levels: " +
               " · ".join(f"**{k}**: {v}" for k, v in hz["actions"].items()))
    if hz["zones"]:
        table(pd.DataFrame(hz["zones"]), f"hesitation_{store_id}_{day}")

# ---------------------------------------------------------------- queue & anomalies
with tabs[3]:
    a = api("/anomalies", **P)
    c = st.columns(3)
    for col, k in zip(c, ("critical", "warning", "info")):
        col.metric(f"{ICON[k]} {k.title()}", a["counts"][k])
    if not ts.empty:
        l, r = st.columns(2)
        with l:
            fig = px.line(ts, x="hour", y="max_queue_depth", markers=True, title="Peak queue depth per hour",
                          color_discrete_sequence=[SERIES[1]])
            fig.add_hline(y=8, line_dash="dot", line_color=STATUS["critical"], annotation_text="alert threshold")
            st.plotly_chart(style(fig), use_container_width=True)
        with r:
            qd = ts.melt(id_vars="hour", value_vars=["queue_joins", "queue_abandons"], var_name="event",
                         value_name="count")
            fig = px.bar(qd, x="hour", y="count", color="event", barmode="group", title="Queue joins vs abandons",
                         color_discrete_sequence=[SERIES[0], SERIES[1]])
            st.plotly_chart(style(fig), use_container_width=True)
    for an in a["anomalies"]:
        when = f" · {an['timestamp'][:16].replace('T', ' ')}" if an.get("timestamp") else ""
        zone = f" · {an['zone_id']}" if an.get("zone_id") else ""
        with st.container(border=True):
            st.markdown(f"{ICON[an['severity']]} **{an['title']}** `{an['type']}`{when}{zone}  \n"
                        f"{an['detail']}  \n➡️ _{an['suggested_action']}_")
    if not a["anomalies"]:
        st.success("No anomalies detected for this period.")
    else:
        st.download_button("⬇ anomalies.csv", pd.DataFrame(a["anomalies"]).to_csv(index=False).encode(),
                           f"anomalies_{store_id}_{day}.csv", "text/csv")

# ---------------------------------------------------------------- conversions
with tabs[4]:
    cv = pd.DataFrame(api("/conversions", **P)["matches"])
    if cv.empty:
        st.info("No POS transactions in this period.")
    else:
        matched = cv["session_id"].notna()
        c = st.columns(3)
        c[0].metric("Transactions", len(cv))
        c[1].metric("Matched to a visitor", int(matched.sum()), f"{100 * matched.mean():.0f}%", delta_color="off")
        c[2].metric("Median billing→POS gap", f"{cv.loc[matched, 'gap_s'].median() / 60:.1f} min"
                    if matched.any() else "–")
        if matched.any():
            fig = px.histogram(cv[matched], x="gap_s", nbins=20, title="Billing activity → POS gap (seconds)",
                               color_discrete_sequence=[SERIES[0]])
            st.plotly_chart(style(fig, 280), use_container_width=True)
        table(cv, f"conversions_{store_id}_{day}")

# ---------------------------------------------------------------- forecast
with tabs[5]:
    horizon = st.slider("Forecast horizon (hours)", 1, 24, 6)
    fc = api("/forecast", **P, horizon=horizon)
    if fc.get("history"):
        hist = pd.DataFrame(fc["history"])
        fut = pd.DataFrame(fc["footfall"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.concat([fut["hour"], fut["hour"][::-1]]),
                                 y=pd.concat([fut["upper"], fut["lower"][::-1]]), fill="toself",
                                 fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="95% band", mode="lines",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist["hour"], y=hist["entries"], name="Actual entries", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=fut["hour"], y=fut["entries"], name="Forecast (open hours)", mode="lines+markers",
                                 line=dict(color=SERIES[1], width=2, dash="dash"), marker=dict(size=8)))
        st.plotly_chart(style(fig.update_layout(title=f"Footfall forecast · {fc['method']}"), 380),
                        use_container_width=True)
        st.caption(f"In-sample MAE: {fc['in_sample_mae']} entries/hour")
        fm = fc.get("fusion_model")
        if fm:
            st.subheader("CCTV + POS fusion model")
            st.markdown(f"Hourly revenue explained by CCTV signals over {fm['n_hours']} hours — "
                        f"**R² = {fm['r2']}** _(small sample; indicative only)_")
            coef = pd.DataFrame(fm["standardised_coefficients"].items(), columns=["signal", "₹ per 1 σ"])
            fig = px.bar(coef, x="₹ per 1 σ", y="signal", orientation="h", color_discrete_sequence=[SERIES[2]])
            st.plotly_chart(style(fig, 220), use_container_width=True)

# ---------------------------------------------------------------- sessions
with tabs[6]:
    only = st.selectbox("Filter", ["all", "converted", "not converted"])
    conv = {"all": None, "converted": True, "not converted": False}[only]
    ss = api("/sessions", **P, converted=conv, limit=2000)
    sdf = pd.DataFrame(ss["sessions"])
    st.caption(f"{ss['total']} sessions")
    if not sdf.empty:
        table(sdf.assign(zones=sdf["zones"].map(" → ".join)), f"sessions_{store_id}_{day}")
        pick = st.selectbox("Inspect session", sdf["session_id"])
        d = api(f"/sessions/{pick}", **P)
        zv = pd.DataFrame(d["zone_visits"])
        if not zv.empty and zv["exit"].notna().any():
            zv["exit"] = zv["exit"].fillna(zv["enter"])
            fig = px.timeline(zv, x_start="enter", x_end="exit", y="zone_id", title=f"Journey of {pick}",
                              color_discrete_sequence=[SERIES[0]], hover_data=["dwell_s"])
            st.plotly_chart(style(fig, 260), use_container_width=True)
        ev = pd.DataFrame(d["events"])
        if not ev.empty:
            st.dataframe(ev[["timestamp", "event_type", "camera_id", "zone_id", "dwell_ms", "confidence"]],
                         use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- store comparison
with tabs[7]:
    cmp_ = pd.DataFrame(api("/compare"))
    if cmp_.empty:
        st.info("No stores with data yet.")
    else:
        cmp_["store_day"] = cmp_["store_id"] + " · " + cmp_["date"]
        st.caption("Each store on every day with data. Charts use one measure each (no dual axes).")
        cols = st.columns(2)
        for i, (col, title) in enumerate([("visitors", "Visitors"), ("conversion_rate", "Conversion rate (%)"),
                                          ("revenue_inr", "Revenue (₹)"),
                                          ("queue_abandonment_rate", "Queue abandonment (%)")]):
            fig = px.bar(cmp_, x="store_day", y=col, title=title, text=col, color_discrete_sequence=[SERIES[i % 3]])
            fig.update_traces(texttemplate="%{text:,.4~f}", textposition="outside")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            cols[i % 2].plotly_chart(style(fig, 300), use_container_width=True)
        table(cmp_.drop(columns=["store_day"]), "store_comparison")

# ---------------------------------------------------------------- model accuracy
with tabs[8]:
    ev = api("/evaluation")
    target = ev["target"]
    det, evr = ev.get("detector"), ev.get("events")
    st.subheader("Person detector (YOLO11m fine-tuned on store footage)")
    if det and det.get("test") and "precision" in det["test"]:
        t = det["test"]
        st.caption(f"Held-out **test videos** · weights `{t['weights']}` · {t['generated_at']}")
        c = st.columns(5)
        for col, k in zip(c, ("precision", "recall", "f1", "map50", "map50_95")):
            col.metric(k.replace("_", "-").upper() if k.startswith("map") else k.title(), f"{100 * t[k]:.1f}%")
        for k in ("precision", "recall", "f1", "map50"):
            pct_bar(k.upper() if k == "map50" else k.title(), t[k], target)
        if det.get("val") and "precision" in det["val"]:
            with st.expander("Validation split"):
                st.json(det["val"])
    else:
        st.info("No detector report yet. Train on your videos (see `training/README.md`):\n\n"
                "`extract_frames → prelabel → review labels → build_dataset → train_detector`\n\n"
                "The training step writes `reports/detector_metrics.json`, which appears here.")
    st.subheader("End-to-end event accuracy (pipeline vs hand-labelled ground truth)")
    if evr:
        o = evr["overall"]
        c = st.columns(4)
        c[0].metric("Event precision", f"{100 * o['precision']:.1f}%")
        c[1].metric("Event recall", f"{100 * o['recall']:.1f}%")
        c[2].metric("Event F1", f"{100 * o['f1']:.1f}%")
        c[3].metric("Labelled events", evr["n_truth"])
        pct_bar("Overall event F1", o["f1"], target)
        for k, v in evr["counting_accuracy"].items():
            pct_bar(f"Counting accuracy · {k.replace('_', ' ')}", v, target)
        if evr["dwell"]["within_20pct"] is not None:
            pct_bar("Dwell within ±20%", evr["dwell"]["within_20pct"], target)
        per = pd.DataFrame([{"event_type": k, **v} for k, v in evr["per_event_type"].items()])
        fig = px.bar(per, x="event_type", y="f1", title="F1 by event type", color_discrete_sequence=[SERIES[0]],
                     range_y=[0, 1.05])
        fig.add_hline(y=target, line_dash="dot", line_color=STATUS["good"], annotation_text="95% target")
        st.plotly_chart(style(fig, 320), use_container_width=True)
        table(per, "event_accuracy_by_type")
    else:
        st.info("No event-accuracy report yet. Label ground truth with "
                "`streamlit run training/label_events_app.py -- --videos videos/`, then run "
                "`python -m training.evaluate_events --pred data/pipeline_outputs --truth training/ground_truth/<store>.csv`.")

st.caption(f"Rendered {datetime.now():%Y-%m-%d %H:%M:%S}")
