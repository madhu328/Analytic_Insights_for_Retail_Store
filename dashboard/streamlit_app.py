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

# Validated reference palette (dataviz skill): categorical order, sequential blue, status.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
STATUS = {"critical": "#d03b3b", "warning": "#fab219", "info": "#2a78d6", "good": "#0ca30c"}
ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
GRID, AXIS, MUTED = "#e1e0d9", "#c3c2b7", "#898781"
ZONE_CLASS_COLOR = {"hot": SERIES[1], "warm": SERIES[3], "cold": SERIES[0]}
QUADRANT_COLOR = {"star": SERIES[0], "problem": SERIES[1], "potential": SERIES[2], "obsolete": MUTED}

st.set_page_config(page_title="Store Intelligence", page_icon="🛍️", layout="wide")


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None),
                      font=dict(size=13))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, title_font_color=MUTED)
    if fig.layout.xaxis.title.text in ("hour", None):  # time axes are self-explanatory
        fig.update_xaxes(title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, title_font_color=MUTED)
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
labels = {s["store_id"]: f"{s.get('name', s['store_id'])} · {s['store_id']}" for s in stores}
store_id = st.sidebar.selectbox("Store", list(labels), format_func=labels.get)
store = next(s for s in stores if s["store_id"] == store_id)
dates = store["dates"] or [None]
day = st.sidebar.selectbox("Date", dates[::-1])
with st.sidebar.expander("POS match window"):
    wb = st.slider("Minutes before POS", 1, 15, 5)
    wa = st.slider("Grace after POS", 0, 5, 1)
if st.sidebar.button("↻ Refresh"):
    st.cache_data.clear()
st.sidebar.caption(f"API: {API}")
P = {"store_id": store_id, "date": day, "window_before_min": wb, "window_after_min": wa}

m = api("/metrics", **P)
st.title(store.get("name", store_id))
st.caption(f"{store_id} · {day} · open {store.get('open_hours', {}).get('open', '?')}–"
           f"{store.get('open_hours', {}).get('close', '?')} · {m['data_quality']['total_events']} events")

tabs = st.tabs(["Overview", "Zones & heatmap", "Journeys", "Queue & anomalies", "Conversions",
                "Forecast", "Sessions"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Visitors", m["visitors"], help="Unique people (re-entries counted once)")
    c[1].metric("Sessions", m["sessions"], f"{m['reentry_sessions']} re-entries", delta_color="off")
    c[2].metric("Converted", m["converted_visitors"])
    c[3].metric("Conversion rate", f"{m['conversion_rate']}%")
    c[4].metric("Revenue", inr_short(m["revenue_inr"]),
                f"{100 * m['attributed_revenue_inr'] / (m['revenue_inr'] or 1):.0f}% attributed", delta_color="off",
                help=f"{inr(m['revenue_inr'])} total POS · {inr(m['attributed_revenue_inr'])} matched to CCTV visitors")
    c[5].metric("Avg basket", inr_short(m["avg_basket_inr"]), help=inr(m["avg_basket_inr"]))
    c = st.columns(6)
    c[0].metric("Avg visit", f"{m['avg_session_duration_s'] / 60:.1f} min")
    c[1].metric("Avg zone dwell", f"{m['avg_zone_dwell_s']:.0f} s")
    c[2].metric("Zones / visit", m["avg_zones_per_session"])
    c[3].metric("Queue joins", m["queue"]["joins"])
    c[4].metric("Queue abandonment", f"{m['queue']['abandonment_rate']}%")
    c[5].metric("Avg wait to pay", f"{m['queue']['avg_wait_to_pay_s'] / 60:.1f} min")

    left, right = st.columns([1, 1])
    with left:
        level = st.radio("Funnel level", ["visitor", "session"], horizontal=True)
        f = pd.DataFrame(api("/funnel", **P, level=level)["stages"])
        fig = go.Figure(go.Funnel(y=f["stage"], x=f["count"], textinfo="value+percent initial",
                                  marker=dict(color=SERIES[0]), connector=dict(fillcolor=GRID),
                                  customdata=f["description"],
                                  hovertemplate="%{y}: %{x}<br>%{customdata}<extra></extra>"))
        st.plotly_chart(style(fig.update_layout(title="Conversion funnel")), use_container_width=True)
    ts = pd.DataFrame(api("/timeseries", **P)["hourly"])
    with right:
        if not ts.empty:
            ts["hour"] = pd.to_datetime(ts["hour"])
            fig = px.line(ts, x="hour", y="entries", markers=True, title="Hourly entries",
                          color_discrete_sequence=[SERIES[0]])
            fig.update_traces(line_width=2, marker_size=8)
            st.plotly_chart(style(fig), use_container_width=True)
    if not ts.empty:
        fig = px.bar(ts, x="hour", y="revenue_inr", title="Hourly POS revenue (₹)",
                     color_discrete_sequence=[SERIES[2]], labels={"revenue_inr": "Revenue ₹"})
        st.plotly_chart(style(fig, 280), use_container_width=True)
    dq = m["data_quality"]
    with st.expander("Data quality"):
        st.json(dq)

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
        st.dataframe(z, use_container_width=True, hide_index=True)
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
        st.dataframe(qd[["zone_id", "quadrant", "attention_pct", "conversion_pct", "strategy"]],
                     use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(j["top_flows"]), use_container_width=True, hide_index=True)
    hz = api("/insights/hesitation", **P)
    st.subheader("Service touch-points (hesitation signals)")
    st.caption(f"Dwell timeout > {hz['thresholds']['dwell_timeout_s']:.0f}s or zone revisited "
               f"≥ {hz['thresholds']['loiter_revisits']}×. Levels: " +
               " · ".join(f"**{k}**: {v}" for k, v in hz["actions"].items()))
    if hz["zones"]:
        st.dataframe(pd.DataFrame(hz["zones"]), use_container_width=True, hide_index=True)

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
        st.dataframe(cv, use_container_width=True, hide_index=True)

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
        st.dataframe(sdf, use_container_width=True, hide_index=True)
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

st.caption(f"Rendered {datetime.now():%Y-%m-%d %H:%M:%S}")
