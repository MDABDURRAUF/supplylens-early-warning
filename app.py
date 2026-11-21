"""SupplyLens Early Warning dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Let the app run straight from a checkout (for example on Streamlit Community
# Cloud) without the package being installed first.
_SRC = _Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import early_warning as ew
from early_warning import evaluate as ev
from early_warning.report import days as fmt_days
from early_warning.report import money, pct

st.set_page_config(page_title="Early Warning", layout="wide")

RED, DARK, AMBER, GREEN, BLUE, GREY = "#d1495b", "#7a1f2b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
LEVEL_COLORS = {"Critical": DARK, "Warning": RED, "Watch": AMBER, "No alert": GREY}
METHOD_COLORS = {ew.METHODS["static"]: GREY, ew.METHODS["ewma"]: "#6fa3cf", ew.METHODS["cusum"]: BLUE, ew.METHODS["index"]: GREEN, ev.MODEL_LABEL: AMBER}
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")
DAY = st.column_config.DateColumn(format="DD MMM YYYY")
ONE = st.column_config.NumberColumn(format="%.1f")
INDEX = st.column_config.ProgressColumn("Risk index", format="%.0f", min_value=0, max_value=100)
VIEWS = ["Today's alerts", "Entity explorer", "Detection performance", "Exposure", "Data", "How it works"]


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


@st.cache_resource(show_spinner="Generating the simulated supply base...")
def sample_world() -> ew.World:
    return ew.generate_sample_world()


@st.cache_resource(max_entries=3, show_spinner="Building baselines and running the detectors...")
def run_detection(key: tuple, _world: ew.World) -> ew.Detection:
    return ew.detect(_world.signals)


@st.cache_resource(max_entries=3, show_spinner="Fitting the onset model on past disruptions...")
def fit_model(key: tuple, _world: ew.World, _det: ew.Detection):
    """The onset model, or None with the reason it could not be fitted."""
    if _world.events.empty:
        return None, "There is no events table, so there is nothing for the onset model to learn from."
    try:
        return ew.OnsetModel().fit(_world, _det), ""
    except ValueError as exc:
        return None, str(exc)


@st.cache_data(max_entries=6, show_spinner="Scoring every method against the known disruptions...")
def evaluation(key: tuple, _world: ew.World, _det: ew.Detection, _model, with_model: bool) -> dict:
    m = _model if with_model else None
    out = {"compare": ev.compare_methods(_det, _world.events, _world.entities, m),
           "types": ev.by_event_type(_det, _world.events, _world.entities, m),
           "timeline": ev.event_timeline(_det, _world.events, _world.entities, m)}
    start = m.split_["test_from"] if m is not None else _det.first_usable
    out["levels"] = ev.by_level(_det, _world.events, _world.entities, start=start)
    out["start"] = start
    return out


@st.cache_data(max_entries=3, show_spinner=False)
def model_tables(key: tuple, _world: ew.World, _det: ew.Detection, _model) -> dict:
    return {"summary": ev.model_summary(_model), "calibration": ev.calibration_table(_model),
            "thresholds": ev.threshold_table(_det, _world.events, _world.entities, _model)}


@st.cache_data(max_entries=3, show_spinner=False)
def exposure(key: tuple, _world: ew.World) -> pd.DataFrame:
    return ew.exposure_table(_world.entities, _world.skus)


@st.cache_data(max_entries=3, show_spinner=False)
def signals_csv(key: tuple, _world: ew.World) -> str:
    return _world.signals.assign(date=_world.signals["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False)


@st.cache_data(max_entries=3, show_spinner="Writing the evaluation report...")
def evaluation_report(key: tuple, _world: ew.World, _det: ew.Detection, _model) -> str:
    return ew.build_evaluation_report(_world, _det, _model)


def signal_chart(det: ew.Detection, entity: str, signal: str, since, until, events: pd.DataFrame | None = None, height: int = 340):
    """One signal with its normal band (expected plus or minus two scatters) and the days the entity was in alert."""
    key, window = (entity, signal), slice(since, until)
    actual, expected, scale = det.values.loc[window, key], det.expected.loc[window, key], det.scale.loc[window, key]
    fig = go.Figure()
    fig.add_scatter(x=expected.index, y=expected + 2 * scale, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=expected.index, y=expected - 2 * scale, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(58,110,165,0.15)", name="Normal range", hoverinfo="skip")
    fig.add_scatter(x=expected.index, y=expected, mode="lines", line=dict(color=BLUE, width=1, dash="dot"), name="Expected")
    fig.add_scatter(x=actual.index, y=actual, mode="lines", line=dict(color="#1c2430", width=1.3), name="Reported")
    level = det.level.loc[window, entity]
    for name in reversed(ew.LEVELS):
        on = level.index[(level == name).to_numpy()]
        if len(on):
            fig.add_scatter(x=on, y=actual.reindex(on), mode="markers", marker=dict(color=LEVEL_COLORS[name], size=7), name=name)
    if events is not None:
        for e in events.itertuples():
            if e.end >= actual.index[0] and e.onset <= actual.index[-1]:
                fig.add_vrect(x0=f"{e.onset:%Y-%m-%d}", x1=f"{e.end:%Y-%m-%d}", fillcolor=RED if e.entity == entity else AMBER, opacity=0.10, line_width=0)
    fig.update_layout(yaxis_title=f"{ew.SIGNALS[signal][0]} ({ew.SIGNALS[signal][1]})", legend=dict(orientation="h", y=-0.2))
    return show(fig, height)


def evidence_chart(why: pd.DataFrame, height: int = 300):
    fig = go.Figure(go.Bar(x=why["strength"], y=why["label"], orientation="h",
                           marker_color=[RED if v >= 0.5 else AMBER if v > 0 else GREY for v in why["strength"]]))
    fig.update_layout(xaxis_title="Evidence strength (0.5 = at its own alarm limit)", xaxis_range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return show(fig, height)


WHY_COLUMNS = {"label": "Signal", "value_7d": st.column_config.NumberColumn("Last 7 days", format="%.3f"),
               "expected_7d": st.column_config.NumberColumn("Normal for those days", format="%.3f"),
               "z": st.column_config.NumberColumn("Today's deviation score", format="%+.1f"),
               "ewma": st.column_config.NumberColumn("EWMA score", format="%+.1f"), "cusum": st.column_config.NumberColumn("CUSUM", format="%.1f"),
               "strength": st.column_config.NumberColumn("Evidence strength", format="%.2f")}

# ------------------------------------------------------------------- sidebar
st.sidebar.title("Early Warning")
st.sidebar.caption("Supply chain early-warning system")
page = st.sidebar.radio("View", VIEWS)

if "world" not in st.session_state:
    st.session_state.update(world=sample_world(), source="sample")
world: ew.World = st.session_state["world"]
key = (len(world.signals), str(world.first_date), str(world.last_date), float(world.signals["value"].sum()), len(world.events), len(world.skus),
       len(world.entities))
if st.session_state["source"] == "sample":
    st.sidebar.info("Using a simulated supply base with known disruptions. Load your own data on the **Data** view.")
try:
    det = run_detection(key, world)
except (ValueError, ew.SchemaError) as exc:
    st.error(f"The detectors could not run on this data: {exc}")
    st.session_state.update(world=sample_world(), source="sample")
    st.stop()
model, model_note = fit_model(key, world, det)
names = world.entities.set_index("entity")["name"]
kinds = world.entities.set_index("entity")["type"]
label_of = lambda e: e if names[e] == e else f"{e}: {names[e]}"  # noqa: E731
st.sidebar.caption(f"{len(world.entities)} entities, {det.values.shape[1]} signals, {world.first_date:%d %b %Y} to {world.last_date:%d %b %Y}.")
if len(det.dates) <= det.settings.burn_in:
    st.sidebar.warning(f"Alerts need about {det.settings.burn_in} days of history; this data has {len(det.dates)}.")

# ============================================================ TODAY'S ALERTS
if page == "Today's alerts":
    picked = st.sidebar.date_input("As of", value=det.as_of.date(), min_value=det.first_usable.date(), max_value=det.as_of.date(),
                                   help="Replay any past day. Only data up to that day is used.")
    as_of = pd.Timestamp(picked)
    st.title(f"Alerts as of {as_of:%A %d %B %Y}")
    alerts = ew.build_alerts(world, det, as_of, model)
    st.info(ew.digest_headline(alerts, world))
    counts = alerts["level"].value_counts()
    exposed = ew.exposure_at_risk(alerts, world.entities, world.skus)
    k = st.columns(5)
    k[0].metric("Critical", int(counts.get("Critical", 0)))
    k[1].metric("Warning", int(counts.get("Warning", 0)))
    k[2].metric("Watch", int(counts.get("Watch", 0)))
    k[3].metric("New today", int((alerts["days_active"] <= 1).sum()))
    k[4].metric("Yearly spend exposed", money(exposed["annual_spend"]), help="Spend on the SKUs behind today's alerts, each counted once. Not a loss forecast.")

    if len(alerts):
        left, right = st.columns(2)
        with left:
            by = alerts.groupby(["type", "level"]).size().reset_index(name="alerts")
            fig = px.bar(by, x="alerts", y="type", color="level", orientation="h", color_discrete_map=LEVEL_COLORS,
                         category_orders={"level": ew.LEVELS}, labels={"alerts": "Entities in alert", "type": ""})
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            show(fig, 320)
        with right:
            fig = px.scatter(alerts, x="annual_spend_exposed", y="risk_index", color="level", size=alerts["days_active"].clip(upper=60) + 5,
                             hover_name="name", hover_data={"likely_cause": True, "days_active": True, "evidence": True},
                             color_discrete_map=LEVEL_COLORS, category_orders={"level": ew.LEVELS},
                             labels={"annual_spend_exposed": "Yearly spend exposed ($)", "risk_index": "Risk index"})
            fig.update_layout(legend=dict(orientation="h", y=-0.25), yaxis_range=[0, 105])
            show(fig, 320)
        st.caption("Right: each bubble is an entity in alert; larger bubbles have been in alert longer. Top right is serious and expensive.")

        st.subheader("Alerts, most serious first")
        st.caption("Ranked by level, then by yearly spend exposed. Click a row to see why it is in alert and what is behind it.")
        cols = ["entity", "name", "level", "risk_index", "change_7d", "since", "likely_cause", "evidence", "onset_probability",
                "skus_exposed", "single_source_skus", "annual_spend_exposed"]
        event = st.dataframe(alerts[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="alert_table", column_config={
            "risk_index": INDEX, "change_7d": st.column_config.NumberColumn("7-day change", format="%+.0f"), "since": DAY,
            "onset_probability": st.column_config.NumberColumn("Onset model", format="percent", help="Chance a disruption begins within 14 days; blank inside the model's training period"),
            "annual_spend_exposed": USD})
        rows = event.selection.rows
        chosen = alerts.iloc[rows[0] if rows and rows[0] < len(alerts) else 0]
        st.subheader(f"{chosen['name']}: {chosen['level']}, index {chosen['risk_index']:.0f}, in alert since {chosen['since']:%d %b %Y}")
        st.markdown(f"**Likely cause:** {chosen['likely_cause'].lower()}. **Suggested first step:** {chosen['suggested_action']}"
                    + (f" **Also in alert upstream:** {chosen['upstream']}." if chosen["upstream"] else ""))
        left, right = st.columns(2)
        with left:
            why = ew.explain(det, chosen["entity"], as_of)
            evidence_chart(why)
            st.caption("How strongly each signal says something has changed, using only data up to this day.")
        with right:
            signal_chart(det, chosen["entity"], chosen["main_signal"], as_of - pd.Timedelta(days=150), as_of, height=300)
            st.caption("The strongest signal against its normal range. Dots mark days in alert.")
        with st.expander(f"SKUs behind {chosen['entity']} ({int(chosen['skus_exposed'])})"):
            st.dataframe(ew.exposed_skus(chosen["entity"], world.entities, world.skus), hide_index=True, column_config={"annual_spend_usd": USD})

    st.subheader("Not in alert yet, but rated likely to start")
    if model is None:
        st.caption(f"The onset model is not available. {model_note}")
    elif as_of < model.split_["test_from"]:
        st.caption(f"The onset model was trained and calibrated on days before {model.split_['test_from']:%d %b %Y}, so replaying it on this "
                   "day would not be a fair test. Pick a later day to see its estimates.")
    else:
        watch = ew.model_watchlist(world, det, model, as_of)
        if watch.empty:
            st.caption(f"No entity below Watch is at or above the model's alert probability of {model.alert_threshold_:.0%}.")
        else:
            st.dataframe(watch, hide_index=True, column_config={"onset_probability": PCT, "risk_index": INDEX, "annual_spend_exposed": USD})
            st.caption(f"The onset model's estimate that a disruption begins within {model.horizon} days, for entities at or above "
                       f"{model.alert_threshold_:.0%}. It is right less often than an alert: see Detection performance.")

    st.subheader("Take it with you")
    d = st.columns(2)
    watchlist = ew.model_watchlist(world, det, model, as_of) if model is not None else None
    d[0].download_button("Alert digest (HTML)", ew.build_digest(alerts, world, as_of, watchlist), f"alert_digest_{as_of:%Y-%m-%d}.html",
                         "text/html", width="stretch", type="primary")
    d[1].download_button("Alerts (CSV)", alerts.to_csv(index=False), f"alerts_{as_of:%Y-%m-%d}.csv", "text/csv", width="stretch")

# =========================================================== ENTITY EXPLORER
elif page == "Entity explorer":
    st.title("Entity explorer: one signal against its own normal")
    current = ew.current_alerts(det)
    options = sorted(det.index.columns, key=lambda e: (ew.data.ENTITY_TYPES.index(kinds[e]), e))
    default = current["entity"].iloc[0] if len(current) else options[0]
    c = st.columns([2, 2, 1])
    entity = c[0].selectbox("Entity", options, index=options.index(default), format_func=label_of)
    signal = c[1].selectbox("Signal", det.signals_of(entity), format_func=lambda s: ew.SIGNALS[s][0])
    span = c[2].selectbox("Show", ["Last 6 months", "Last 12 months", "All history"], index=1)
    since = det.dates[0] if span == "All history" else det.as_of - pd.Timedelta(days=183 if "6" in span else 365)
    up = ew.upstream_of(world.entities).loc[entity]
    own_events = world.events[world.events["entity"] == entity]
    near_events = world.events[world.events["entity"].isin([entity] + [u for u in (up["port"], up["region"]) if u])]
    history = ew.episodes(det.flags["index"][[entity]], det.settings.gap)
    history["peak_index"] = [det.index.loc[a.start:a.last, entity].max() for a in history.itertuples()]
    history["peak_day"] = [det.index.loc[a.start:a.last, entity].idxmax() for a in history.itertuples()]
    today_level = det.level[entity].iloc[-1] or "no alert"
    text = (f"**{names[entity]}** has a risk index of {det.index[entity].iloc[-1]:.0f} today ({today_level}). Over the history it "
            f"raised {len(history)} alert{'s' if len(history) != 1 else ''}")
    if len(world.events):
        text += f", and the events table lists {len(own_events)} known disruption{'s' if len(own_events) != 1 else ''} for it"
    st.info(text + ". The shaded band is its normal range for each day, learned from its own past and adjusted for the day of the week.")
    k = st.columns(4)
    k[0].metric("Risk index today", f"{det.index[entity].iloc[-1]:.0f}")
    k[1].metric("Level today", today_level)
    k[2].metric("Alerts in the history", len(history))
    k[3].metric("Days in alert", int(det.flags["index"][entity].sum()))

    marks = st.checkbox("Shade known disruptions", value=bool(len(near_events)), disabled=not len(near_events))
    signal_chart(det, entity, signal, since, det.as_of, near_events if marks else None, height=360)
    st.caption("Band: expected value plus or minus two typical scatters. Dots: days the entity was in alert. "
               + ("Red shading: a known disruption at this entity; amber: one at the port or region it depends on." if marks else ""))
    idx = det.index.loc[since:, entity]
    fig = go.Figure(go.Scatter(x=idx.index, y=idx, mode="lines", line=dict(color=BLUE), name="Risk index"))
    for name in ew.LEVELS:
        fig.add_hline(y=getattr(det.settings, name.lower()), line_dash="dash", line_color=LEVEL_COLORS[name], annotation_text=name, annotation_position="top left")
    fig.update_layout(yaxis_title="Risk index", yaxis_range=[0, 105], showlegend=False)
    show(fig, 260)

    st.subheader("Alerts raised for this entity")
    if history.empty:
        st.caption("None. The table below shows the evidence as of the last day.")
        when = det.as_of
    else:
        st.caption("Click an alert to see the evidence on its strongest day.")
        table = history.sort_values("start", ascending=False).reset_index(drop=True)
        event = st.dataframe(table[["start", "last", "days", "peak_index", "peak_day"]], hide_index=True, on_select="rerun", selection_mode="single-row",
                             key="entity_alerts", column_config={"start": DAY, "last": st.column_config.DateColumn("Last day in alert", format="DD MMM YYYY"),
                                                                 "days": "Days in alert", "peak_index": INDEX, "peak_day": st.column_config.DateColumn("Strongest day", format="DD MMM YYYY")})
        rows = event.selection.rows
        when = table.loc[rows[0] if rows and rows[0] < len(table) else 0, "peak_day"]
    why = ew.explain(det, entity, when)
    st.markdown(f"**Why, on {when:%d %b %Y}** (risk index {det.index.at[when, entity]:.0f}): {ew.evidence_text(why) or 'no signal is raised.'}")
    left, right = st.columns([2, 3])
    with left:
        evidence_chart(why, 280)
    with right:
        st.dataframe(why[list(WHY_COLUMNS)], hide_index=True, column_config=WHY_COLUMNS)
    st.download_button("This entity's daily index and signals (CSV)",
                       pd.concat([det.index[entity].rename("risk_index"), det.level[entity].rename("level"), det.values[entity]], axis=1).to_csv(),
                       f"{entity}_history.csv", "text/csv")

# ===================================================== DETECTION PERFORMANCE
elif page == "Detection performance":
    st.title("Detection performance against the known disruptions")
    if world.events.empty:
        st.info("This view scores alerts against a table of known disruptions. The current data has none: add an events file on the "
                "Data view (entity, type, onset, peak, end), or reset to the simulated data.")
        st.stop()
    choices = ["Whole history (detectors)"] + (["Onset model's test period (all methods)"] if model is not None else [])
    period = st.radio("Period", choices, horizontal=True)
    with_model = period != choices[0]
    res = evaluation(key, world, det, model, with_model)
    comp, by = res["compare"], res["compare"].set_index("method")
    idx, static = by.loc[ew.METHODS["index"]], by.loc[ew.METHODS["static"]]
    st.info(ew.evaluation_headline(comp, res["start"], det.as_of))
    k = st.columns(5)
    k[0].metric("Disruptions scored", int(idx["events"]))
    k[1].metric("Detected by their peak", pct(idx["detection_rate"]), f"{(idx['detection_rate'] - static['detection_rate']) * 100:+.0f} pts vs static rule")
    k[2].metric("Median days from onset", fmt_days(idx["median_days_from_onset"]),
                f"{idx['median_days_from_onset'] - static['median_days_from_onset']:+.1f} vs static rule", delta_color="inverse")
    k[3].metric("Days gained on the static rule", fmt_days(idx["median_days_gained"], True), help="Median over disruptions both caught")
    k[4].metric("False alarms a month", fmt_days(idx["false_alarms_per_month"]),
                f"{idx['false_alarms_per_month'] - static['false_alarms_per_month']:+.1f} vs static rule", delta_color="inverse")
    st.caption("The figures above are for the risk index at Watch or higher. A disruption counts as detected when the entity is in alert "
               "between 14 days before its onset and its peak. " + ("" if with_model else "Detector settings were chosen on this data."))

    st.subheader("Method comparison")
    c = st.columns(3)
    for col, (field, title, fmt) in zip(c, [("detection_rate", "Detected by the peak", ".0%"), ("median_days_from_onset", "Median days from onset to first alert", ".1f"),
                                            ("false_alarms_per_month", "False alarms a month", ".1f")]):
        with col:
            fig = px.bar(comp, x=field, y="method", orientation="h", color="method", color_discrete_map=METHOD_COLORS, text_auto=fmt,
                         labels={field: title, "method": ""})
            fig.update_layout(showlegend=False)
            fig.update_yaxes(autorange="reversed")
            if fmt.endswith("%"):
                fig.update_xaxes(tickformat=".0%")
            show(fig, 280)
    st.dataframe(comp, hide_index=True, column_config={"detection_rate": PCT, "precision": PCT, "median_days_from_onset": ONE,
                                                       "median_days_gained": st.column_config.NumberColumn(format="%+.1f"), "false_alarms_per_month": ONE})
    st.caption("Median days gained: on disruptions that both this method and the static rule caught, how many days earlier this method was. "
               "Precision: true alerts over true plus false alerts; alerts explained by a disruption upstream are left out.")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk index by alert level")
        lv = res["levels"]
        fig = go.Figure()
        fig.add_bar(x=lv["level"], y=lv["recall"], name="Recall (disruptions detected)", marker_color=BLUE)
        fig.add_bar(x=lv["level"], y=lv["precision"], name="Precision (alerts that were true)", marker_color=GREEN)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        show(fig, 300)
        st.dataframe(lv.drop(columns=["index_at_least"]), hide_index=True, column_config={"precision": PCT, "recall": PCT, "median_days_from_onset": ONE,
                                                                                         "false_alarms_per_month": ONE})
    with right:
        st.subheader("By kind of disruption")
        grid = res["types"].pivot(index="event_type", columns="method", values="detection_rate")[list(comp["method"])]
        counts = res["types"].groupby("event_type")["events"].first()
        grid.index = [f"{t} ({counts[t]})" for t in grid.index]
        show(px.imshow(grid, color_continuous_scale="Blues", zmin=0, zmax=1, text_auto=".0%", aspect="auto",
                       labels={"x": "", "y": "", "color": "Detected"}), 300)
        st.caption("Share detected by the peak; the number of disruptions of each kind is in brackets. Few events per cell: read with care.")

    st.subheader("How early, disruption by disruption")
    tl = res["timeline"].melt(id_vars=["event_id", "entity", "type", "onset"], value_vars=list(comp["method"]), var_name="method", value_name="days_from_onset").dropna()
    fig = px.strip(tl, x="days_from_onset", y="method", color="method", color_discrete_map=METHOD_COLORS, hover_data=["entity", "type", "onset"],
                   labels={"days_from_onset": "Days from onset to first alert (negative = before the onset)", "method": ""})
    fig.add_vline(x=0, line_dash="dash", line_color=GREY)
    fig.update_layout(showlegend=False)
    show(fig, 320)
    st.caption("Each dot is one detected disruption. Dots left of the dashed line are warnings that came before the disruption began.")

    if model is not None:
        st.subheader(f"Onset model: will a disruption begin within {model.horizon} days?")
        mt = model_tables(key, world, det, model)
        s, split = mt["summary"], model.split_
        st.caption(f"Trained on days before {split['train_until']:%d %b %Y} ({split['train_onsets']} onsets), calibrated up to "
                   f"{split['test_from']:%d %b %Y} ({split['calibration_onsets']} onsets), tested from then on ({split['test_onsets']} onsets, "
                   f"{s['entity_days']:,} entity-days not already in a disruption).")
        k = st.columns(4)
        k[0].metric("ROC AUC", f"{s['roc_auc']:.3f}", help="0.5 is guessing; 1.0 ranks every day before an onset above every ordinary day")
        k[1].metric("Risk index as the score", f"{s['index_roc_auc']:.3f}", help="ROC AUC when the risk index is used to answer the same question")
        k[2].metric("Brier score", f"{s['brier_score']:.4f}", f"{s['brier_score'] - s['brier_baseline']:+.4f} vs base rate", delta_color="inverse")
        k[3].metric("Alert probability", pct(model.alert_threshold_), help="The model raises its own alert at or above this probability")
        left, right = st.columns(2)
        with left:
            cal = mt["calibration"]
            top = float(max(cal["predicted"].max(), cal["observed"].max()) * 1.1)
            fig = px.scatter(cal, x="predicted", y="observed", size="entity_days", color_discrete_sequence=[BLUE], hover_data=["band", "entity_days"],
                             labels={"predicted": "Predicted onset rate", "observed": "Observed onset rate"})
            fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color=GREY), showlegend=False)
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            show(fig, 300)
            st.caption("Each point is a band of predicted probability, sized by the number of entity-days in it. On the dashed line, "
                       "predictions match what happened. Bands with few days are noisy.")
        with right:
            st.dataframe(mt["thresholds"], hide_index=True, column_config={"threshold": PCT, "detection_rate": PCT, "warned_before_onset": PCT,
                                                                          "median_days_from_onset": ONE, "false_alarms_per_month": ONE, "precision": PCT})
            st.caption("What the model's own alert would do at other probability thresholds, on the test period.")
    else:
        st.caption(f"Onset model not available. {model_note}")
    d = st.columns(2)
    d[0].download_button("Evaluation report (HTML)", evaluation_report(key, world, det, model), "evaluation_report.html", "text/html",
                         width="stretch", type="primary")
    d[1].download_button("Method comparison (CSV)", comp.to_csv(index=False), "method_comparison.csv", "text/csv", width="stretch")

# ================================================================== EXPOSURE
elif page == "Exposure":
    st.title("Exposure: what sits behind each entity")
    if world.skus.empty:
        st.info("Exposure needs a SKU file that links suppliers to SKUs and yearly spend. Add one on the Data view, or reset to the simulated data.")
        st.stop()
    table = exposure(key, world).copy()
    table["level_today"] = table["entity"].map(det.level.iloc[-1]).replace("", "No alert")
    alerts = ew.current_alerts(det)
    at_risk = ew.exposure_at_risk(alerts, world.entities, world.skus)
    total = float(world.skus["annual_spend_usd"].sum())
    sources = world.skus.groupby("sku")["entity"].nunique()
    sup = table[table["type"] == "supplier"]
    st.info(f"{money(total)} of yearly spend on {world.skus['sku'].nunique()} SKUs from {len(sup)} suppliers; {int((sources == 1).sum())} SKUs have a "
            f"single source. The largest supplier is {sup.iloc[0]['name']} with {pct(sup.iloc[0]['spend_share'])} of spend. Today's alerts sit in "
            f"front of {money(at_risk['annual_spend'])} ({pct(at_risk['spend_share'])}). Exposure is what is behind an entity, not a forecast of loss.")
    k = st.columns(5)
    k[0].metric("Yearly spend", money(total))
    k[1].metric("Suppliers", len(sup))
    k[2].metric("SKUs", world.skus["sku"].nunique())
    k[3].metric("Single-source SKUs", int((sources == 1).sum()))
    k[4].metric("Behind today's alerts", money(at_risk["annual_spend"]), pct(at_risk["spend_share"]) + " of spend", delta_color="off")

    left, right = st.columns(2)
    with left:
        types = st.multiselect("Entity types", ew.data.ENTITY_TYPES, default=[t for t in ("supplier", "port") if t in set(table["type"])])
        top = table[table["type"].isin(types or ew.data.ENTITY_TYPES)].head(20)
        fig = px.bar(top, x="annual_spend_exposed", y="entity", color="level_today", orientation="h", color_discrete_map=LEVEL_COLORS,
                     hover_name="name", category_orders={"level_today": ew.LEVELS + ["No alert"], "entity": top["entity"].tolist()},
                     labels={"annual_spend_exposed": "Yearly spend exposed ($)", "entity": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        show(fig, 520)
    with right:
        tree = sup.merge(world.entities[["entity", "region", "port"]], on="entity")
        group = st.radio("Group suppliers by", ["region", "port"], horizontal=True)
        tree[group] = tree[group].replace("", "not given")
        fig = px.treemap(tree, path=[group, "entity"], values="annual_spend_exposed", color="level_today", color_discrete_map={**LEVEL_COLORS, "(?)": "#e6eaf0"})
        show(fig, 520)
    st.caption("Left: the 20 largest exposures, coloured by today's alert level. A port, region or commodity is exposed to every supplier "
               "that depends on it, so those bars overlap with the supplier bars.")

    st.subheader("All entities")
    st.caption("Click a row to list the SKUs behind it.")
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="exposure_table",
                         column_config={"annual_spend_exposed": USD, "spend_share": PCT}, height=320)
    rows = event.selection.rows
    chosen = table.iloc[rows[0] if rows and rows[0] < len(table) else 0]
    behind = ew.exposed_skus(chosen["entity"], world.entities, world.skus)
    st.markdown(f"**{chosen['name']}**: {len(behind)} SKU rows, {money(behind['annual_spend_usd'].sum())} a year, "
                f"{int(behind.loc[behind['single_source'], 'sku'].nunique())} single-source SKUs.")
    st.dataframe(behind, hide_index=True, column_config={"annual_spend_usd": USD}, height=260)
    st.download_button("Exposure by entity (CSV)", table.to_csv(index=False), "exposure.csv", "text/csv")

# ====================================================================== DATA
elif page == "Data":
    st.title("Data")
    st.info(("You are looking at simulated data: a made-up supply base with disruptions planted in it. " if st.session_state["source"] == "sample"
             else "You are looking at your own data. ")
            + "To load data, upload a signals file with one row per date, entity and signal. Entities, SKUs and events files are "
            "optional: they add upstream links, exposure, and scoring against known disruptions with the onset model.")
    c = st.columns(4)
    up_signals = c[0].file_uploader("Signals CSV (required)", type="csv")
    up_entities = c[1].file_uploader("Entities CSV", type="csv")
    up_skus = c[2].file_uploader("SKUs CSV", type="csv")
    up_events = c[3].file_uploader("Events CSV", type="csv")
    c = st.columns(3)
    if up_signals is not None and c[0].button("Use these files", type="primary"):
        try:
            read = lambda f: None if f is None else pd.read_csv(BytesIO(f.getvalue()))  # noqa: E731
            loaded = ew.make_world(read(up_signals), read(up_entities), read(up_skus), read(up_events))
            if (loaded.last_date - loaded.first_date).days < 60:
                raise ew.SchemaError("at least 60 days of signals are needed before any baseline can be built")
        except (ew.SchemaError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use these files: {exc}")
        else:
            st.session_state.update(world=loaded, source="yours")
            st.rerun()
    if c[1].button("Reset to simulated data"):
        st.session_state.update(world=sample_world(), source="sample")
        st.rerun()
    c[2].download_button("Download current signals (CSV)", signals_csv(key, world), "signals.csv", "text/csv")

    st.subheader("Columns")
    tabs = st.tabs(["Signals", "Entities", "SKUs", "Events"])
    ref = lambda rows: pd.DataFrame(rows, columns=["column", "required", "meaning"])  # noqa: E731
    tabs[0].dataframe(ref([("date", "yes", "Day the value refers to"), ("entity", "yes", "Supplier, port, region or commodity identifier"),
                           ("signal", "yes", "One of the signal names below"), ("value", "yes", "The reading; blank means not reported that day")]), hide_index=True)
    tabs[0].dataframe(pd.DataFrame([(name, label, unit, "higher" if bad > 0 else "lower", kind) for name, (label, unit, bad, kind) in ew.SIGNALS.items()],
                                   columns=["signal", "meaning", "unit", "worse when", "reported by"]), hide_index=True)
    tabs[1].dataframe(ref([("entity", "yes", "Identifier, as used in the signals"), ("type", "no", "supplier, port, region or commodity; taken from the signals if blank"),
                           ("name", "no", "Display name"), ("state", "no", "Two-letter US state code; sets the Census region"),
                           ("region", "no", "Census region; for a region entity, the region it stands for"),
                           ("port", "no", "For a supplier: the port entity its goods come through"),
                           ("commodity", "no", "For a supplier: the commodity entity it depends on"), ("category", "no", "What the supplier makes")]), hide_index=True)
    tabs[2].dataframe(ref([("sku", "yes", "SKU identifier"), ("entity", "yes", "Supplier that provides it; repeat the SKU for a second source"),
                           ("annual_spend_usd", "yes", "Yearly spend on this SKU from this supplier"), ("description", "no", "Free text")]), hide_index=True)
    tabs[3].dataframe(ref([("entity", "yes", "Where the disruption happened"), ("type", "yes", "For example capacity, quality, port_congestion, weather, price_spike"),
                           ("onset", "yes", "Day it began"), ("peak", "no", "Worst day; the end is used if absent"), ("end", "yes", "Day it was over"),
                           ("severity", "no", "0 to 1"), ("event_id", "no", "Identifier")]), hide_index=True)

    st.subheader("Current data")
    k = st.columns(5)
    k[0].metric("Signal rows", f"{len(world.signals):,}")
    k[1].metric("Entities", len(world.entities))
    k[2].metric("Days", (world.last_date - world.first_date).days + 1)
    k[3].metric("SKU rows", len(world.skus))
    k[4].metric("Known disruptions", len(world.events))
    left, right = st.columns(2)
    with left:
        per = world.signals.groupby("signal").agg(series=("entity", "nunique"), rows=("value", "size")).reset_index()
        per["signal"] = per["signal"].map(lambda s: ew.SIGNALS[s][0])
        fig = px.bar(per, x="series", y="signal", orientation="h", color_discrete_sequence=[BLUE], labels={"series": "Entities reporting", "signal": ""})
        fig.update_yaxes(categoryorder="total ascending")
        show(fig, 300)
    with right:
        if len(world.events):
            monthly = world.events.assign(month=world.events["onset"].dt.to_period("M").dt.to_timestamp(),
                                          kind=world.events["type"].map(lambda t: ew.EVENT_TYPES.get(t, t))).groupby(["month", "kind"]).size().reset_index(name="onsets")
            fig = px.bar(monthly, x="month", y="onsets", color="kind", labels={"month": "", "onsets": "Known disruptions starting"})
            fig.update_layout(legend=dict(orientation="h", y=-0.3))
            show(fig, 300)
        else:
            st.caption("No events table loaded.")
    tabs = st.tabs(["Signals (last 300 rows)", "Entities", "SKUs", "Events"])
    tabs[0].dataframe(world.signals.tail(300), hide_index=True, height=300, column_config={"date": DAY})
    tabs[1].dataframe(world.entities, hide_index=True, height=300)
    tabs[2].dataframe(world.skus, hide_index=True, height=300, column_config={"annual_spend_usd": USD})
    tabs[3].dataframe(world.events, hide_index=True, height=300, column_config={"onset": DAY, "peak": DAY, "end": DAY})

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    s = det.settings
    st.info("The system learns what is normal for every signal from its own past, measures each day how far the signal has moved, "
            "and combines that evidence into one risk index per supplier, port, region and commodity. It is built to warn while the "
            "evidence is still building; how often that beats a fixed limit is measured on the Detection performance view.")
    st.markdown(
        f"""
**What it watches.** Every day, each supplier reports lead time, on-time rate, fill rate, reject rate and order-acknowledgment
delay; each port reports dwell time; each region a weather severity; each commodity a price index.

**What is normal.** For every signal of every entity, the expected value for a day is the median of its own previous {s.window}
days, adjusted for the day of the week. The typical scatter around that is measured the same robust way, so a glitch or the start
of a disruption barely moves the baseline. Today's reading becomes a *deviation score*: how many typical scatters it is from
normal, signed so that positive always means worse. US federal holidays and the day after are skipped.

**Four ways to raise an alert**

1. *Static threshold*: a fixed limit set from the first {s.static_days} days, broken {s.static_persist} days in a row. The simple rule the others are compared with.
2. *EWMA*: a weighted average of recent deviation scores. It reacts to a small shift that lasts.
3. *CUSUM*: a running total of deviations above an allowance. It reacts to slow drifts.
4. *Risk index* (0 to 100): for each signal, how close EWMA, CUSUM or a sudden jump is to its alarm limit; plus the same for all of
   the entity's signals added together, which picks up several signals each leaning slightly the wrong way. The pieces are
   combined so that one signal at its limit gives 50. The index must hold for {s.confirm_days} days: **Watch** from {s.watch:.0f},
   **Warning** from {s.warning:.0f}, **Critical** from {s.critical:.0f}.

**No peeking.** Every number for a day uses only that day and earlier days. Running the system on data cut off at a date gives
exactly the same alerts up to that date; a test rewrites all later data and checks that nothing earlier changes. That is what
makes the as-of replay on the first view honest.

**The onset model** answers a different question: *will a disruption begin here within 14 days?* It is a gradient-boosted
classifier trained on past disruptions, using recent levels and trends of each signal and the state of the port, region and
commodity an entity depends on. It is trained on the oldest part of the history, calibrated on the next, and tested on the most
recent. It needs an events table to learn from.

**Scoring.** A disruption counts as detected when the entity is in alert between 14 days before its onset and its peak. An alert
outside any disruption is a false alarm, unless a port or region the entity depends on is disrupted at the time.

**Exposure.** Suppliers are linked to SKUs and yearly spend; a port, region or commodity is linked to every supplier that depends
on it. Alerts are ranked by level and then by the spend behind them, and each comes with a rule-based first step.

**Limits.** The sample data is simulated, with disruptions and early signs that the simulation puts there; results on it show
that the method works on such data, not how it will do on yours. Detector settings were chosen on that data. With few
disruptions, every rate on the Detection performance view rests on a small number of events. A rolling baseline absorbs a change
that lasts for months and stops alerting on it, and it follows strong seasonal swings with a lag. The onset model can only learn
patterns that precede the disruptions in its history. Exposure is spend behind an entity, not expected loss, and suggested
actions are starting points.
"""
    )
