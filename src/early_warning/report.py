"""Printable outputs: the daily alert digest and the evaluation report."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from . import evaluate
from .data import World
from .detect import METHODS, Detection
from .impact import exposure_at_risk
from .model import OnsetModel


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def days(x, signed: bool = False) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.1f}" if signed else f"{x:.1f}"


def _date(x) -> str:
    return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%d %b %Y}"


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4;vertical-align:top}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff;white-space:nowrap}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""
LEVEL_COLORS = {"Critical": "#7a1f2b", "Warning": "#d1495b", "Watch": "#e9a23b"}


def _table(df: pd.DataFrame, formats: dict | None = None, empty: str = "Nothing to show.") -> str:
    if df is None or df.empty:
        return f'<p class="note">{escape(empty)}</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if c == "level" and v in LEVEL_COLORS:
                cells.append(f'<td><span class="tag" style="background:{LEVEL_COLORS[v]}">{escape(str(v))}</span></td>')
            elif c in formats:
                cells.append(f'<td class="num">{escape(formats[c](v))}</td>')
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">{escape(subtitle)}</p>{body}</main></body></html>')


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def digest_headline(alerts: pd.DataFrame, world: World) -> str:
    """One paragraph on today's alerts and what is behind them."""
    if alerts.empty:
        return "No entity is at Watch level or higher. Every signal is within its normal range."
    counts = alerts["level"].value_counts()
    crit, warn, watch = (int(counts.get(k, 0)) for k in ("Critical", "Warning", "Watch"))
    exp = exposure_at_risk(alerts, world.entities, world.skus)
    text = f"{_plural(len(alerts), 'entity').replace('entitys', 'entities')} in alert: {crit} Critical, {warn} Warning, {watch} Watch."
    if exp["annual_spend"]:
        text += (f" Behind them are {_plural(exp['suppliers'], 'supplier')} and {_plural(exp['skus'], 'SKU')} with {money(exp['annual_spend'])} "
                 f"of yearly spend ({pct(exp['spend_share'])} of the total), {exp['single_source_skus']} of them single-source.")
    new = alerts[alerts["days_active"] <= 1]
    if len(new):
        text += f" New today: {', '.join(new['entity'].head(6))}."
    top = alerts.iloc[0]
    text += f" Most serious: {top['name']} ({top['level']}, index {top['risk_index']:.0f}), likely {top['likely_cause'].lower()}."
    return text


def build_digest(alerts: pd.DataFrame, world: World, as_of, watchlist: pd.DataFrame | None = None, top: int = 40) -> str:
    """The daily alert digest as one HTML page: headline, counts, alerts ranked by level and exposure, suggested actions."""
    body = f'<p class="lead">{escape(digest_headline(alerts, world))}</p>'
    if not alerts.empty:
        counts = alerts["level"].value_counts()
        exp = exposure_at_risk(alerts, world.entities, world.skus)
        body += _kpis([("Critical", f"{int(counts.get('Critical', 0))}"), ("Warning", f"{int(counts.get('Warning', 0))}"),
                       ("Watch", f"{int(counts.get('Watch', 0))}"), ("suppliers exposed", f"{exp['suppliers']}"),
                       ("SKUs exposed", f"{exp['skus']}"), ("yearly spend exposed", money(exp["annual_spend"]))])
        cols = ["entity", "name", "level", "risk_index", "since", "likely_cause", "evidence", "skus_exposed", "single_source_skus",
                "annual_spend_exposed", "suggested_action"]
        body += ('<h2>Alerts, most serious first</h2><p class="note">Ranked by level, then by yearly spend exposed. Evidence compares '
                 "the last 7 days with what the entity's own history says is normal.</p>")
        body += _table(alerts.head(top)[cols], {"risk_index": lambda v: f"{v:.0f}", "since": _date, "annual_spend_exposed": money,
                                                "skus_exposed": lambda v: f"{v:.0f}", "single_source_skus": lambda v: f"{v:.0f}"})
    if watchlist is not None:
        body += ('<h2>Not in alert yet, but rated likely to start</h2><p class="note">Entities below Watch that the onset model rates '
                 "most likely to begin a disruption within its horizon. An estimate from past patterns, not a detection.</p>")
        body += _table(watchlist[["entity", "name", "onset_probability", "risk_index", "skus_exposed", "annual_spend_exposed"]],
                       {"onset_probability": pct, "risk_index": lambda v: f"{v:.0f}", "annual_spend_exposed": money,
                        "skus_exposed": lambda v: f"{v:.0f}"}, empty="None today.")
    body += ('<h2>How to read this</h2><p class="caveat">The risk index (0 to 100) combines how far each of an entity\'s signals has '
             "moved from its own normal range, using only data up to the date of this digest. Watch means look; Warning means "
             "prepare; Critical means act. An alert is evidence that something has changed, not proof of a disruption: some alerts "
             "are false alarms. Exposure is the yearly spend and the SKUs behind an entity, not a forecast of loss. Suggested "
             "actions are rule-based starting points.</p>")
    return _page(f"Early-warning digest: {_date(as_of)}", "SupplyLens Early Warning", body)


def evaluation_headline(comparison: pd.DataFrame, start, end) -> str:
    """One paragraph comparing the risk index with the static rule on the scored period."""
    by = comparison.set_index("method")
    idx, static = by.loc[METHODS["index"]], by.loc[METHODS["static"]]
    text = (f"Of {int(idx['events'])} known disruptions that began between {_date(start)} and {_date(end)}, the risk index raised an alert "
            f"for {pct(idx['detection_rate'])} and the static threshold rule for {pct(static['detection_rate'])}. The index's first alert came "
            f"a median of {days(idx['median_days_from_onset'])} days after onset, against {days(static['median_days_from_onset'])} for the "
            "static rule")
    if idx["events_compared"]:
        text += (f"; on the {int(idx['events_compared'])} disruptions both caught, the index was a median of "
                 f"{days(idx['median_days_gained'])} days earlier")
    text += (f". False alarms across all entities: {days(idx['false_alarms_per_month'])} a month for the index, "
             f"{days(static['false_alarms_per_month'])} for the static rule.")
    return text


def build_evaluation_report(world: World, det: Detection, model: OnsetModel | None = None) -> str:
    """How the detectors and the onset model did against the known disruptions, as one HTML page."""
    if world.events.empty:
        raise ValueError("An evaluation needs a table of known disruptions (events)")
    comp_fmt = {"detection_rate": pct, "median_days_from_onset": days, "median_days_gained": lambda v: days(v, True),
                "false_alarms_per_month": days, "precision": pct, "events": lambda v: f"{v:.0f}", "detected": lambda v: f"{v:.0f}",
                "events_compared": lambda v: f"{v:.0f}", "false_alarms": lambda v: f"{v:.0f}"}
    full = evaluate.compare_methods(det, world.events, world.entities)
    body = f'<p class="lead">{escape(evaluation_headline(full, det.first_usable, det.as_of))}</p>'
    idx = full.set_index("method").loc[METHODS["index"]]
    body += _kpis([("disruptions scored", f"{int(idx['events'])}"), ("detected by the index", pct(idx["detection_rate"])),
                   ("median days from onset", days(idx["median_days_from_onset"])),
                   ("days gained on the static rule", days(idx["median_days_gained"], True)),
                   ("false alarms a month", days(idx["false_alarms_per_month"]))])
    body += ('<h2>How alerts are scored</h2><p>An alert detects a disruption when the same entity is in alert between 14 days before '
             "the onset and the end of the disruption. An entity that alerts again within 7 days continues the same alert. An alert "
             "that starts outside every such window is a false alarm, unless a port or region the entity depends on is disrupted "
             "at the time; those related alerts are left out of precision and of the false-alarm count. Days from onset is the first "
             "alert minus the onset; days gained is how much earlier a method was than the static rule on disruptions both caught.</p>")
    body += (f"<h2>Detectors on the whole history</h2><p class=\"note\">{_date(det.first_usable)} to {_date(det.as_of)}: every day after "
             "the baseline's burn-in. Detector settings were chosen on this data.</p>") + _table(full, comp_fmt)
    body += "<h2>Risk index by alert level</h2>" + _table(evaluate.by_level(det, world.events, world.entities).drop(columns="index_at_least"), {
        "precision": pct, "recall": pct, "median_days_from_onset": days, "false_alarms_per_month": days})
    kinds = evaluate.by_event_type(det, world.events, world.entities)
    wide = kinds.pivot(index="event_type", columns="method", values="detection_rate")[list(METHODS.values())]
    wide.insert(0, "events", kinds.groupby("event_type")["events"].first())
    body += "<h2>Detection rate by kind of disruption</h2>" + _table(wide.reset_index(), {m: pct for m in METHODS.values()})
    if model is not None:
        s, split = evaluate.model_summary(model), model.split_
        body += (f"<h2>All methods on the onset model's test period</h2><p class=\"note\">{_date(split['test_from'])} to {_date(split['last_day'])}: "
                 f"days the onset model never saw. It was trained on days before {_date(split['train_until'])} "
                 f"({split['train_onsets']} onsets) and calibrated on the period in between ({split['calibration_onsets']} onsets). "
                 f"The model raises its own alert at a probability of {pct(model.alert_threshold_)} or more.</p>")
        body += _table(evaluate.compare_methods(det, world.events, world.entities, model), comp_fmt)
        body += (f"<h2>Onset model: will a disruption begin within {model.horizon} days?</h2><p>On {s['entity_days']:,} entity-days of the "
                 f"test period that were not already in a disruption ({s['positives']} of them within {model.horizon} days before one of "
                 f"{s['onsets']} onsets), the model's ROC AUC is {s['roc_auc']:.3f}; using the risk index as the score gives "
                 f"{s['index_roc_auc']:.3f}. Its Brier score is {s['brier_score']:.4f} against {s['brier_baseline']:.4f} for always "
                 "quoting the base rate.</p>")
        body += "<p class=\"note\">Predicted against observed onset rate, in bands of predicted probability.</p>"
        body += _table(evaluate.calibration_table(model), {"predicted": lambda v: pct(v, 1), "observed": lambda v: pct(v, 1),
                                                           "gap": lambda v: f"{v * 100:+.1f} pts", "entity_days": lambda v: f"{v:,.0f}"})
        body += "<p class=\"note\">The model's own alert at other probability thresholds.</p>"
        body += _table(evaluate.threshold_table(det, world.events, world.entities, model), {
            "threshold": pct, "detection_rate": pct, "warned_before_onset": pct, "median_days_from_onset": days,
            "false_alarms_per_month": days, "precision": pct})
    body += ('<h2>Limits</h2><p class="caveat">These results describe one history with a limited number of disruptions, so each rate '
             "rests on few events and would differ on another history. Detector settings were chosen on the data scored here. Onsets "
             "in a real events table are judgments made afterwards, and the scores depend on them. A rolling baseline treats a "
             "change that lasts for months as the new normal and stops alerting on it.</p>")
    return _page("Evaluation report: early warning against known disruptions", "SupplyLens Early Warning", body)


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
