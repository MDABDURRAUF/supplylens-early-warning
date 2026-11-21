"""How good the warnings are, measured against the table of known disruptions.

The rules used everywhere in this module:

* An event is **detected** when the same entity is in alert on some day from
  ``early_days`` (14) before the event's onset to its **peak**. A warning
  that comes after the worst day is not counted as a detection. Days before
  the onset count only if the alert itself started inside that window, so a
  stale alert that happens to be running gets no credit for being early.
* Alerts are counted once per episode: an entity that alerts again within
  ``gap`` (7) days continues the same alert.
* An alert is **true** when it starts between ``early_days`` before an onset
  and the end of that event (so a late alert is still a true one). An alert
  that starts outside every such window is a **false alarm**, unless a port
  or region the entity depends on is in an event at the time (or up to
  ``gap`` days after it); those are counted as **related** and left out of
  both precision and the false-alarm count.
* ``days_from_onset`` is the first detecting day minus the onset; negative
  means the warning came before the disruption began.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .data import EVENT_TYPES, generate_sample_world
from .detect import LEVELS, METHODS, Detection, Settings, detect, episodes
from .model import OnsetModel, upstream_of

MODEL_LABEL = "Onset model"


def score_alerts(flags: pd.DataFrame, events: pd.DataFrame, entities: pd.DataFrame, start=None, end=None,
                 early_days: int = 14, gap: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match one method's alerts to the known events, by the rules at the top of this module.

    ``flags`` is a dates x entities table of booleans. Only events with an
    onset from ``start`` to ``end``, and alerts that start in that period, are
    scored. Returns two tables:

    * events, with ``detected``, ``first_alert`` and ``days_from_onset``;
    * alert episodes, with ``outcome``: "true", "false" or "related".
    """
    dates = flags.index
    start = dates[0] if start is None else pd.Timestamp(start)
    end = dates[-1] if end is None else pd.Timestamp(end)
    day_of = lambda ts: (pd.DatetimeIndex(ts) - dates[0]).days.to_numpy()  # noqa: E731  (days since the first date)
    ev = events[(events["onset"] >= start) & (events["onset"] <= end)].reset_index(drop=True).copy()

    # For every entity: its flagged days, and for each of them the first day of the alert it belongs to.
    flagged, began = {}, {}
    for entity in flags.columns:
        idx = np.flatnonzero(flags[entity].to_numpy(dtype=bool))
        new_alert = np.r_[True, np.diff(idx) > gap] if len(idx) else np.zeros(0, dtype=bool)
        flagged[entity], began[entity] = idx, (idx[new_alert][np.cumsum(new_alert) - 1] if len(idx) else idx)

    first = np.full(len(ev), np.nan)
    for i, (entity, onset, last) in enumerate(zip(ev["entity"], day_of(ev["onset"]), day_of(ev["peak"]))):
        idx = flagged.get(entity, np.zeros(0, dtype=int))
        inside = (idx >= onset - early_days) & (idx <= last)
        ok = inside & ((idx >= onset) | (began.get(entity, idx) >= onset - early_days))
        if ok.any():
            first[i] = idx[ok][0]
    ev["detected"] = ~np.isnan(first)
    ev["first_alert"] = pd.to_datetime([dates[0] + pd.Timedelta(days=int(d)) if not np.isnan(d) else pd.NaT for d in first])
    ev["days_from_onset"] = first - day_of(ev["onset"])
    ev = ev[[c for c in ev.columns if c not in ("first_alert", "detected", "days_from_onset")] + ["detected", "first_alert", "days_from_onset"]]

    eps = episodes(flags, gap)
    eps = eps[(eps["start"] >= start) & (eps["start"] <= end)].reset_index(drop=True)
    windows: dict = {}
    for entity, onset, last in zip(events["entity"], day_of(events["onset"]), day_of(events["end"])):
        windows.setdefault(entity, []).append((onset, last))
    up = upstream_of(entities)
    above = {entity: [u for u in (port, region) if u] for entity, port, region in zip(up.index, up["port"], up["region"])}
    outcome = []
    for entity, day in zip(eps["entity"], day_of(eps["start"])):
        if any(onset - early_days <= day <= last for onset, last in windows.get(entity, [])):
            outcome.append("true")
            continue
        related = any(onset <= day <= last + gap for u in above.get(entity, []) for onset, last in windows.get(u, []))
        outcome.append("related" if related else "false")
    eps["outcome"] = outcome
    return ev, eps


def method_flags(det: Detection, model: OnsetModel | None = None, entities: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Daily alert flags (dates x entities) for each method, keyed by its display name."""
    out = {label: det.flags[key] for key, label in METHODS.items()}
    if model is not None:
        out[MODEL_LABEL] = model.alert_flags(det, entities)
    return out


def _period(det: Detection, model: OnsetModel | None, start, end):
    if start is None:
        start = model.split_["test_from"] if model is not None else det.first_usable
    return pd.Timestamp(start), det.as_of if end is None else pd.Timestamp(end)


def _row(name: str, ev: pd.DataFrame, eps: pd.DataFrame, months: float) -> dict:
    true, false = int((eps["outcome"] == "true").sum()), int((eps["outcome"] == "false").sum())
    hit = ev[ev["detected"]]
    return {"method": name, "events": len(ev), "detected": len(hit), "detection_rate": len(hit) / len(ev) if len(ev) else np.nan,
            "median_days_from_onset": float(hit["days_from_onset"].median()) if len(hit) else np.nan,
            "false_alarms": false, "false_alarms_per_month": false / months if months else np.nan,
            "precision": true / (true + false) if true + false else np.nan}


def compare_methods(det: Detection, events: pd.DataFrame, entities: pd.DataFrame, model: OnsetModel | None = None,
                    start=None, end=None) -> pd.DataFrame:
    """The comparison table: static threshold, EWMA, CUSUM, risk index and (when given) the onset model.

    All methods are scored on the same period and events: by default the
    model's test period when a model is given, otherwise everything after the
    baseline's burn-in. Columns:

    ``detection_rate``              share of events detected by their peak
    ``median_days_from_onset``      median over detected events; negative = before onset
    ``median_days_gained``          median, over events that both this method and the static
                                    rule detected (``events_compared``), of how many days
                                    earlier this method's alert came
    ``false_alarms_per_month``      false alarms across all entities per 30.44 days
    ``precision``                   true alerts / (true + false alerts)
    """
    start, end = _period(det, model, start, end)
    months = ((end - start).days + 1) / 30.44
    scored = {name: score_alerts(f, events, entities, start, end, gap=det.settings.gap) for name, f in method_flags(det, model, entities).items()}
    static = scored[METHODS["static"]][0].set_index("event_id")["first_alert"]
    rows = []
    for name, (ev, eps) in scored.items():
        row = _row(name, ev, eps, months)
        gained = (static.reindex(ev["event_id"]).to_numpy() - ev["first_alert"].to_numpy()) / np.timedelta64(1, "D")
        gained = gained[~np.isnan(gained)]
        row["median_days_gained"] = float(np.median(gained)) if len(gained) else np.nan
        row["events_compared"] = len(gained)
        rows.append(row)
    cols = ["method", "events", "detected", "detection_rate", "median_days_from_onset", "median_days_gained", "events_compared",
            "false_alarms", "false_alarms_per_month", "precision"]
    return pd.DataFrame(rows)[cols]


def by_level(det: Detection, events: pd.DataFrame, entities: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Precision and recall of the risk index at each alert level.

    For each level, an entity is in alert while its index is at or above that
    level's threshold. ``recall`` is the share of events detected at that
    level or higher; ``precision`` is the share of alerts that were true.
    """
    start, end = _period(det, None, start, end)
    months = ((end - start).days + 1) / 30.44
    rows = []
    for name in reversed(LEVELS):
        threshold = getattr(det.settings, name.lower())
        ev, eps = score_alerts(det.held >= threshold, events, entities, start, end, gap=det.settings.gap)
        r = _row(name, ev, eps, months)
        rows.append({"level": f"{name} or higher" if name != "Critical" else name, "index_at_least": threshold, "alerts": len(eps),
                     "true_alerts": int((eps["outcome"] == "true").sum()), "false_alarms": r["false_alarms"],
                     "related_alerts": int((eps["outcome"] == "related").sum()), "precision": r["precision"],
                     "recall": r["detection_rate"], "median_days_from_onset": r["median_days_from_onset"],
                     "false_alarms_per_month": r["false_alarms_per_month"]})
    return pd.DataFrame(rows)


def by_event_type(det: Detection, events: pd.DataFrame, entities: pd.DataFrame, model: OnsetModel | None = None,
                  start=None, end=None) -> pd.DataFrame:
    """Detection rate and median days from onset for each kind of disruption and each method (long format)."""
    start, end = _period(det, model, start, end)
    rows = []
    for name, f in method_flags(det, model, entities).items():
        ev, _ = score_alerts(f, events, entities, start, end, gap=det.settings.gap)
        for kind, g in ev.groupby("type"):
            hit = g[g["detected"]]
            rows.append({"event_type": EVENT_TYPES.get(kind, kind), "method": name, "events": len(g), "detected": len(hit),
                         "detection_rate": len(hit) / len(g), "median_days_from_onset": float(hit["days_from_onset"].median()) if len(hit) else np.nan})
    return pd.DataFrame(rows, columns=["event_type", "method", "events", "detected", "detection_rate", "median_days_from_onset"])


def event_timeline(det: Detection, events: pd.DataFrame, entities: pd.DataFrame, model: OnsetModel | None = None,
                   start=None, end=None) -> pd.DataFrame:
    """One row per event with each method's days from onset to first alert (blank = missed)."""
    start, end = _period(det, model, start, end)
    out = None
    for name, f in method_flags(det, model, entities).items():
        ev, _ = score_alerts(f, events, entities, start, end, gap=det.settings.gap)
        if out is None:
            out = ev[["event_id", "entity", "type", "onset", "peak", "end", "severity"]].copy()
        out[name] = ev["days_from_onset"].to_numpy()
    return out


# --------------------------------------------------------------- onset model
def _holdout(model: OnsetModel) -> pd.DataFrame:
    if model.holdout_ is None:
        raise RuntimeError("Call fit() first")
    h = model.holdout_
    return h[h["known"] & ~h["in_event"]]


def model_summary(model: OnsetModel) -> dict:
    """Accuracy of the onset model on the test period, for entity-days not already in a disruption.

    ``roc_auc``        chance that a day before an onset is scored above an ordinary day (0.5 = guessing)
    ``index_roc_auc``  the same for the risk index used as the score, for comparison
    ``brier_score``    mean squared error of the probability; ``brier_baseline`` is the
                       error of always quoting the training base rate
    """
    h = _holdout(model)
    y = h["label"].to_numpy()
    both = len(np.unique(y)) == 2
    out = {"entity_days": len(h), "positives": int(y.sum()), "base_rate": float(y.mean()), "onsets": model.split_["test_onsets"],
           "roc_auc": roc_auc_score(y, h["p"]) if both else np.nan,
           "average_precision": average_precision_score(y, h["p"]) if both else np.nan,
           "index_roc_auc": roc_auc_score(y, h["risk_index"]) if both else np.nan,
           "index_average_precision": average_precision_score(y, h["risk_index"]) if both else np.nan,
           "brier_score": float(brier_score_loss(y, h["p"])),
           "brier_baseline": float(brier_score_loss(y, np.full(len(h), model.base_rate_)))}
    out["skill"] = 1 - out["brier_score"] / out["brier_baseline"]
    return out


def calibration_table(model: OnsetModel, edges=(0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)) -> pd.DataFrame:
    """Predicted against observed onset rate on the test period, in bands of predicted probability."""
    h = _holdout(model)
    band = pd.cut(h["p"], list(edges), include_lowest=True)
    g = h.groupby(band, observed=True)
    out = pd.DataFrame({"band": [f"{max(i.left, 0):.0%} to {i.right:.0%}" for i in g.size().index], "entity_days": g.size().to_numpy(),
                        "predicted": g["p"].mean().to_numpy(), "observed": g["label"].mean().to_numpy()})
    out["gap"] = out["observed"] - out["predicted"]
    return out


def threshold_table(det: Detection, events: pd.DataFrame, entities: pd.DataFrame, model: OnsetModel,
                    thresholds=(0.05, 0.10, 0.15, 0.25, 0.35, 0.50)) -> pd.DataFrame:
    """What the onset model's own alert would do at other probability thresholds, on the test period."""
    start, end = _period(det, model, None, None)
    months = ((end - start).days + 1) / 30.44
    p = model.predict(det, entities)
    rows = []
    for t in thresholds:
        ev, eps = score_alerts(p >= t, events, entities, start, end, gap=det.settings.gap)
        r = _row(f"{t:.0%}", ev, eps, months)
        early = ev["days_from_onset"] < 0
        rows.append({"threshold": t, "detection_rate": r["detection_rate"], "warned_before_onset": float(early.mean()) if len(ev) else np.nan,
                     "median_days_from_onset": r["median_days_from_onset"], "false_alarms_per_month": r["false_alarms_per_month"],
                     "precision": r["precision"]})
    return pd.DataFrame(rows)


def across_seeds(seeds=range(1, 11), settings: Settings | None = None) -> pd.DataFrame:
    """The comparison table averaged over several simulated histories, to show how much one history can mislead.

    For each seed a new sample world is generated, the detectors are run, the
    onset model is fitted, and every method is scored on that history's test
    period. Each column is the plain average of that figure across the
    histories; ``events`` is the average number of disruptions scored.
    """
    tables = []
    for seed in seeds:
        world = generate_sample_world(seed=int(seed))
        det = detect(world.signals, settings=settings)
        model = OnsetModel().fit(world, det)
        tables.append(compare_methods(det, world.events, world.entities, model))
    cols = ["events", "detection_rate", "median_days_from_onset", "median_days_gained", "false_alarms_per_month", "precision"]
    out = pd.concat(tables).groupby("method", sort=False)[cols].mean().reset_index()
    out.insert(1, "histories", len(tables))
    return out
