import numpy as np
import pandas as pd
import pytest

import early_warning as ew
from early_warning import evaluate as ev
from early_warning.cli import main
from early_warning.data import event_shape, holiday_mask, validate_entities, validate_events, validate_skus
from early_warning.detect import run_lengths

INDEX = ew.METHODS["index"]
STATIC = ew.METHODS["static"]


@pytest.fixture(scope="module")
def world():
    return ew.generate_sample_world()


@pytest.fixture(scope="module")
def det(world):
    return ew.detect(world.signals)


@pytest.fixture(scope="module")
def model(world, det):
    return ew.OnsetModel().fit(world, det)


def days(n, start="2025-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def frame(values, columns=("x",), start="2025-01-01"):
    """A small daily table for the detector functions."""
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(values.reshape(len(values), -1), index=days(len(values), start), columns=list(columns))


def tiny_entities():
    """Two suppliers behind one port, one supplier elsewhere, a region and a commodity."""
    rows = [("S1", "supplier", "OH", "Midwest", "P1", "C1"), ("S2", "supplier", "MI", "Midwest", "P1", ""),
            ("S3", "supplier", "TX", "South", "", "C1"), ("P1", "port", "NJ", "Northeast", "", ""),
            ("R1", "region", "", "Midwest", "", ""), ("C1", "commodity", "", "", "", "")]
    out = pd.DataFrame(rows, columns=["entity", "type", "state", "region", "port", "commodity"])
    return out.assign(name=out["entity"], category="")[ew.data.ENTITY_COLUMNS]


# ------------------------------------------------------------------ US data
def test_us_reference_data():
    assert ew.region_of("oh") == "Midwest" and len({s for g in ew.CENSUS_REGIONS.values() for s in g}) == 51
    assert ew.federal_holidays(2026)[pd.Timestamp("2026-11-26").date()] == "Thanksgiving Day"
    mask = holiday_mask(pd.DatetimeIndex(["2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "2027-01-01", "2027-01-02"]), days_after=1)
    assert mask.tolist() == [False, True, True, False, True, True]  # the holiday and the day after it


# --------------------------------------------------------------------- data
def test_sample_world_is_reproducible_and_sensible(world):
    again = ew.generate_sample_world()
    pd.testing.assert_frame_equal(world.signals, again.signals)
    pd.testing.assert_frame_equal(world.events, again.events)
    assert world.entities["type"].value_counts().to_dict() == {"supplier": 40, "port": 5, "region": 4, "commodity": 3}
    assert (world.last_date - world.first_date).days == 729 and 150_000 < len(world.signals) < 730 * 212  # a few days are not reported
    assert set(world.signals["signal"]) == set(ew.SIGNALS)
    rates = world.signals[world.signals["signal"].isin(["on_time_rate", "fill_rate", "reject_rate", "weather_severity"])]
    assert rates["value"].between(0, 1).all()
    sup = world.entities[world.entities["type"] == "supplier"]
    assert set(sup["region"]) == set(ew.CENSUS_REGIONS) and (sup["state"].map(ew.region_of) == sup["region"]).all()
    assert sup["port"].isin(world.entities.loc[world.entities["type"] == "port", "entity"]).all()
    e = world.events
    assert len(e) == 77 and set(e["type"]) == set(ew.EVENT_TYPES) and e["event_id"].is_unique
    assert ((e["onset"] <= e["peak"]) & (e["peak"] <= e["end"])).all() and e["severity"].between(0.3, 1.0).all()
    assert e["onset"].min() >= world.first_date + pd.Timedelta(days=120)  # a clean stretch for every baseline to learn from
    kinds = world.entities.set_index("entity")["type"]
    assert set(e.loc[e["type"].isin(["capacity", "quality"]), "entity"].map(kinds)) == {"supplier"}
    assert set(world.skus["entity"]) == set(sup["entity"]) and (world.skus["annual_spend_usd"] > 0).all()
    assert world.skus.groupby("sku")["entity"].nunique().max() == 2  # some SKUs are dual-sourced
    # A capacity problem shows in the data: lead time at the peak is well above the month before the onset.
    hit = e[(e["type"] == "capacity") & (e["severity"] > 0.8)].iloc[0]
    lead = world.signals[(world.signals["entity"] == hit["entity"]) & (world.signals["signal"] == "lead_time_days")].set_index("date")["value"]
    before = lead[hit["onset"] - pd.Timedelta(days=45): hit["onset"] - pd.Timedelta(days=15)].mean()
    assert lead[hit["peak"] - pd.Timedelta(days=3): hit["peak"] + pd.Timedelta(days=3)].mean() > 1.08 * before


def test_event_shape_hand_worked():
    shape = event_shape(np.arange(8, 20), onset=10, peak=13, end=17)
    by_day = dict(zip(range(8, 20), shape))
    assert by_day[9] == 0 and by_day[18] == 0
    assert by_day[10] == pytest.approx((1 / 4) ** 1.5) and by_day[11] == pytest.approx((2 / 4) ** 1.5)  # slow start
    assert by_day[13] == 1.0
    assert by_day[15] == pytest.approx(1 - (2 / 5) ** 2) and by_day[17] == pytest.approx(1 - (4 / 5) ** 2)


def test_validation_reports_problems(world):
    good = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "entity": ["A", "A"], "signal": ["lead_time_days", "Lead_Time_Days "], "value": [10, ""]})
    clean = ew.validate_signals(good)
    assert clean["signal"].tolist() == ["lead_time_days"] * 2 and np.isnan(clean["value"].iloc[1])  # a blank value means "not reported"
    with pytest.raises(ew.SchemaError, match="Missing required column"):
        ew.validate_signals(good.drop(columns="value"))
    with pytest.raises(ew.SchemaError, match="unknown signal"):
        ew.validate_signals(good.assign(signal="mood"))
    with pytest.raises(ew.SchemaError, match="1 duplicate"):
        ew.validate_signals(good.assign(date="2026-01-05"))
    with pytest.raises(ew.SchemaError, match="unreadable 'date'"):
        ew.validate_signals(good.assign(date="soon"))
    with pytest.raises(ew.SchemaError, match="not a number"):
        ew.validate_signals(good.assign(value=["ten", "11"]))
    with pytest.raises(ew.SchemaError, match="more than one kind"):
        ew.validate_signals(good.assign(signal=["lead_time_days", "dwell_days"]))
    with pytest.raises(ew.SchemaError, match="no rows"):
        ew.make_world(good.iloc[:0])

    inferred = validate_entities(None, clean)
    assert inferred.loc[0, ["entity", "type", "name"]].tolist() == ["A", "supplier", "A"]
    filled = validate_entities(pd.DataFrame({"entity": ["A"], "state": ["oh"]}), clean)
    assert filled.loc[0, ["type", "state", "region"]].tolist() == ["supplier", "OH", "Midwest"]
    with pytest.raises(ew.SchemaError, match="not a US state"):
        validate_entities(pd.DataFrame({"entity": ["A"], "state": ["ZZ"]}), clean)
    with pytest.raises(ew.SchemaError, match="does not match the signals"):
        validate_entities(pd.DataFrame({"entity": ["A"], "type": ["port"]}), clean)
    with pytest.raises(ew.SchemaError, match="'port' is not a port"):
        validate_entities(pd.DataFrame({"entity": ["A"], "port": ["NOWHERE"]}), clean)
    with pytest.raises(ew.SchemaError, match="not a supplier"):
        validate_skus(pd.DataFrame({"sku": ["K"], "entity": ["PORT-LAX"], "annual_spend_usd": [1]}), world.entities)
    with pytest.raises(ew.SchemaError, match="zero or more"):
        validate_skus(pd.DataFrame({"sku": ["K"], "entity": ["SUP-01"], "annual_spend_usd": [-1]}), world.entities)
    event = pd.DataFrame({"entity": ["SUP-01"], "type": ["capacity"], "onset": ["2026-02-01"], "end": ["2026-03-01"]})
    checked = validate_events(event, world.entities)
    assert checked.loc[0, "peak"] == pd.Timestamp("2026-03-01") and checked.loc[0, "event_id"] == "EV-001"  # no peak given: the end is used
    with pytest.raises(ew.SchemaError, match="in that order"):
        validate_events(event.assign(end="2026-01-01"), world.entities)
    with pytest.raises(ew.SchemaError, match="not in the data"):
        validate_events(event.assign(entity="SUP-99"), world.entities)


# ----------------------------------------------------------------- baseline
def test_baseline_hand_worked():
    b = ew.robust_baseline(frame([10, 12, 11, 13, 20]), window=4, season_weeks=0, min_history=2)
    # Level: median of the previous days only (at least two of them). The 20 on the last day does not move its own baseline.
    assert b["level"]["x"].tolist()[2:] == [11.0, 11.0, 11.5] and b["level"]["x"].iloc[:2].isna().all()
    assert b["expected"]["x"].tolist()[2:] == [11.0, 11.0, 11.5]
    # Gaps so far: 0 (day 3) and 2 (day 4). Scatter on day 5 = 1.4826 x median(0, 2) = 1.4826; score = (20 - 11.5) / 1.4826.
    assert b["scale"]["x"].iloc[4] == pytest.approx(1.4826) and np.isnan(b["scale"]["x"].iloc[3])
    assert b["z"]["x"].iloc[4] == pytest.approx(8.5 / 1.4826)
    assert (b["persistence"]["x"] == 1.0).all()  # too little history to judge: treated as unrelated days


def test_baseline_learns_the_weekday_pattern():
    dates = days(140, "2025-01-06")  # starts on a Monday
    monday = np.asarray(dates.day_of_week) == 0
    x = pd.DataFrame({"x": np.where(monday, 110.0, 100.0)}, index=dates)
    with_weekday = ew.robust_baseline(x)
    assert with_weekday["level"]["x"].iloc[-1] == 100.0
    assert with_weekday["expected"]["x"][monday].iloc[-1] == 110.0 and with_weekday["expected"]["x"][~monday].iloc[-1] == 100.0
    assert with_weekday["z"]["x"].iloc[-7:].abs().max() == 0.0  # Mondays are no surprise
    without = ew.robust_baseline(x, season_weeks=0)
    assert without["expected"]["x"][monday].iloc[-1] == 100.0 and without["z"]["x"][monday].iloc[-1] > 10


def test_persistence_is_larger_for_signals_whose_bad_days_come_in_runs():
    rng = np.random.default_rng(0)
    unrelated = rng.normal(0, 1, 400)
    runs = np.zeros(400)
    for i in range(1, 400):
        runs[i] = 0.8 * runs[i - 1] + rng.normal(0, 0.6)
    p = ew.robust_baseline(frame(np.column_stack([unrelated, runs]), columns=("unrelated", "runs")))["persistence"].iloc[250:]
    assert p["unrelated"].median() < 1.3 and p["runs"].median() > 1.6


# ---------------------------------------------------------------- detectors
def test_ewma_and_cusum_hand_worked():
    z = frame([1, 2, np.nan, 3])
    # lam = 0.5: 0.5, then 0.5*2 + 0.5*0.5 = 1.25, unchanged on the missing day, then 0.5*3 + 0.5*1.25 = 2.125.
    expected = np.array([0.5, 1.25, 1.25, 2.125]) / np.sqrt(0.5 / 1.5)
    assert ew.ewma_score(z, lam=0.5)["x"].to_numpy() == pytest.approx(expected)
    # CUSUM with allowance 0.5: 0.5, 2.0, unchanged, 4.5; and never above the cap.
    assert ew.cusum(z, k=0.5)["x"].tolist() == [0.5, 2.0, 2.0, 4.5]
    assert ew.cusum(z, k=0.5, cap=3.0)["x"].tolist() == [0.5, 2.0, 2.0, 3.0]
    assert ew.cusum(frame([1, -2, 1]), k=0.5)["x"].tolist() == [0.5, 0.0, 0.5]  # a good day pulls the sum back to zero


def test_static_rule_and_run_lengths_hand_worked():
    wide = frame(np.column_stack([[10, 11, 9, 10, 12, 30], [10, 11, 9, 10, 12, 1]]), columns=("up", "down"))
    limits = ew.static_limits(wide, pd.Series({"up": 1.0, "down": -1.0}), days=5, k=3.0)
    # First five days: median 10, absolute deviations 0, 1, 1, 0, 2 with median 1, so one scatter is 1.4826.
    assert limits["up"] == pytest.approx(10 + 3 * 1.4826) and limits["down"] == pytest.approx(10 - 3 * 1.4826)
    runs = run_lengths(frame([0, 1, 1, 0, 1]).astype(bool))
    assert runs["x"].tolist() == [0, 1, 2, 0, 1]


def test_episodes_merge_alerts_that_return_within_the_gap():
    flags = pd.DataFrame(False, index=days(30), columns=["A", "B"])
    flags.iloc[[0, 1, 5, 20], 0] = True
    merged = ew.episodes(flags, gap=7)
    assert merged[["entity", "days"]].values.tolist() == [["A", 3], ["A", 1]]
    assert merged["start"].tolist() == [days(30)[0], days(30)[20]] and merged["last"].iloc[0] == days(30)[5]
    assert ew.episodes(flags, gap=3)["days"].tolist() == [2, 1, 1]


def test_risk_index_follows_its_formula(det):
    s = det.settings
    ratio = np.maximum(np.maximum(det.ewma / s.ewma_limit, det.cusum / s.cusum_limit), (det.z / s.jump_limit).fillna(0))
    pd.testing.assert_frame_equal(det.strength, (ratio - 0.5).clip(0, 1))
    day = det.as_of - pd.Timedelta(days=20)
    for entity in ("SUP-16", "PORT-SEA"):
        own = det.strength.loc[day, entity]
        pooled = det.pooled["strength"].at[day, entity]
        expected = 100 * (1 - (1 - (0.0 if np.isnan(pooled) else pooled)) * np.prod(1 - own.to_numpy()))
        assert det.index.at[day, entity] == pytest.approx(expected)
    assert det.pooled["z"]["PORT-SEA"].isna().all() and det.pooled["z"]["SUP-16"].notna().any()  # one signal: nothing to pool
    assert det.index.min().min() >= 0 and det.index.max().max() <= 100
    # A level needs the index at or above its threshold on two days in a row.
    two_days = det.index.rolling(2).min()
    assert ((det.level == "Critical") == (two_days >= s.critical)).all().all()
    assert ((det.level != "") == (two_days >= s.watch)).all().all() and (det.flags["index"] == (det.level != "")).all().all()


def test_slow_drift_is_caught_by_the_index_before_the_static_rule():
    rng = np.random.default_rng(3)
    n, start = 320, 220
    drift = np.clip(np.arange(n) - start, 0, 60) * 0.05  # one scatter after 20 days, two after 40, then level at three
    rows = []
    for signal, level, sign in (("lead_time_days", 20.0, 1), ("on_time_rate", 0.90, -1)):
        scale = 1.0 if sign == 1 else 0.02
        values = level + sign * scale * drift + rng.normal(0, scale, n)
        rows.append(pd.DataFrame({"date": days(n), "entity": "S", "signal": signal, "value": values}))
    det = ew.detect(pd.concat(rows))
    onset = days(n)[start]
    assert det.z.loc[onset + pd.Timedelta(days=30): onset + pd.Timedelta(days=60)].mean().min() > 1  # both score as worse, whichever way they moved
    quiet = slice(det.first_usable, onset - pd.Timedelta(days=1))
    assert not det.flags["index"].loc[quiet, "S"].any() and not det.flags["static"].loc[quiet, "S"].any()
    first = {k: det.flags[k]["S"][det.flags[k]["S"]].index[0] for k in ("index", "static")}
    assert onset < first["index"] < first["static"]
    why = ew.explain(det, "S", first["index"])
    assert set(why["signal"]) == {"lead_time_days", "on_time_rate", "all_signals"} and why["strength"].iloc[0] > 0.3


def test_holidays_and_single_glitches_do_not_raise_alerts(world, det):
    christmas = pd.Timestamp("2025-12-25")
    assert det.z.loc[christmas].isna().all() and det.z.loc[christmas + pd.Timedelta(days=1)].isna().all()
    assert det.z.loc[christmas + pd.Timedelta(days=2)].notna().mean() > 0.95
    shown = ew.detect(world.signals[world.signals["entity"] == "SUP-01"], settings=ew.Settings(skip_holidays=False))
    assert shown.z.loc[christmas].notna().all()
    # One wild day in an otherwise calm series: a score far beyond the cap, but no alert level.
    rng = np.random.default_rng(1)
    values = 50 + rng.normal(0, 1, 250)
    values[200] = 80
    one = ew.detect(pd.DataFrame({"date": days(250), "entity": "P", "signal": "dwell_days", "value": values}))
    assert one.z["P"]["dwell_days"].iloc[200] > 10 and one.z_adj["P"]["dwell_days"].iloc[200] == one.settings.z_clip
    assert (one.level["P"].iloc[195:215] == "").all()


def test_later_data_cannot_change_earlier_alerts(world, det):
    cut = pd.Timestamp("2025-09-30")
    truncated = ew.detect(world.signals, as_of=cut)
    tampered = world.signals.copy()
    later = tampered["date"] > cut
    tampered.loc[later, "value"] = tampered.loc[later, "value"] * 3 + 1  # rewrite everything after the cut
    rewritten = ew.detect(tampered)
    for other in (truncated, rewritten):
        for name in ("z", "ewma", "cusum", "static", "strength", "index", "level"):
            pd.testing.assert_frame_equal(getattr(det, name).loc[:cut], getattr(other, name).loc[:cut], check_freq=False)
        for name in ew.METHODS:
            pd.testing.assert_frame_equal(det.flags[name].loc[:cut], other.flags[name].loc[:cut], check_freq=False)
    assert truncated.as_of == cut and not det.index.loc[cut:].iloc[1:60].equals(rewritten.index.loc[cut:].iloc[1:60])
    for day in ("2025-03-14", "2025-09-30"):
        pd.testing.assert_frame_equal(ew.current_alerts(det, day), ew.current_alerts(truncated, day))
    then = world.as_of(cut)  # the world as it was known that day: no later signal, no event that had not begun
    assert then.last_date == cut and then.events["onset"].max() <= cut and 0 < len(then.events) < len(world.events)
    # The onset model's inputs obey the same rule.
    full, cut_off = ew.build_features(det, world.entities), ew.build_features(truncated, world.entities)
    pd.testing.assert_frame_equal(full[full.index.get_level_values("date") <= cut], cut_off)


def test_current_alerts_and_explanations(det):
    alerts = ew.current_alerts(det)
    assert len(alerts) >= 3 and set(alerts["level"]) <= set(ew.LEVELS) and alerts["risk_index"].is_monotonic_decreasing
    assert (alerts["since"] <= det.as_of).all() and (alerts["days_active"] >= 1).all() and alerts["evidence"].str.len().min() > 10
    quiet_day = ew.current_alerts(det, det.dates[60])  # before the baseline has enough history
    assert quiet_day.empty and list(quiet_day.columns) == list(alerts.columns)
    top = alerts.iloc[0]
    why = ew.explain(det, top["entity"])
    assert why["strength"].is_monotonic_decreasing and why["strength"].iloc[0] > 0.3
    assert set(why["signal"]) - {"all_signals"} == set(det.signals_of(top["entity"]))
    with pytest.raises(ValueError, match="not an entity"):
        ew.explain(det, "SUP-99")
    with pytest.raises(ValueError, match="outside the data"):
        ew.current_alerts(det, "2030-01-01")


# --------------------------------------------------------------- evaluation
def test_scoring_rules_hand_worked():
    dates = days(60)
    flags = pd.DataFrame(False, index=dates, columns=["S1", "S2", "S3"])
    flags.iloc[5, 0] = True              # S1: long before its event
    flags.iloc[25:28, 0] = True          # S1: five days after the onset, before the peak
    flags.iloc[50, 0] = True             # S1: after the event is over
    flags.iloc[4:9, 1] = True            # S2: already running when the 14-day window opens on day 6
    flags.iloc[35, 1] = True             # S2: during the event but after its peak
    flags.iloc[12, 2] = True             # S3: its port is disrupted
    flags.iloc[30, 2] = True             # S3: nothing going on
    events = pd.DataFrame({"event_id": ["a", "b", "p"], "entity": ["S1", "S2", "P1"], "type": ["capacity", "quality", "port_congestion"],
                           "onset": [dates[20], dates[20], dates[10]], "peak": [dates[30], dates[30], dates[12]],
                           "end": [dates[40], dates[40], dates[15]], "severity": [1.0, 1.0, 1.0]})
    entities = tiny_entities().replace({"entity": {"S3": "S3"}})
    entities.loc[entities["entity"] == "S3", "port"] = "P1"
    scored, alerts = ev.score_alerts(flags, events, entities, early_days=14, gap=7)
    by = scored.set_index("event_id")
    assert by.loc["a", "detected"] and by.loc["a", "first_alert"] == dates[25] and by.loc["a", "days_from_onset"] == 5
    assert not by.loc["b", "detected"] and np.isnan(by.loc["b", "days_from_onset"])  # stale alert, then one after the peak
    assert not by.loc["p", "detected"]
    outcome = {(r.entity, r.start): r.outcome for r in alerts.itertuples()}
    assert outcome == {("S1", dates[5]): "false", ("S1", dates[25]): "true", ("S1", dates[50]): "false",
                       ("S2", dates[4]): "false", ("S2", dates[35]): "true", ("S3", dates[12]): "related", ("S3", dates[30]): "false"}
    # An alert that starts inside the 14 days before the onset is an early detection.
    flags.iloc[5, 0], flags.iloc[12, 0] = False, True
    early, _ = ev.score_alerts(flags, events, entities)
    assert early.set_index("event_id").loc["a", "days_from_onset"] == -8


def test_comparison_against_ground_truth(world, det):
    table = ev.compare_methods(det, world.events, world.entities)
    assert table["method"].tolist() == list(ew.METHODS.values()) and (table["events"] == 77).all()
    by = table.set_index("method")
    index, static = by.loc[INDEX], by.loc[STATIC]
    assert index["detection_rate"] > static["detection_rate"] + 0.05
    assert index["median_days_from_onset"] < static["median_days_from_onset"]
    assert index["median_days_gained"] > 0 and static["median_days_gained"] == 0 and static["events_compared"] == static["detected"]
    assert (table["false_alarms_per_month"] < 6).all() and table["precision"].between(0.4, 0.9).all()
    scored, alerts = ev.score_alerts(det.flags["index"], world.events, world.entities, det.first_usable)
    months = ((det.as_of - det.first_usable).days + 1) / 30.44
    assert index["false_alarms_per_month"] == pytest.approx((alerts["outcome"] == "false").sum() / months)
    assert index["detection_rate"] == pytest.approx(scored["detected"].mean()) and (alerts["start"] >= det.first_usable).all()

    levels = ev.by_level(det, world.events, world.entities)
    assert levels["level"].tolist() == ["Watch or higher", "Warning or higher", "Critical"]
    assert levels["precision"].is_monotonic_increasing and levels["recall"].is_monotonic_decreasing
    assert levels["false_alarms"].is_monotonic_decreasing and levels.loc[0, "recall"] == pytest.approx(index["detection_rate"])
    kinds = ev.by_event_type(det, world.events, world.entities)
    slow = kinds[kinds["event_type"] == "Quality drift"].set_index("method")["detection_rate"]
    assert slow[INDEX] > slow[STATIC]  # the slow, single-signal drift is where a fixed limit does worst
    timeline = ev.event_timeline(det, world.events, world.entities)
    assert len(timeline) == 77 and (timeline[INDEX] < timeline[["EWMA", "CUSUM"]].min(axis=1)).sum() >= 10


# -------------------------------------------------------------------- model
def test_onset_labels_hand_worked():
    dates = days(40)
    events = pd.DataFrame({"entity": ["A"], "onset": [dates[20]], "end": [dates[25]]})
    y = ew.onset_labels(dates, ["A", "B"], events, horizon=14)
    a, b = y.xs("A", level="entity"), y.xs("B", level="entity")
    assert a.index[a["label"] == 1].tolist() == list(dates[6:20])  # the 14 days before the onset, not the onset itself
    assert a.index[a["in_event"]].tolist() == list(dates[20:26]) and b["label"].sum() == 0
    assert a.index[~a["known"]].tolist() == list(dates[26:])  # the last 14 days cannot be labelled yet


def test_model_is_split_by_time_and_beats_the_index_as_a_predictor(world, det, model):
    s, h = model.split_, model.holdout_
    assert s["first_day"] < s["train_until"] < s["test_from"] < s["last_day"]
    assert h["date"].min() == s["test_from"] and s["train_onsets"] + s["calibration_onsets"] + s["test_onsets"] == len(world.events)
    assert h["p"].between(0, 1).all() and 0 < model.alert_threshold_ < 1
    summary = ev.model_summary(model)
    assert summary["roc_auc"] > 0.58 and summary["roc_auc"] > summary["index_roc_auc"] + 0.03
    assert summary["brier_score"] < summary["brier_baseline"] and summary["positives"] > 100
    cal = ev.calibration_table(model)
    assert cal["entity_days"].sum() == summary["entity_days"]
    assert abs((cal["predicted"] * cal["entity_days"]).sum() / cal["entity_days"].sum() - summary["base_rate"]) < 0.03
    p = model.predict(det, world.entities, dates=[det.as_of])
    assert p.shape == (1, 52) and p.notna().all().all()
    table = ev.compare_methods(det, world.events, world.entities, model)
    assert table["method"].tolist()[-1] == ev.MODEL_LABEL and (table["events"] == s["test_onsets"]).all()
    thresholds = ev.threshold_table(det, world.events, world.entities, model)
    assert thresholds["false_alarms_per_month"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="needs a table of past disruptions"):
        ew.OnsetModel().fit(ew.World(world.signals, world.entities))
    with pytest.raises(ValueError, match="At least 200 days"):
        ew.OnsetModel().fit(world.as_of("2025-03-01"))


def test_across_seeds_averages_other_histories():
    table = ev.across_seeds(seeds=[3])
    assert table["method"].tolist() == list(ew.METHODS.values()) + [ev.MODEL_LABEL] and (table["histories"] == 1).all()
    assert table["detection_rate"].between(0, 1).all()


# ------------------------------------------------------------------- impact
def test_exposure_hand_worked():
    entities = tiny_entities()
    skus = pd.DataFrame({"sku": ["K1", "K2", "K2", "K3", "K4"], "entity": ["S1", "S1", "S2", "S2", "S3"],
                         "annual_spend_usd": [100.0, 50.0, 50.0, 200.0, 600.0], "description": ""})
    table = ew.exposure_table(entities, skus).set_index("entity")
    assert table.loc["S1", ["suppliers_exposed", "skus_exposed", "single_source_skus", "annual_spend_exposed"]].tolist() == [1, 2, 1, 150.0]
    assert table.loc["P1", ["suppliers_exposed", "skus_exposed", "single_source_skus", "annual_spend_exposed"]].tolist() == [2, 3, 2, 400.0]
    assert table.loc["R1", "annual_spend_exposed"] == 400.0 and table.loc["C1", "annual_spend_exposed"] == 750.0  # S1 and S3 use C1
    assert table.loc["S3", "spend_share"] == pytest.approx(0.6) and table.index[0] == "C1"
    assert ew.exposed_suppliers("P1", entities) == ["S1", "S2"] and ew.exposed_suppliers("S2", entities) == ["S2"]
    behind = ew.exposed_skus("P1", entities, skus)
    assert behind["sku"].tolist() == ["K3", "K1", "K2", "K2"] and behind["single_source"].tolist() == [True, True, False, False]
    alerts = pd.DataFrame({"entity": ["P1", "S1"]})  # S1 sits behind P1: counted once
    assert ew.exposure_at_risk(alerts, entities, skus) == {"suppliers": 2, "skus": 3, "single_source_skus": 2, "annual_spend": 400.0, "spend_share": 0.4}
    assert ew.likely_cause("supplier", "reject_rate") == "quality" and ew.likely_cause("supplier", "ack_delay_hours") == "capacity"
    assert ew.likely_cause("port", "dwell_days") == "port_congestion" and "second source" in ew.suggested_action("capacity", "Critical")
    with pytest.raises(ValueError, match="not an entity"):
        ew.exposed_suppliers("S9", entities)


def test_alerts_are_ranked_and_carry_exposure_and_actions(world, det, model):
    alerts = ew.build_alerts(world, det, model=model)
    order = alerts["level"].map({lv: i for i, lv in enumerate(ew.LEVELS)})
    assert order.is_monotonic_increasing
    for _, group in alerts.groupby("level"):
        assert group["annual_spend_exposed"].is_monotonic_decreasing  # within a level, the most spend first
    assert alerts["suggested_action"].str.len().min() > 20 and set(alerts["likely_cause"]) <= set(ew.EVENT_TYPES.values())
    assert np.allclose(alerts["risk_weighted_spend"], alerts["annual_spend_exposed"] * alerts["risk_index"] / 100)
    assert alerts["onset_probability"].between(0, 1).all()
    exposure = ew.exposure_table(world.entities, world.skus).set_index("entity")["annual_spend_exposed"]
    assert (alerts["annual_spend_exposed"].to_numpy() == exposure[alerts["entity"]].to_numpy()).all()
    # Inside the model's training period its probability is withheld, and the watch list is empty.
    early = model.split_["train_until"] - pd.Timedelta(days=30)
    assert ew.build_alerts(world, det, early, model)["onset_probability"].isna().all()
    assert ew.model_watchlist(world, det, model, early).empty
    watch = ew.model_watchlist(world, det, model)
    assert not set(watch["entity"]) & set(alerts["entity"]) and (watch["onset_probability"] >= model.alert_threshold_).all()


def test_signals_alone_are_enough(world):
    some = world.signals[world.signals["entity"].isin(["SUP-16", "SUP-18", "PORT-SEA"])][["date", "entity", "signal", "value"]]
    own = ew.make_world(some.assign(date=some["date"].dt.strftime("%Y-%m-%d")))
    assert own.entities.set_index("entity")["type"].to_dict() == {"PORT-SEA": "port", "SUP-16": "supplier", "SUP-18": "supplier"}
    assert own.skus.empty and own.events.empty
    det = ew.detect(own.signals)
    alerts = ew.build_alerts(own, det)
    assert len(alerts) >= 1 and (alerts["annual_spend_exposed"] == 0).all() and alerts["upstream"].eq("").all()
    assert "in alert" in ew.digest_headline(alerts, own)


# ------------------------------------------------------------------ reports
def test_reports(world, det, model):
    alerts = ew.build_alerts(world, det, model=model)
    headline = ew.digest_headline(alerts, world)
    assert f"{len(alerts)} entities in alert" in headline and "yearly spend" in headline
    html = ew.build_digest(alerts, world, det.as_of, ew.model_watchlist(world, det, model))
    assert html.startswith("<!doctype html>") and "Early-warning digest: 30 Jun 2026" in html
    assert all(name in html for name in alerts["entity"]) and "Critical" in html and "How to read this" in html
    assert "No entity is at Watch level" in ew.build_digest(alerts.iloc[:0], world, det.as_of)
    report = ew.build_evaluation_report(world, det, model)
    assert all(m in report for m in list(ew.METHODS.values()) + [ev.MODEL_LABEL]) and "Limits" in report and "ROC AUC" in report
    text = ew.evaluation_headline(ev.compare_methods(det, world.events, world.entities), det.first_usable, det.as_of)
    assert text.startswith("Of 77 known disruptions") and "days earlier" in text
    with pytest.raises(ValueError, match="needs a table of known disruptions"):
        ew.build_evaluation_report(ew.World(world.signals, world.entities), det)


# ---------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys, world, det):
    data = tmp_path / "data"
    assert main(["sample", "--out", str(data)]) == 0
    assert "Simulated, not real data" in capsys.readouterr().out and (data / "events.csv").exists()
    pd.testing.assert_frame_equal(ew.load_world(data).signals, world.signals)

    assert main(["detect", "--data", str(data), "--as-of", "2026-03-10", "--out", str(tmp_path / "alerts.csv")]) == 0
    assert "As of 2026-03-10" in capsys.readouterr().out
    saved = pd.read_csv(tmp_path / "alerts.csv")
    replay = ew.build_alerts(world, det, "2026-03-10")  # the full-history run, looked up at the same date
    assert saved["entity"].tolist() == replay["entity"].tolist() and np.allclose(saved["risk_index"], replay["risk_index"])
    assert saved["onset_probability"].notna().all()  # a model fitted only on what was known that day

    assert main(["digest", "--data", str(data), "--no-model", "--out", str(tmp_path / "digest.html")]) == 0
    assert "Early-warning digest: 30 Jun 2026" in (tmp_path / "digest.html").read_text(encoding="utf-8")
    assert main(["explain", "--data", str(data), "--entity", "SUP-16"]) == 0
    assert "risk index" in capsys.readouterr().out
    assert main(["evaluate", "--data", str(data), "--report", str(tmp_path / "eval.html"), "--out", str(tmp_path / "compare.csv")]) == 0
    out = capsys.readouterr().out
    assert "Of 77 known disruptions" in out and "Onset model" in out
    assert len(pd.read_csv(tmp_path / "compare.csv")) == 5 and "Evaluation report" in (tmp_path / "eval.html").read_text(encoding="utf-8")

    assert main(["detect", "--data", str(tmp_path / "missing")]) == 1 and "signals.csv not found" in capsys.readouterr().err
    assert main(["detect", "--data", str(data), "--as-of", "2031-01-01"]) == 1 and "--as-of must be between" in capsys.readouterr().err
    assert main(["explain", "--data", str(data), "--entity", "SUP-99"]) == 1 and "not an entity" in capsys.readouterr().err
    (data / "events.csv").unlink()
    assert main(["evaluate", "--data", str(data)]) == 1 and "needs events.csv" in capsys.readouterr().err
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "signals.csv").write_text("a,b\n1,2\n")
    assert main(["detect", "--data", str(bad)]) == 1 and "Missing required column" in capsys.readouterr().err
