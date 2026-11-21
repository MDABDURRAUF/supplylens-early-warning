"""Change detection and the composite risk index.

Four ways of deciding that a signal has left its normal range, from naive to
combined:

* **static threshold**: a fixed limit per signal, set once from its first
  weeks; the simple rule the others are compared with;
* **EWMA**: an exponentially weighted moving average of the daily deviation
  score, which reacts to a small shift that persists;
* **CUSUM**: a running sum of deviations above an allowance, which reacts to
  slow drifts;
* **risk index** (0 to 100): evidence from EWMA and CUSUM, combined across
  all of an entity's signals, with levels Watch, Warning and Critical.

Every number for a day is computed from that day and the days before it.
Running the detection on data cut off at a date gives exactly the same
history up to that date as running it on the full data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import robust_baseline
from .data import BAD_DIRECTION, SIGNAL_LABEL, holiday_mask, validate_signals, wide_signals

LEVELS = ["Critical", "Warning", "Watch"]
METHODS = {"static": "Static threshold", "ewma": "EWMA", "cusum": "CUSUM", "index": "Risk index (Watch or higher)"}


@dataclass(frozen=True)
class Settings:
    """Detector settings. The defaults were chosen on the simulated sample data.

    ``window``, ``season_weeks``, ``min_history``  baseline (see ``baseline.robust_baseline``)
    ``z_clip``          scores fed to EWMA and CUSUM are capped at plus or minus this, so that one
                        glitch cannot set off an alarm by itself
    ``jump_limit``      a single day's deviation score counts as evidence in the risk index in
                        proportion to this limit (see ``detect``)
    ``skip_holidays``   ignore US federal holidays and the day after each
    ``ewma_lambda``     weight of today in the EWMA (0.2 means roughly the last 9 days matter)
    ``ewma_limit``      EWMA alarm limit, in standard deviations of the EWMA under normal conditions
    ``cusum_k``         CUSUM allowance: only deviations above this many scatters accumulate
    ``cusum_limit``     CUSUM alarm limit
    ``static_days``     days at the start used to set the static limits
    ``static_k``        static limit: median plus this many scatters (in the bad direction)
    ``static_persist``  the static rule fires when the limit is broken this many days in a row
    ``watch``, ``warning``, ``critical``  risk index needed for each level
    ``confirm_days``    the index must stay at or above a level's threshold this many days in a row
    ``gap``             an entity that alerts again within this many days continues the same alert
    """

    window: int = 91
    season_weeks: int = 26
    min_history: int = 42
    z_clip: float = 4.0
    jump_limit: float = 3.5
    skip_holidays: bool = True
    ewma_lambda: float = 0.2
    ewma_limit: float = 3.5
    cusum_k: float = 0.5
    cusum_limit: float = 7.0
    static_days: int = 90
    static_k: float = 3.0
    static_persist: int = 2
    watch: float = 40.0
    warning: float = 60.0
    critical: float = 85.0
    confirm_days: int = 2
    gap: int = 7

    @property
    def burn_in(self) -> int:
        """Days of history before alerts are trusted: the baseline window plus four weeks."""
        return self.window + 28


def ewma_score(z: pd.DataFrame, lam: float = 0.2) -> pd.DataFrame:
    """Exponentially weighted moving average of ``z``, in units of its own standard deviation.

    ``average_today = lam * z_today + (1 - lam) * average_yesterday``, starting
    from 0. If ``z`` were independent with standard deviation 1, the average
    would settle at a standard deviation of ``sqrt(lam / (2 - lam))``; the
    result is divided by that, so a score of 3 means "3 standard deviations
    of the average". A day with no ``z`` leaves the average unchanged.
    """
    arr, out = z.to_numpy(dtype=float), np.zeros(z.shape)
    state = np.zeros(z.shape[1])
    for i in range(len(arr)):
        ok = ~np.isnan(arr[i])
        state = np.where(ok, lam * np.nan_to_num(arr[i]) + (1 - lam) * state, state)
        out[i] = state
    return pd.DataFrame(out / np.sqrt(lam / (2 - lam)), index=z.index, columns=z.columns)


def cusum(z: pd.DataFrame, k: float = 0.5, cap: float | None = None) -> pd.DataFrame:
    """One-sided CUSUM: ``sum_today = max(0, sum_yesterday + z_today - k)``, starting from 0.

    Deviations smaller than the allowance ``k`` pull the sum back toward zero;
    a drift that stays above ``k`` makes it grow day after day. ``cap`` limits
    how high the sum can go, so that it comes back down within a reasonable
    time once the signal recovers. A day with no ``z`` leaves the sum unchanged.
    """
    arr, out = z.to_numpy(dtype=float), np.zeros(z.shape)
    state = np.zeros(z.shape[1])
    for i in range(len(arr)):
        ok = ~np.isnan(arr[i])
        state = np.where(ok, np.maximum(0.0, state + np.nan_to_num(arr[i]) - k), state)
        if cap is not None:
            state = np.minimum(state, cap)
        out[i] = state
    return pd.DataFrame(out, index=z.index, columns=z.columns)


def static_limits(wide: pd.DataFrame, direction: pd.Series, days: int = 90, k: float = 3.0) -> pd.Series:
    """A fixed limit per series, set from its first ``days`` days and never updated.

    The limit is the median of those days plus ``k`` scatters (1.4826 x the
    median absolute deviation) in the bad direction: above the median where
    higher is worse, below it where lower is worse. No weekday adjustment.
    """
    first = wide.iloc[:days]
    med = first.median()
    mad = 1.4826 * (first - med).abs().median()
    return med + direction * k * mad


def run_lengths(flags: pd.DataFrame) -> pd.DataFrame:
    """For each cell, how many days in a row (ending today) the flag has been on."""
    arr, out = flags.to_numpy(dtype=bool), np.zeros(flags.shape, dtype=int)
    run = np.zeros(flags.shape[1], dtype=int)
    for i in range(len(arr)):
        run = np.where(arr[i], run + 1, 0)
        out[i] = run
    return pd.DataFrame(out, index=flags.index, columns=flags.columns)


def episodes(flags: pd.DataFrame, gap: int = 7) -> pd.DataFrame:
    """Turn daily alert flags (dates x entities) into alerts with a start and a last day.

    An entity's flagged day starts a new alert only if none of the previous
    ``gap`` days was flagged; otherwise it continues the alert in progress.
    Returns one row per alert: ``entity``, ``start``, ``last`` and ``days``
    (number of flagged days).
    """
    rows = []
    dates = flags.index
    for entity in flags.columns:
        idx = np.flatnonzero(flags[entity].to_numpy(dtype=bool))
        if not len(idx):
            continue
        cuts = np.flatnonzero(np.diff(idx) > gap) + 1
        for part in np.split(idx, cuts):
            rows.append((entity, dates[part[0]], dates[part[-1]], len(part)))
    return pd.DataFrame(rows, columns=["entity", "start", "last", "days"])


@dataclass
class Detection:
    """Everything the detectors computed. Tables are indexed by date.

    Per series, with columns (entity, signal):

    ``values``, ``expected``, ``scale``  what was reported, and the baseline
    ``z``         deviation score, signed so that positive always means "worse"
    ``z_adj``     ``z`` divided by the signal's persistence and capped at ``z_clip``
    ``ewma``      EWMA score of ``z_adj``
    ``cusum``     CUSUM of ``z_adj``
    ``static``    whether the static rule fires
    ``strength``  evidence from this signal, 0 to 1 (see ``detect``)

    Per entity:

    ``pooled``    ``z``, ``ewma``, ``cusum`` and ``strength`` of all the entity's signals
                  taken together (blank for an entity with a single signal)
    ``index``     risk index, 0 to 100
    ``held``      the lowest index over the last ``confirm_days`` days, which sets the level
    ``level``     "Critical", "Warning", "Watch" or ""
    ``flags``     for each method in ``METHODS``, whether the entity is in alert
    """

    settings: Settings
    values: pd.DataFrame
    expected: pd.DataFrame
    scale: pd.DataFrame
    z: pd.DataFrame
    z_adj: pd.DataFrame
    ewma: pd.DataFrame
    cusum: pd.DataFrame
    static: pd.DataFrame
    static_limit: pd.Series
    strength: pd.DataFrame
    pooled: dict
    index: pd.DataFrame
    held: pd.DataFrame
    level: pd.DataFrame
    flags: dict = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.index.index

    @property
    def as_of(self) -> pd.Timestamp:
        return self.index.index[-1]

    @property
    def first_usable(self) -> pd.Timestamp:
        """First date with enough history behind it for alerts to be trusted."""
        return self.dates[min(self.settings.burn_in, len(self.dates) - 1)]

    def signals_of(self, entity: str) -> list[str]:
        return [s for e, s in self.values.columns if e == entity]


def detect(signals: pd.DataFrame, as_of=None, settings: Settings | None = None) -> Detection:
    """Run every detector on the signals up to ``as_of`` (default: the last date in the data).

    Steps, each using only the past:

    1. Baseline and deviation score ``z`` per series (``baseline.robust_baseline``),
       signed so that positive is worse, and ignored on federal holidays and
       the day after. ``z_adj`` is ``z`` divided by the series' persistence
       and capped at ``z_clip``.
    2. EWMA score and CUSUM of ``z_adj`` per series; the static rule on raw values.
    3. Evidence per signal.
       ``ratio = max(EWMA / ewma_limit, CUSUM / cusum_limit, z / jump_limit)``
       says how close the signal is to an alarm (1.0 = at the limit). The
       first two react to shifts that persist, the third to a sudden jump.
       ``strength = (ratio - 0.5)``, kept between 0 and 1: nothing below half
       way to the limit, 0.5 at the limit, 1 at one and a half times the limit.
    4. Evidence from all of an entity's signals together. The day's ``z_adj`` are
       added up and divided by the square root of how many there are; EWMA,
       CUSUM and strength are computed on that sum as for one signal (without
       the jump term). Several signals each leaning a little the wrong way
       add up here long before any one of them reaches its own limit.
       Entities with a single signal skip this step.
    5. Risk index per entity:
       ``100 x (1 - (1 - pooled strength) x product over its signals of (1 - strength))``.
       One signal at its limit gives 50; two signals each at 80% of their
       limits give 51, although neither would alarm alone.
    6. Level: Watch, Warning or Critical when the index has been at or above
       ``watch``, ``warning`` or ``critical`` for ``confirm_days`` days in a row.
    """
    s = settings or Settings()
    if not {"date", "entity", "signal", "value"} <= set(signals.columns) or not pd.api.types.is_datetime64_any_dtype(signals["date"]):
        signals = validate_signals(signals)
    if as_of is not None:
        signals = signals[signals["date"] <= pd.Timestamp(as_of).normalize()]
    if signals.empty:
        raise ValueError("There are no signals on or before the as-of date")
    wide = wide_signals(signals)
    direction = pd.Series([BAD_DIRECTION[sig] for _, sig in wide.columns], index=wide.columns, dtype=float)
    base = robust_baseline(wide, s.window, s.season_weeks, s.min_history)
    z = base["z"] * direction
    if s.skip_holidays:
        z = z.mask(holiday_mask(wide.index, days_after=1)[:, None] & np.ones(z.shape, dtype=bool))
    z_adj = (z / base["persistence"]).clip(-s.z_clip, s.z_clip)
    e = ewma_score(z_adj, s.ewma_lambda)
    c = cusum(z_adj, s.cusum_k, cap=1.5 * s.cusum_limit)

    limit = static_limits(wide, direction, s.static_days, s.static_k)
    beyond = ((wide - limit) * direction > 0)
    beyond.iloc[:s.static_days] = False
    static = run_lengths(beyond) >= s.static_persist

    ratio = np.maximum(np.maximum(e / s.ewma_limit, c / s.cusum_limit), (z / s.jump_limit).fillna(0.0))
    strength = (ratio - 0.5).clip(0, 1)
    entities = sorted(set(wide.columns.get_level_values(0)))

    # All of an entity's signals together: the sum of today's scores over the square root of how many were reported,
    # which behaves like one more deviation score when the signals are unrelated.
    reported = z_adj.notna().T.groupby(level=0).sum().T.reindex(columns=entities)
    several = z_adj.T.groupby(level=0).size().reindex(entities) > 1
    pool_z = (z_adj.fillna(0.0).T.groupby(level=0).sum().T.reindex(columns=entities) / np.sqrt(reported.where(reported > 0)))
    pool_z = pool_z.clip(-s.z_clip, s.z_clip).where(several.to_numpy()[None, :] & np.ones(pool_z.shape, dtype=bool))
    pool_e, pool_c = ewma_score(pool_z, s.ewma_lambda), cusum(pool_z, s.cusum_k, cap=1.5 * s.cusum_limit)
    pool_strength = (np.maximum(pool_e / s.ewma_limit, pool_c / s.cusum_limit) - 0.5).clip(0, 1)

    def any_signal(flag: pd.DataFrame) -> pd.DataFrame:
        return flag.T.groupby(level=0).any().T.reindex(columns=entities)

    index = 100 * (1 - (1 - strength).T.groupby(level=0).prod().T.reindex(columns=entities) * (1 - pool_strength))
    held = index.rolling(s.confirm_days, min_periods=s.confirm_days).min().fillna(0.0)  # lowest index over the confirming days
    v = held.to_numpy()
    level = pd.DataFrame(np.select([v >= s.critical, v >= s.warning, v >= s.watch], LEVELS, ""), index=index.index, columns=entities)
    flags = {"static": any_signal(static), "ewma": any_signal(e >= s.ewma_limit), "cusum": any_signal(c >= s.cusum_limit),
             "index": held >= s.watch}
    pooled = {"z": pool_z, "ewma": pool_e, "cusum": pool_c, "strength": pool_strength}
    return Detection(s, wide, base["expected"], base["scale"], z, z_adj, e, c, static, limit, strength, pooled, index, held, level, flags)


def explain(det: Detection, entity: str, date=None) -> pd.DataFrame:
    """Why an entity has the risk index it has on a date: one row per signal, strongest evidence first.

    ``value`` and ``expected`` are that day's figures in the signal's own
    units; ``value_7d`` and ``expected_7d`` are their averages over the last 7
    days; ``z`` is the day's deviation score (positive = worse than normal);
    ``ewma`` and ``cusum`` are the detector states; ``strength`` is the
    evidence (0 to 1) that goes into the index. For an entity with several
    signals, the row "All signals together" is the pooled evidence.
    """
    date = det.as_of if date is None else pd.Timestamp(date).normalize()
    if entity not in det.index.columns:
        raise ValueError(f"'{entity}' is not an entity in the data")
    if date not in det.index.index:
        raise ValueError(f"{date:%Y-%m-%d} is outside the data ({det.dates[0]:%Y-%m-%d} to {det.as_of:%Y-%m-%d})")
    rows = []
    week = slice(date - pd.Timedelta(days=6), date)
    for sig in det.signals_of(entity):
        key = (entity, sig)
        seen = det.values.loc[week, key]
        rows.append({"signal": sig, "label": SIGNAL_LABEL[sig], "value": det.values.at[date, key], "expected": det.expected.at[date, key],
                     "value_7d": seen.mean(), "expected_7d": det.expected.loc[week, key].where(seen.notna()).mean(),
                     "z": det.z.at[date, key], "ewma": det.ewma.at[date, key], "cusum": det.cusum.at[date, key],
                     "static_rule": bool(det.static.at[date, key]), "strength": det.strength.at[date, key]})
    if det.pooled["z"][entity].notna().any():
        p = {k: v.at[date, entity] for k, v in det.pooled.items()}
        rows.append({"signal": "all_signals", "label": "All signals together", "value": np.nan, "expected": np.nan, "value_7d": np.nan,
                     "expected_7d": np.nan, "z": p["z"], "ewma": p["ewma"], "cusum": p["cusum"], "static_rule": False, "strength": p["strength"]})
    return pd.DataFrame(rows).sort_values("strength", ascending=False, kind="stable").reset_index(drop=True)


def evidence_text(table: pd.DataFrame, top: int = 2) -> str:
    """The strongest signals from ``explain`` as one line: the 7-day average against what is normal for those days."""
    parts = []
    own = table[table["signal"] != "all_signals"]
    shown = own[own["strength"] > 0].head(top)
    shown = shown[(shown["strength"] >= 0.15) | (shown.index == shown.index[0])] if len(shown) else shown  # leave out faint second signals
    if shown.empty and len(table) and table["strength"].max() > 0:  # only the pooled evidence is raised: name the signals leaning most
        return "Several signals slightly worse than normal: " + ", ".join(own.sort_values("ewma", ascending=False)["label"].head(3))
    for _, r in shown.iterrows():
        if pd.isna(r["value_7d"]) or pd.isna(r["expected_7d"]):
            parts.append(f"{r['label']} raised over recent days")
        elif _fmt(r["signal"], r["value_7d"]) != _fmt(r["signal"], r["expected_7d"]) or not parts:  # skip one that has since recovered
            parts.append(f"{r['label']} {_fmt(r['signal'], r['value_7d'])} against {_fmt(r['signal'], r['expected_7d'])} normal")
    return "; ".join(parts)


def _fmt(signal: str, v: float) -> str:
    if signal in ("on_time_rate", "fill_rate", "reject_rate"):
        return f"{v:.1%}"
    if signal == "weather_severity":
        return f"{v:.2f}"
    return f"{v:,.1f}"


def current_alerts(det: Detection, date=None) -> pd.DataFrame:
    """Entities at Watch or higher on a date, highest index first.

    ``since`` is the first day of the alert in progress (an entity that drops
    below Watch for up to ``gap`` days and comes back continues the same
    alert); ``days_active`` counts from then; ``change_7d`` is the index today
    minus the index seven days earlier.
    """
    date = det.as_of if date is None else pd.Timestamp(date).normalize()
    if date not in det.index.index:
        raise ValueError(f"{date:%Y-%m-%d} is outside the data ({det.dates[0]:%Y-%m-%d} to {det.as_of:%Y-%m-%d})")
    pos = det.index.index.get_loc(date)
    today = det.index.iloc[pos]
    week_ago = det.index.iloc[max(0, pos - 7)]
    flags = det.flags["index"].iloc[:pos + 1]
    rows = []
    for entity in today.index[flags.iloc[pos].to_numpy()]:
        idx = np.flatnonzero(flags[entity].to_numpy())
        start = idx[-1]
        for j in range(len(idx) - 1, 0, -1):
            if idx[j] - idx[j - 1] > det.settings.gap:
                break
            start = idx[j - 1]
        table = explain(det, entity, date)
        rows.append({"entity": entity, "level": det.level.iat[pos, det.level.columns.get_loc(entity)], "risk_index": float(today[entity]),
                     "change_7d": float(today[entity] - week_ago[entity]), "since": det.dates[start], "days_active": int(pos - start + 1),
                     "main_signal": table.loc[table["signal"] != "all_signals"].sort_values(["strength", "ewma"], ascending=False)["signal"].iloc[0],
                     "evidence": evidence_text(table)})
    out = pd.DataFrame(rows, columns=["entity", "level", "risk_index", "change_7d", "since", "days_active", "main_signal", "evidence"])
    return out.sort_values("risk_index", ascending=False).reset_index(drop=True)
