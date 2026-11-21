"""The onset model: the chance that a disruption begins at an entity within the next 14 days.

The detectors in ``detect`` react to a signal that has already moved. This
model is trained on past disruptions to recognise what the days *before* an
onset look like: for example order acknowledgments slowing down, or trouble
at the port or in the region a supplier depends on.

All splits are by time. The model is trained on the oldest part of the
history, calibrated on the next part, and tested on the most recent part. A
day is used for training only if its 14-day outcome was already known when
the training period ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import ENTITY_TYPES, SIGNALS, World
from .detect import Detection, detect

LINKS = {"port": "port", "region": "region", "commodity": "commodity"}


def upstream_of(entities: pd.DataFrame) -> pd.DataFrame:
    """For each entity, the port, region and commodity entities it depends on (blank where there is none).

    A supplier depends on its port, on the weather of its region and on its
    commodity; a port depends on the weather of its region.
    """
    region_entity = dict(zip(entities.loc[entities["type"] == "region", "region"], entities.loc[entities["type"] == "region", "entity"]))
    out = entities.set_index("entity")[["type", "port", "commodity", "region"]].copy()
    out["region"] = np.where(out["type"].isin(["supplier", "port"]), out["region"].map(lambda r: region_entity.get(r, "")), "")
    out.loc[out["type"] != "supplier", ["port", "commodity"]] = ""
    return out[["port", "region", "commodity"]]


def build_features(det: Detection, entities: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for every date and entity, indexed by (date, entity). Each uses only data up to its date.

    For each of the eight signals (blank where the entity does not report it):

    ``<signal>_z7``     average deviation score over the last 7 days
    ``<signal>_trend``  the last 7 days against the 14 days before them
    ``<signal>_ewma``   EWMA score

    About the entity: its risk index, how many of its signals are raised
    (EWMA score above 1.5), and its type.

    About the port, the region and the commodity it depends on (blank where
    there is none): today's risk index, the highest index in the last 21 days,
    and the number of the last 30 days spent at Watch or higher.
    """
    ents = list(det.index.columns)
    cols = {}
    for sig in SIGNALS:
        have = [e for e, s in det.z_adj.columns if s == sig]
        z = det.z_adj.xs(sig, axis=1, level=1)[have] if have else pd.DataFrame(index=det.dates)
        z7 = z.rolling(7, min_periods=3).mean()
        frames = {"z7": z7, "trend": z7 - z.shift(7).rolling(14, min_periods=5).mean(),
                  "ewma": det.ewma.xs(sig, axis=1, level=1)[have] if have else z}
        for name, frame in frames.items():
            cols[f"{sig}_{name}"] = frame.reindex(columns=ents).to_numpy(dtype=float).ravel()
    cols["risk_index"] = det.index.to_numpy(dtype=float).ravel()
    raised = (det.ewma > 1.5).T.groupby(level=0).sum().T.reindex(columns=ents)
    cols["signals_raised"] = raised.to_numpy(dtype=float).ravel()
    known = entities.set_index("entity")["type"].reindex(ents)
    cols["entity_type"] = np.tile(known.map({t: i for i, t in enumerate(ENTITY_TYPES)}).to_numpy(dtype=float), len(det.dates))
    up = upstream_of(entities).reindex(ents).fillna("")
    recent_high = det.index.rolling(21, min_periods=1).max()
    days_in_alert = det.flags["index"].astype(float).rolling(30, min_periods=1).sum()
    for link in LINKS:
        target = up[link].to_numpy()  # a blank link becomes an all-NaN column
        cols[f"{link}_index"] = det.index.reindex(columns=target).to_numpy(dtype=float).ravel()
        cols[f"{link}_index_high21"] = recent_high.reindex(columns=target).to_numpy(dtype=float).ravel()
        cols[f"{link}_alert_days30"] = days_in_alert.reindex(columns=target).to_numpy(dtype=float).ravel()
    return pd.DataFrame(cols, index=pd.MultiIndex.from_product([det.dates, ents], names=["date", "entity"]))


def onset_labels(dates: pd.DatetimeIndex, entities: list[str], events: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """For every date and entity: does a disruption begin within the next ``horizon`` days?

    ``label`` is 1 when one of the entity's events has its onset after the
    date and at most ``horizon`` days later. ``in_event`` marks days from an
    onset to the event's end, when the question no longer applies. ``known``
    is False for the last ``horizon`` days of the data, whose outcome cannot
    be known yet.
    """
    label = np.zeros((len(dates), len(entities)), dtype=int)
    inside = np.zeros_like(label, dtype=bool)
    pos = {e: i for i, e in enumerate(entities)}
    day = dates.to_numpy()
    for ev in events.itertuples():
        if ev.entity not in pos:
            continue
        onset, end = np.datetime64(ev.onset), np.datetime64(ev.end)
        label[(day < onset) & (day >= onset - np.timedelta64(horizon, "D")), pos[ev.entity]] = 1
        inside[(day >= onset) & (day <= end), pos[ev.entity]] = True
    known = np.repeat(day <= day[-1] - np.timedelta64(horizon, "D"), len(entities))
    return pd.DataFrame({"label": label.ravel(), "in_event": inside.ravel(), "known": known},
                        index=pd.MultiIndex.from_product([dates, entities], names=["date", "entity"]))


@dataclass
class OnsetModel:
    """Fit with ``fit(world)``, then ``predict(detection, entities)``.

    ``horizon``          days ahead the model looks
    ``train_fraction``   share of the usable history used for training
    ``calib_fraction``   share used to calibrate the probabilities
    ``alert_share``      the model raises its own alert at the probability that this share of
                         entity-days in the calibration period reached (stored in ``alert_threshold_``)

    The rest of the history is the held-out test period; its predictions are
    kept in ``holdout_`` for the checks in ``evaluate``. Days before
    ``split_['test_from']`` were seen in training or calibration, so
    predictions for them are not a fair replay.
    """

    horizon: int = 14
    train_fraction: float = 0.55
    calib_fraction: float = 0.20
    alert_share: float = 0.02
    alert_threshold_: float = 1.0
    random_state: int = 0
    features_: list = field(default_factory=list, repr=False)
    holdout_: pd.DataFrame | None = field(default=None, repr=False)
    split_: dict = field(default_factory=dict, repr=False)

    def fit(self, world: World, detection: Detection | None = None) -> "OnsetModel":
        if world.events is None or world.events.empty:
            raise ValueError("The onset model needs a table of past disruptions (events) to learn from")
        if not 0 < self.train_fraction < 1 or not 0 < self.calib_fraction < 1 or self.train_fraction + self.calib_fraction >= 1:
            raise ValueError("train_fraction and calib_fraction must be positive and sum to less than 1")
        det = detection if detection is not None else detect(world.signals)
        X = build_features(det, world.entities)
        y = onset_labels(det.dates, list(det.index.columns), world.events, self.horizon)
        date = X.index.get_level_values("date")
        usable = det.dates[det.dates >= det.first_usable]
        if len(usable) < 200:
            raise ValueError("At least 200 days of history beyond the baseline period are needed to fit the onset model")
        t1 = usable[int(len(usable) * self.train_fraction)]
        t2 = usable[int(len(usable) * (self.train_fraction + self.calib_fraction))]
        gap = pd.Timedelta(days=self.horizon)
        open_ = ~y["in_event"].to_numpy()
        train = (date >= usable[0]) & (date < t1 - gap) & open_
        calib = (date >= t1) & (date < t2 - gap) & open_
        test = date >= t2
        if y.loc[train, "label"].nunique() < 2 or y.loc[calib, "label"].nunique() < 2:
            raise ValueError("The training and calibration periods must each contain days before an onset and ordinary days")

        self.features_ = list(X.columns)
        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.04, max_leaf_nodes=5, min_samples_leaf=150, l2_regularization=5.0,
            random_state=self.random_state).fit(X[train], y.loc[train, "label"])
        # Calibration: a one-input logistic regression on the raw score, fitted on days the classifier never saw.
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._raw(X[calib]).reshape(-1, 1), y.loc[calib, "label"])
        self.alert_threshold_ = float(np.quantile(self._calibrate(self._raw(X[calib])), 1 - self.alert_share))
        self.base_rate_ = float(y.loc[train, "label"].mean())
        self.split_ = {"first_day": usable[0], "train_until": t1, "test_from": t2, "last_day": det.as_of,
                       "train_rows": int(train.sum()), "calibration_rows": int(calib.sum()),
                       "train_onsets": int(((world.events["onset"] >= usable[0]) & (world.events["onset"] < t1)).sum()),
                       "calibration_onsets": int(((world.events["onset"] >= t1) & (world.events["onset"] < t2)).sum()),
                       "test_onsets": int((world.events["onset"] >= t2).sum())}
        h = y[test].copy()
        h["p_raw"] = self.classifier_.predict_proba(X[test])[:, 1]
        h["p"] = self._calibrate(self._raw(X[test]))
        h["risk_index"] = X.loc[test, "risk_index"].to_numpy()
        self.holdout_ = h.reset_index()
        return self

    def _check(self) -> None:
        if not self.features_:
            raise RuntimeError("Call fit() first")

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        """The classifier's score on the log-odds scale."""
        p = np.clip(self.classifier_.predict_proba(X[self.features_])[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict(self, detection: Detection, entities: pd.DataFrame, dates=None) -> pd.DataFrame:
        """Calibrated probability that a disruption begins within ``horizon`` days: a table of dates x entities.

        The probability answers the question for an entity that is not
        already in a disruption. ``dates`` limits the output (default: every date).
        """
        self._check()
        X = build_features(detection, entities)
        if dates is not None:
            X = X[X.index.get_level_values("date").isin(pd.DatetimeIndex(dates))]
        p = pd.Series(self._calibrate(self._raw(X)), index=X.index)
        return p.unstack("entity").reindex(columns=detection.index.columns)

    def alert_flags(self, detection: Detection, entities: pd.DataFrame) -> pd.DataFrame:
        """Dates x entities: whether the probability is at or above ``alert_threshold_``."""
        return self.predict(detection, entities) >= self.alert_threshold_
