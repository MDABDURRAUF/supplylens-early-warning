"""Daily signals, entities, SKUs and disruption events: expected columns, validation, and a simulated US world."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .us import STATE_REGION, federal_holidays

# signal: (label, unit, direction that is bad (+1 = higher is worse), type of entity that reports it)
SIGNALS = {
    "lead_time_days": ("Supplier lead time", "days", 1, "supplier"),
    "on_time_rate": ("On-time delivery rate", "share", -1, "supplier"),
    "fill_rate": ("Fill rate", "share", -1, "supplier"),
    "reject_rate": ("Quality reject rate", "share", 1, "supplier"),
    "ack_delay_hours": ("Order-acknowledgment delay", "hours", 1, "supplier"),
    "dwell_days": ("Port dwell time", "days", 1, "port"),
    "weather_severity": ("Regional weather severity", "0 to 1", 1, "region"),
    "price_index": ("Commodity price index", "index", 1, "commodity"),
}
SIGNAL_LABEL = {k: v[0] for k, v in SIGNALS.items()}
BAD_DIRECTION = {k: v[2] for k, v in SIGNALS.items()}
ENTITY_TYPES = ["supplier", "port", "region", "commodity"]
EVENT_TYPES = {
    "capacity": "Supplier capacity problem",
    "quality": "Quality drift",
    "port_congestion": "Port congestion",
    "weather": "Regional weather event",
    "price_spike": "Commodity price spike",
}

SIGNAL_COLUMNS = ["date", "entity", "signal", "value"]
ENTITY_COLUMNS = ["entity", "type", "name", "state", "region", "port", "commodity", "category"]
SKU_COLUMNS = ["sku", "entity", "annual_spend_usd", "description"]
EVENT_COLUMNS = ["event_id", "entity", "type", "onset", "peak", "end", "severity"]


class SchemaError(ValueError):
    """Raised when an input table does not match the expected layout."""


@dataclass
class World:
    """Everything the system reads.

    ``signals``   one row per date, entity and signal (long format)
    ``entities``  one row per supplier, port, region or commodity
    ``skus``      which supplier provides which SKU, and the yearly spend
    ``events``    known disruptions, used only to score the detectors and to
                  train the onset model; may be empty
    """

    signals: pd.DataFrame
    entities: pd.DataFrame
    skus: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SKU_COLUMNS))
    events: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EVENT_COLUMNS))

    @property
    def first_date(self) -> pd.Timestamp:
        return self.signals["date"].min()

    @property
    def last_date(self) -> pd.Timestamp:
        return self.signals["date"].max()

    def as_of(self, date) -> "World":
        """The world as it was known on ``date``: no signal after it, and no event that began after it."""
        date = pd.Timestamp(date).normalize()
        return World(self.signals[self.signals["date"] <= date].reset_index(drop=True), self.entities, self.skus,
                     self.events[self.events["onset"] <= date].reset_index(drop=True))


# ---------------------------------------------------------------- validation
def _dates(column: pd.Series) -> pd.Series:
    """Parse a column of dates to midnight timestamps; anything unreadable becomes NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pandas warns when it cannot guess one format for the column
        return pd.to_datetime(column, errors="coerce").dt.normalize()


def validate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Check a signals table and return a cleaned copy, sorted by date.

    Expects ``date``, ``entity``, ``signal`` and ``value``. A blank value is
    allowed and means "not reported that day". Problems raise a
    ``SchemaError`` saying how many rows are affected; nothing is dropped
    silently.
    """
    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")
    out = df[SIGNAL_COLUMNS].copy()
    blank = out["value"].isna() | (out["value"].astype(str).str.strip() == "")
    out["date"] = _dates(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["entity"] = out["entity"].astype(str).str.strip()
    out["signal"] = out["signal"].astype(str).str.strip().str.lower()

    problems = []
    if out["date"].isna().any():
        problems.append(f"{int(out['date'].isna().sum())} row(s) with missing or unreadable 'date'")
    unreadable = out["value"].isna() & ~blank
    if unreadable.any():
        problems.append(f"{int(unreadable.sum())} row(s) with a 'value' that is not a number")
    unknown = sorted(set(out["signal"]) - set(SIGNALS))
    if unknown:
        problems.append(f"{int(out['signal'].isin(unknown).sum())} row(s) with an unknown signal ({', '.join(unknown[:5])}); "
                        f"known signals are {', '.join(SIGNALS)}")
    if out.duplicated(["date", "entity", "signal"]).any():
        problems.append(f"{int(out.duplicated(['date', 'entity', 'signal']).sum())} duplicate date/entity/signal row(s)")
    if not problems:
        kinds = out.assign(kind=out["signal"].map(lambda s: SIGNALS[s][3])).groupby("entity")["kind"].nunique()
        if (kinds > 1).any():
            problems.append(f"{int((kinds > 1).sum())} entity(ies) report signals of more than one kind "
                            "(for example a supplier signal and a port signal); use one entity name per supplier, port, region or commodity")
    if problems:
        raise SchemaError("Invalid signals data: " + "; ".join(problems))
    return out.sort_values(["date", "entity", "signal"]).reset_index(drop=True)


def validate_entities(entities: pd.DataFrame | None, signals: pd.DataFrame) -> pd.DataFrame:
    """Check the entities table against the signals and return a cleaned copy.

    The table is optional. An entity that appears in the signals but not in
    the table is added, with its type taken from the signals it reports.
    ``region`` is filled from ``state`` (Census regions). ``port`` and
    ``commodity`` link a supplier to the port its goods come through and the
    commodity it depends on; both may be blank.
    """
    kind = signals.assign(kind=signals["signal"].map(lambda s: SIGNALS[s][3])).groupby("entity")["kind"].first()
    if entities is None or len(entities) == 0:
        out = pd.DataFrame({"entity": kind.index, "type": kind.to_numpy()})
    else:
        if "entity" not in entities.columns:
            raise SchemaError("Missing required column(s): entity")
        out = entities.copy()
        out["entity"] = out["entity"].astype(str).str.strip()
        if out["entity"].duplicated().any():
            raise SchemaError(f"Invalid entities data: {int(out['entity'].duplicated().sum())} duplicate entity value(s)")
        extra = [e for e in kind.index if e not in set(out["entity"])]
        if extra:
            out = pd.concat([out, pd.DataFrame({"entity": extra})], ignore_index=True)
    for c in ENTITY_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[ENTITY_COLUMNS].astype(object)
    for c in ENTITY_COLUMNS:
        out[c] = out[c].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
    out["type"] = out["type"].str.lower()
    from_signals = out["entity"].map(kind).fillna("")
    out["type"] = out["type"].where(out["type"] != "", from_signals)
    out["name"] = out["name"].where(out["name"] != "", out["entity"])
    out["state"] = out["state"].str.upper()

    problems = []
    bad_type = ~out["type"].isin(ENTITY_TYPES)
    if bad_type.any():
        problems.append(f"{int(bad_type.sum())} row(s) with a type that is not one of {', '.join(ENTITY_TYPES)}")
    clash = (from_signals != "") & (out["type"] != from_signals)
    if clash.any():
        problems.append(f"{int(clash.sum())} entity(ies) whose type does not match the signals they report")
    bad_state = (out["state"] != "") & ~out["state"].isin(STATE_REGION)
    if bad_state.any():
        problems.append(f"{int(bad_state.sum())} row(s) with a state code that is not a US state")
    for c, t in (("port", "port"), ("commodity", "commodity")):
        bad = (out[c] != "") & ~out[c].isin(set(out.loc[out["type"] == t, "entity"]))
        if bad.any():
            problems.append(f"{int(bad.sum())} row(s) whose '{c}' is not a {t} in the entities table")
    if problems:
        raise SchemaError("Invalid entities data: " + "; ".join(problems))
    region = out["state"].map(STATE_REGION).fillna("")
    out["region"] = out["region"].where(out["region"] != "", region)
    return out.sort_values(["type", "entity"]).reset_index(drop=True)


def validate_skus(skus: pd.DataFrame | None, entities: pd.DataFrame) -> pd.DataFrame:
    """Check the SKU table: one row per SKU and supplier, with the yearly spend on that SKU from that supplier."""
    if skus is None or len(skus) == 0:
        return pd.DataFrame(columns=SKU_COLUMNS)
    missing = [c for c in ("sku", "entity", "annual_spend_usd") if c not in skus.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")
    out = skus.copy()
    if "description" not in out.columns:
        out["description"] = ""
    out = out[SKU_COLUMNS]
    out["sku"], out["entity"] = out["sku"].astype(str).str.strip(), out["entity"].astype(str).str.strip()
    out["description"] = out["description"].fillna("").astype(str)
    out["annual_spend_usd"] = pd.to_numeric(out["annual_spend_usd"], errors="coerce")
    problems = []
    suppliers = set(entities.loc[entities["type"] == "supplier", "entity"])
    if (~out["entity"].isin(suppliers)).any():
        problems.append(f"{int((~out['entity'].isin(suppliers)).sum())} row(s) whose entity is not a supplier in the data")
    if out["annual_spend_usd"].isna().any() or (out["annual_spend_usd"] < 0).any():
        problems.append("annual_spend_usd must be a number that is zero or more in every row")
    if out.duplicated(["sku", "entity"]).any():
        problems.append(f"{int(out.duplicated(['sku', 'entity']).sum())} duplicate sku/entity row(s)")
    if problems:
        raise SchemaError("Invalid SKU data: " + "; ".join(problems))
    return out.sort_values(["entity", "sku"]).reset_index(drop=True)


def validate_events(events: pd.DataFrame | None, entities: pd.DataFrame) -> pd.DataFrame:
    """Check the table of known disruptions: entity, type, onset, peak, end and severity (0 to 1).

    ``peak`` (the worst day) and ``severity`` are optional; without a peak, the end is used.
    """
    if events is None or len(events) == 0:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("onset", "peak", "end") else float if c == "severity" else object)
                             for c in EVENT_COLUMNS})
    missing = [c for c in ("entity", "type", "onset", "end") if c not in events.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")
    out = events.copy()
    if "event_id" not in out.columns:
        out["event_id"] = [f"EV-{i + 1:03d}" for i in range(len(out))]
    if "peak" not in out.columns:
        out["peak"] = out["end"]
    if "severity" not in out.columns:
        out["severity"] = np.nan
    out = out[EVENT_COLUMNS]
    for c in ("onset", "peak", "end"):
        out[c] = _dates(out[c])
    out["severity"] = pd.to_numeric(out["severity"], errors="coerce")
    for c in ("event_id", "entity", "type"):
        out[c] = out[c].astype(str).str.strip()
    problems = []
    if out[["onset", "peak", "end"]].isna().any().any():
        problems.append(f"{int(out[['onset', 'peak', 'end']].isna().any(axis=1).sum())} row(s) with a missing or unreadable date")
    elif ((out["peak"] < out["onset"]) | (out["end"] < out["peak"])).any():
        problems.append("onset, peak and end must be in that order in every row")
    if (~out["entity"].isin(set(entities["entity"]))).any():
        problems.append(f"{int((~out['entity'].isin(set(entities['entity']))).sum())} row(s) whose entity is not in the data")
    if problems:
        raise SchemaError("Invalid events data: " + "; ".join(problems))
    return out.sort_values(["onset", "entity"]).reset_index(drop=True)


def make_world(signals: pd.DataFrame, entities: pd.DataFrame | None = None, skus: pd.DataFrame | None = None,
               events: pd.DataFrame | None = None) -> World:
    """Validate the four tables together and return a ``World``. Only ``signals`` is required."""
    signals = validate_signals(signals)
    if signals.empty:
        raise SchemaError("Invalid signals data: the table has no rows")
    entities = validate_entities(entities, signals)
    return World(signals, entities, validate_skus(skus, entities), validate_events(events, entities))


def load_world(folder: str | Path) -> World:
    """Read ``signals.csv`` and, when present, ``entities.csv``, ``skus.csv`` and ``events.csv`` from a folder."""
    folder = Path(folder)
    if not (folder / "signals.csv").exists():
        raise FileNotFoundError(f"{folder / 'signals.csv'} not found")
    read = lambda name: pd.read_csv(folder / name) if (folder / name).exists() else None  # noqa: E731
    return make_world(pd.read_csv(folder / "signals.csv"), read("entities.csv"), read("skus.csv"), read("events.csv"))


def save_world(world: World, folder: str | Path) -> Path:
    """Write the four tables as CSV files into a folder (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    world.signals.assign(date=world.signals["date"].dt.strftime("%Y-%m-%d")).to_csv(folder / "signals.csv", index=False)
    world.entities.to_csv(folder / "entities.csv", index=False)
    world.skus.to_csv(folder / "skus.csv", index=False)
    ev = world.events.copy()
    for c in ("onset", "peak", "end"):
        ev[c] = pd.to_datetime(ev[c]).dt.strftime("%Y-%m-%d")
    ev.to_csv(folder / "events.csv", index=False)
    return folder


def wide_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Signals as a table with one row per calendar day and one column per (entity, signal).

    Every day between the first and last date is present; a value that was
    not reported is NaN.
    """
    wide = signals.pivot(index="date", columns=["entity", "signal"], values="value").sort_index(axis=1)
    return wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D", name="date"))


def holiday_mask(dates: pd.DatetimeIndex, days_after: int = 1) -> np.ndarray:
    """True for US federal holidays and the ``days_after`` days that follow each one."""
    years = range(int(dates.year.min()) - 1, int(dates.year.max()) + 1)
    hol = pd.DatetimeIndex([pd.Timestamp(d) for d in federal_holidays(years)])
    marked = hol
    for k in range(1, days_after + 1):
        marked = marked.union(hol + pd.Timedelta(days=k))
    return dates.normalize().isin(marked)


# --------------------------------------------------------------- sample data
_PORTS = {  # port entity: (name, state)
    "PORT-LAX": ("Los Angeles gateway", "CA"), "PORT-SEA": ("Seattle gateway", "WA"), "PORT-HOU": ("Houston gateway", "TX"),
    "PORT-SAV": ("Savannah gateway", "GA"), "PORT-NWK": ("Newark gateway", "NJ"),
}
_REGION_PORTS = {"West": ["PORT-LAX", "PORT-SEA"], "South": ["PORT-HOU", "PORT-SAV"], "Northeast": ["PORT-NWK", "PORT-NWK", "PORT-SAV"],
                 "Midwest": ["PORT-NWK", "PORT-LAX", "PORT-HOU", "PORT-SAV"]}
_REGION_STATES = {"Midwest": ["OH", "MI", "IL", "IN", "WI", "MN", "MO"], "South": ["TX", "GA", "NC", "TN", "SC", "AL", "KY"],
                  "Northeast": ["PA", "NY", "NJ", "MA", "CT"], "West": ["CA", "WA", "AZ", "OR", "UT", "CO"]}
_CATEGORIES = {  # what a supplier makes: the commodity it depends on
    "castings": "COM-STEEL", "fasteners": "COM-STEEL", "machined parts": "COM-STEEL", "stampings": "COM-STEEL",
    "injection moldings": "COM-RESIN", "packaging": "COM-RESIN", "resin compounds": "COM-RESIN",
    "circuit boards": "COM-ELEC", "sensors": "COM-ELEC", "wire harnesses": "COM-ELEC",
}
_COMMODITIES = {"COM-STEEL": "Steel price index", "COM-RESIN": "Resin price index", "COM-ELEC": "Electronic components price index"}
_WEATHER = {"Northeast": (0.16, 15), "Midwest": (0.18, 20), "South": (0.14, 245), "West": (0.12, 225)}  # typical level, day of year of the seasonal high
_EVENT_PLAN = {
    # type: (how many, days from onset to peak, days from onset to end, severity range)
    "port_congestion": (9, (12, 25), (35, 60), (0.4, 1.0)),
    "weather": (10, (4, 8), (12, 25), (0.4, 1.0)),
    "price_spike": (4, (15, 30), (50, 90), (0.5, 1.0)),
    "capacity": (36, (18, 40), (45, 85), (0.3, 1.0)),
    "quality": (18, (25, 50), (55, 90), (0.3, 1.0)),
}
_BURN_IN = 120  # no disruption starts in the first 120 days, so every baseline has a clean history to learn from


def event_shape(days: np.ndarray, onset: int, peak: int, end: int) -> np.ndarray:
    """How much of an event's full effect is present on each day: 0 before onset and after end, 1 at the peak.

    The effect builds slowly at first: ``((day - onset + 1) / (peak - onset + 1)) ** 1.5``
    up to the peak, then fades as ``1 - ((day - peak) / (end - peak + 1)) ** 2``.
    """
    days = np.asarray(days, dtype=float)
    rise = np.clip((days - onset + 1) / (peak - onset + 1), 0, None) ** 1.5
    fall = 1 - ((days - peak) / (end - peak + 1)) ** 2
    return np.where((days < onset) | (days > end), 0.0, np.where(days <= peak, rise, fall))


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Noise with day-to-day persistence ``phi`` and overall standard deviation ``sd``."""
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    out = np.zeros(n)
    out[0] = rng.normal(0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def _schedule(rng, kind: str, candidates: list[str], busy: dict, days: int, forced: list | None = None) -> list[dict]:
    """Place events of one type on random entities and dates so that no entity has two events within 45 days."""
    count, rise, length, sev = _EVENT_PLAN[kind]
    out, tries = [], 0
    forced = list(forced or [])
    while len(out) < count and tries < 5000:
        tries += 1
        if forced:
            entity, onset = forced.pop()
        else:
            entity, onset = candidates[int(rng.integers(len(candidates)))], int(rng.integers(_BURN_IN, days - 8))
        peak = onset + int(rng.integers(rise[0], rise[1] + 1))
        end = max(peak + 5, onset + int(rng.integers(length[0], length[1] + 1)))
        if onset < _BURN_IN or onset > days - 8 or any(onset - 45 <= b and end + 45 >= a for a, b in busy.get(entity, [])):
            continue
        busy.setdefault(entity, []).append((onset, end))
        out.append({"entity": entity, "type": kind, "onset": onset, "peak": peak, "end": end,
                    "severity": round(float(rng.uniform(*sev)), 2), "lead": int(rng.integers(14, 26))})
    return out


def generate_sample_world(days: int = 730, start: str = "2024-07-01", seed: int = 7) -> World:
    """A simulated supply base with known disruptions, for demos and tests. None of it is real data.

    40 suppliers in US states, 5 port gateways, the 4 Census regions and 3
    commodity price indices report daily signals. Each signal has its own
    level per entity, a weekday pattern, a mild yearly pattern, noise, a few
    one-day glitches, a few missing days, and a bump on and just after US
    federal holidays.

    Hidden rules inject the disruptions listed in ``events``. Each one builds
    up gradually from its ``onset`` to its ``peak`` and then fades by its
    ``end`` (see ``event_shape``):

    ``capacity``         lead time, acknowledgment delay up; on-time and fill rate down.
                         Acknowledgments start slowing 14 to 25 days *before* the onset.
    ``quality``          reject rate up, fill rate slightly down. On-time delivery
                         starts slipping 14 to 25 days *before* the onset.
    ``port_congestion``  dwell time up; suppliers using that port see longer lead times
    ``weather``          regional severity up; suppliers and ports in the region are affected
    ``price_spike``      a commodity price index rises

    Trouble spreads: about half of the weather events set off congestion at a
    port in the region 3 to 10 days later, and two thirds of the capacity
    problems are set off by a port or weather event upstream of the supplier,
    12 to 30 days after it starts. These early signs and knock-on effects are
    what the onset model can learn; they exist because the simulation puts
    them there. The detectors and the model never see the rules, only the
    signals.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    wd = np.asarray(dates.day_of_week)
    doy = np.asarray(dates.day_of_year)
    year_end = np.cos(2 * np.pi * (doy - 345) / 365.25)  # +1 in mid December, -1 in mid June
    holiday = holiday_mask(dates, days_after=1)

    # ------------------------------------------------------------- entities
    rows = []
    regions = list(_REGION_STATES)
    categories = list(_CATEGORIES)
    for i in range(40):
        region = regions[i % 4]
        state = _REGION_STATES[region][(i // 4) % len(_REGION_STATES[region])]
        category = categories[int(rng.integers(len(categories)))]
        port = _REGION_PORTS[region][int(rng.integers(len(_REGION_PORTS[region])))]
        rows.append({"entity": f"SUP-{i + 1:02d}", "type": "supplier", "name": f"Supplier {i + 1:02d} ({category}, {state})",
                     "state": state, "region": region, "port": port, "commodity": _CATEGORIES[category], "category": category})
    for p, (name, state) in _PORTS.items():
        rows.append({"entity": p, "type": "port", "name": name, "state": state, "region": STATE_REGION[state]})
    for r in regions:
        rows.append({"entity": f"REG-{r.upper()}", "type": "region", "name": f"{r} weather", "region": r})
    for c, name in _COMMODITIES.items():
        rows.append({"entity": c, "type": "commodity", "name": name})
    entities = pd.DataFrame(rows, columns=ENTITY_COLUMNS).fillna("")
    sup = entities[entities["type"] == "supplier"].set_index("entity")
    port_region = {p: STATE_REGION[s] for p, (_, s) in _PORTS.items()}

    # --------------------------------------------------------------- events
    busy: dict = {}
    events = _schedule(rng, "weather", [f"REG-{r.upper()}" for r in regions], busy, days)
    knock_on = []  # bad weather sets off congestion at a port in the region about half the time
    for e in events:
        ports = [p for p, r in port_region.items() if r == e["entity"][4:].title()]
        if ports and rng.random() < 0.5:
            knock_on.append((ports[int(rng.integers(len(ports)))], e["onset"] + int(rng.integers(3, 11))))
    events += _schedule(rng, "port_congestion", list(_PORTS), busy, days, forced=knock_on)
    triggered = []  # a port or weather event sets off capacity problems at one to three suppliers that depend on it
    for e in list(events):
        if e["type"] == "port_congestion":
            below = sup.index[sup["port"] == e["entity"]].tolist()
        else:
            below = sup.index[sup["region"] == e["entity"][4:].title()].tolist()
        for name in rng.permutation(below)[:int(rng.integers(1, 4))]:
            triggered.append((str(name), e["onset"] + int(rng.integers(12, 31))))
    triggered = [triggered[i] for i in rng.permutation(len(triggered))][:24]
    events += _schedule(rng, "price_spike", list(_COMMODITIES), busy, days)
    events += _schedule(rng, "capacity", sup.index.tolist(), busy, days, forced=triggered)
    events += _schedule(rng, "quality", sup.index.tolist(), busy, days)

    def effect(entity_events):
        total = np.zeros(days)
        for e in entity_events:
            total += e["severity"] * event_shape(t, e["onset"], e["peak"], e["end"])
        return total

    by_entity: dict = {}
    for e in events:
        by_entity.setdefault((e["entity"], e["type"]), []).append(e)
    port_effect = {p: effect(by_entity.get((p, "port_congestion"), [])) for p in _PORTS}
    weather_effect = {r: effect(by_entity.get((f"REG-{r.upper()}", "weather"), [])) for r in regions}

    series: dict = {}

    def finish(key, values, noise_sd, lo=None, hi=None, digits=3):
        """Add one-day glitches and missing days, clip to the valid range, and store the series."""
        values = np.array(values, dtype=float)
        glitch = rng.random(days) < 0.004
        values[glitch] += rng.choice([-1, 1], int(glitch.sum())) * rng.uniform(3, 6, int(glitch.sum())) * noise_sd
        if lo is not None or hi is not None:
            values = np.clip(values, lo, hi)
        values = np.round(values, digits)
        values[rng.random(days) < 0.004] = np.nan
        series[key] = values

    # -------------------------------------------------------------- weather
    for r in regions:
        base, high_day = _WEATHER[r]
        level = base + 0.03 * np.cos(2 * np.pi * (doy - high_day) / 365.25)
        blips = np.zeros(days)
        for s in np.flatnonzero(rng.random(days) < 0.012):  # ordinary short spells of bad weather, not listed as events
            blips[s:s + int(rng.integers(1, 4))] = rng.uniform(0.06, 0.14)
        finish((f"REG-{r.upper()}", "weather_severity"), level + _ar1(rng, days, 0.5, 0.04) + blips + 0.22 * weather_effect[r], 0.04, 0, 1)

    # ---------------------------------------------------------------- ports
    dwell_extra = {}
    for p in _PORTS:
        base = rng.uniform(2.8, 5.0)
        weekday = np.array([1.10, 1.04, 1.00, 0.97, 0.95, 1.00, 1.07])[wd]
        normal = base * weekday * (1 + 0.03 * np.cos(2 * np.pi * (doy - 260) / 365.25)) * np.where(holiday, 1.12, 1.0)
        lift = 0.34 * port_effect[p] + 0.05 * weather_effect[port_region[p]]
        dwell_extra[p] = base * 0.34 * port_effect[p]
        finish((p, "dwell_days"), normal * (1 + lift) * np.exp(_ar1(rng, days, 0.35, 0.06)), 0.06 * base, 0.2, None, 2)

    # ---------------------------------------------------------- commodities
    for c in _COMMODITIES:
        spike = effect(by_entity.get((c, "price_spike"), []))
        finish((c, "price_index"), 100 * np.exp(_ar1(rng, days, 0.93, 0.018)) * (1 + 0.09 * spike), 1.8, 1, None, 2)

    # ------------------------------------------------------------ suppliers
    for s, row in sup.iterrows():
        cap_events = by_entity.get((s, "capacity"), [])
        qual_events = by_entity.get((s, "quality"), [])
        cap, qual = effect(cap_events), effect(qual_events)

        def run_up(entity_events, share):
            """The early sign of an event: rises in a straight line over ``lead`` days to ``share`` of the full effect at
            the onset, and stays at least that high until the peak."""
            out = np.zeros(days)
            for e in entity_events:
                before = (t >= e["onset"] - e["lead"]) & (t < e["onset"])
                out[before] = np.maximum(out[before], share * e["severity"] * (t[before] - (e["onset"] - e["lead"]) + 1) / e["lead"])
                rising = (t >= e["onset"]) & (t <= e["peak"])
                out[rising] = np.maximum(out[rising], share * e["severity"])
            return out

        early = run_up(cap_events, 0.6)   # acknowledgments slow down before a capacity problem shows anywhere else
        strain = run_up(qual_events, 1.0)  # on-time delivery slips before quality starts to drift
        storm = weather_effect[row["region"]]
        congestion = np.concatenate([np.zeros(3), dwell_extra[row["port"]][:-3]])  # reaches the supplier's lead time 3 days later

        lead = rng.uniform(8, 30)
        lt = lead * np.array([1.03, 1.00, 0.99, 0.99, 1.00, 1.01, 1.02])[wd] * (1 + 0.025 * year_end) * np.where(holiday, 1.04, 1.0)
        lt = lt * (1 + 0.16 * cap + 0.05 * storm) * np.exp(_ar1(rng, days, 0.3, 0.05)) + 0.5 * congestion
        finish((s, "lead_time_days"), lt, 0.05 * lead, 1, None, 2)

        otd = rng.uniform(0.90, 0.97)
        p = otd + np.array([-0.012, 0.0, 0.004, 0.004, 0.0, -0.004, -0.008])[wd] - 0.01 * year_end.clip(0) - 0.06 * cap - 0.03 * strain - 0.02 * storm \
            - 0.01 * port_effect[row["port"]]
        finish((s, "on_time_rate"), p + rng.normal(0, 0.035, days), 0.035, 0, 1, 4)

        fill = rng.uniform(0.94, 0.985)
        finish((s, "fill_rate"), fill - 0.004 * (wd == 0) - 0.03 * cap - 0.008 * qual + rng.normal(0, 0.012, days), 0.012, 0, 1, 4)

        rej = rng.uniform(0.008, 0.03)
        inspected = 1500
        q = np.clip(rej * (1 + 0.15 * (wd == 0)) * (1 + 0.7 * qual), 0, 1)
        finish((s, "reject_rate"), rng.binomial(inspected, q) / inspected, np.sqrt(rej / inspected), 0, 1, 4)

        ack = rng.uniform(6, 30)
        hours = ack * np.array([1.00, 0.97, 0.97, 1.03, 1.30, 1.45, 1.20])[wd] * np.where(holiday, 1.5, 1.0)
        hours = hours * (1 + 0.7 * np.maximum(cap, early)) * rng.lognormal(0, 0.18, days)
        finish((s, "ack_delay_hours"), hours, 0.18 * ack, 0.1, None, 1)

    wide = pd.DataFrame(series, index=dates)
    signals = pd.DataFrame({"date": np.repeat(dates.to_numpy(), wide.shape[1]),
                            "entity": np.tile([k[0] for k in wide.columns], days),
                            "signal": np.tile([k[1] for k in wide.columns], days),
                            "value": wide.to_numpy().ravel()}).dropna(subset=["value"])

    # ----------------------------------------------------------------- SKUs
    sku_rows, n = [], 0
    for s, row in sup.iterrows():
        for _ in range(int(rng.integers(2, 6))):
            n += 1
            spend = float(np.round(rng.lognormal(np.log(180_000), 0.8), -2))
            sku, desc = f"SKU-{n:04d}", f"{row['category']} item {n:04d}"
            others = sup.index[(sup["category"] == row["category"]) & (sup.index != s)].tolist()
            if others and rng.random() < 0.25:  # dual-sourced: a second supplier of the same category takes part of the spend
                share = float(rng.uniform(0.5, 0.8))
                sku_rows.append({"sku": sku, "entity": s, "annual_spend_usd": round(spend * share, -2), "description": desc})
                sku_rows.append({"sku": sku, "entity": others[int(rng.integers(len(others)))],
                                 "annual_spend_usd": round(spend * (1 - share), -2), "description": desc})
            else:
                sku_rows.append({"sku": sku, "entity": s, "annual_spend_usd": spend, "description": desc})

    ev = pd.DataFrame(events).sort_values(["onset", "entity"]).reset_index(drop=True)
    last = days - 1
    ev = pd.DataFrame({
        "event_id": [f"EV-{i + 1:03d}" for i in range(len(ev))], "entity": ev["entity"], "type": ev["type"],
        "onset": dates[ev["onset"].to_numpy()], "peak": dates[ev["peak"].clip(upper=last).to_numpy()],
        "end": dates[ev["end"].clip(upper=last).to_numpy()], "severity": ev["severity"],
    })
    return make_world(signals, entities, pd.DataFrame(sku_rows), ev)
