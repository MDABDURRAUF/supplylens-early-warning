"""What an alert puts at risk, and what to do about it.

Suppliers are linked to SKUs and yearly spend. A supplier alert exposes that
supplier's SKUs; a port, region or commodity alert exposes the SKUs of every
supplier that depends on it. Exposure says what is behind an alert; it is
not a forecast of loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EVENT_TYPES, World
from .detect import LEVELS, Detection, current_alerts
from .model import OnsetModel, upstream_of

# Suggested first step by likely cause and level. Rule-based text, not optimised advice.
ACTIONS = {
    "capacity": {
        "Watch": "Ask the supplier to confirm open order dates and current capacity; review the next four weeks of demand on its SKUs.",
        "Warning": "Hold a capacity review with the supplier, prioritise open orders by need date, and check the safety stock on its SKUs.",
        "Critical": "Expedite critical open orders, release orders to an approved second source where one exists, and raise safety stock on single-source SKUs.",
    },
    "quality": {
        "Watch": "Ask for the supplier's recent inspection data and tighten sampling on incoming lots.",
        "Warning": "Raise a corrective-action request, move to tightened incoming inspection, and check stock of good parts.",
        "Critical": "Quarantine suspect lots, inspect every incoming lot, and qualify or switch to an alternative source for the affected SKUs.",
    },
    "port_congestion": {
        "Watch": "Check containers booked through this port over the next two weeks and add buffer days to their expected arrival.",
        "Warning": "Advance bookings for urgent inbound orders and ask the forwarder for routings through another port.",
        "Critical": "Reroute urgent inbound freight through another port or mode and tell planners to expect late arrivals from the suppliers affected.",
    },
    "weather": {
        "Watch": "Check which inbound orders ship from this region in the coming week and confirm pickup dates with carriers.",
        "Warning": "Bring forward shipments that can leave early and confirm that suppliers in the region have contingency plans.",
        "Critical": "Assume delays for all suppliers in the region, re-plan production around their SKUs, and use stock or sources elsewhere.",
    },
    "price_spike": {
        "Watch": "Review open and planned buys that depend on this commodity and check the price clauses of the contracts affected.",
        "Warning": "Consider buying forward for near-term needs and ask affected suppliers how they will pass the cost on.",
        "Critical": "Lock in prices where contracts allow, update cost forecasts for the SKUs affected, and review alternative materials.",
    },
}


def _supplier_links(entities: pd.DataFrame) -> pd.DataFrame:
    """The port, region and commodity each supplier depends on, indexed by supplier."""
    return upstream_of(entities).loc[entities.loc[entities["type"] == "supplier", "entity"]]


def exposed_suppliers(entity: str, entities: pd.DataFrame) -> list[str]:
    """Suppliers behind an entity: itself for a supplier; for a port, region or commodity, every supplier that depends on it."""
    kinds = entities.set_index("entity")["type"]
    if entity not in kinds.index:
        raise ValueError(f"'{entity}' is not an entity in the data")
    if kinds[entity] == "supplier":
        return [entity]
    up = _supplier_links(entities)
    return up.index[up[kinds[entity]] == entity].tolist()


def exposed_skus(entity: str, entities: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """The SKU rows behind an entity, largest spend first, with ``single_source`` marking SKUs bought from one supplier only."""
    sources = skus.groupby("sku")["entity"].nunique()
    out = skus[skus["entity"].isin(exposed_suppliers(entity, entities))].copy()
    out["single_source"] = out["sku"].map(sources).to_numpy() == 1
    return out.sort_values("annual_spend_usd", ascending=False).reset_index(drop=True)


def exposure_table(entities: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """For every entity: how many suppliers and SKUs are behind it, and the yearly spend on them.

    ``single_source_skus`` counts exposed SKUs that have no other supplier.
    ``spend_share`` is the entity's exposed spend as a share of all spend.
    """
    up = _supplier_links(entities)
    sources = skus.groupby("sku")["entity"].nunique() if len(skus) else pd.Series(dtype=int)
    per = skus.assign(single=skus["sku"].map(sources).to_numpy() == 1) if len(skus) else skus.assign(single=[])
    total = float(skus["annual_spend_usd"].sum()) if len(skus) else 0.0
    rows = []
    for r in entities.itertuples():
        members = [r.entity] if r.type == "supplier" else up.index[up[r.type] == r.entity].tolist()
        mine = per[per["entity"].isin(members)]
        spend = float(mine["annual_spend_usd"].sum())
        rows.append({"entity": r.entity, "name": r.name, "type": r.type, "suppliers_exposed": len(members), "skus_exposed": int(mine["sku"].nunique()),
                     "single_source_skus": int(mine.loc[mine["single"], "sku"].nunique()), "annual_spend_exposed": spend,
                     "spend_share": spend / total if total else np.nan})
    return pd.DataFrame(rows).sort_values("annual_spend_exposed", ascending=False).reset_index(drop=True)


def likely_cause(entity_type: str, main_signal: str) -> str:
    """A rule of thumb for what kind of disruption an alert points to, from the entity type and its strongest signal."""
    if entity_type == "supplier":
        return "quality" if main_signal == "reject_rate" else "capacity"
    return {"port": "port_congestion", "region": "weather", "commodity": "price_spike"}.get(entity_type, "capacity")


def suggested_action(cause: str, level: str) -> str:
    """The first step to consider for an alert of this cause and level."""
    return ACTIONS.get(cause, ACTIONS["capacity"]).get(level, "")


def build_alerts(world: World, det: Detection, as_of=None, model: OnsetModel | None = None) -> pd.DataFrame:
    """Today's alerts with what is exposed and what to do, most serious first.

    One row per entity at Watch or higher on ``as_of`` (default: the last
    date). Sorted by level, then by exposed spend. Columns beyond those of
    ``detect.current_alerts``:

    ``likely_cause``          rule of thumb from the entity type and strongest signal
    ``upstream``              ports or regions this entity depends on that are also in alert
    ``onset_probability``     the onset model's estimate, when a model is given and the date
                              is in its test period (blank otherwise)
    ``suppliers_exposed``, ``skus_exposed``, ``single_source_skus``, ``annual_spend_exposed``
    ``risk_weighted_spend``   exposed spend x risk index / 100: a way to rank, not a loss forecast
    ``suggested_action``      first step to consider
    """
    date = det.as_of if as_of is None else pd.Timestamp(as_of).normalize()
    alerts = current_alerts(det, date)
    info = world.entities.set_index("entity")
    up = upstream_of(world.entities)
    exp = exposure_table(world.entities, world.skus).set_index("entity")
    levels = det.level.loc[date]
    alerts["name"] = alerts["entity"].map(info["name"]).to_numpy()
    alerts["type"] = alerts["entity"].map(info["type"]).to_numpy()
    alerts["likely_cause"] = [EVENT_TYPES[likely_cause(t, s)] for t, s in zip(alerts["type"], alerts["main_signal"])]
    alerts["upstream"] = [", ".join(f"{u} ({levels[u]})" for u in up.loc[e, ["port", "region"]] if u and levels.get(u, "")) for e in alerts["entity"]]
    p = np.full(len(alerts), np.nan)
    if model is not None and len(alerts) and date >= model.split_["test_from"]:
        p = model.predict(det, world.entities, dates=[date]).iloc[0].reindex(alerts["entity"]).to_numpy()
    alerts["onset_probability"] = p
    for c in ("suppliers_exposed", "skus_exposed", "single_source_skus", "annual_spend_exposed"):
        alerts[c] = alerts["entity"].map(exp[c]).to_numpy()
    alerts["risk_weighted_spend"] = alerts["annual_spend_exposed"] * alerts["risk_index"] / 100
    alerts["suggested_action"] = [suggested_action(likely_cause(t, s), lv) for t, s, lv in zip(alerts["type"], alerts["main_signal"], alerts["level"])]
    order = alerts["level"].map({lv: i for i, lv in enumerate(LEVELS)})
    alerts = alerts.assign(_order=order).sort_values(["_order", "annual_spend_exposed", "risk_index"], ascending=[True, False, False])
    cols = ["entity", "name", "type", "level", "risk_index", "change_7d", "since", "days_active", "likely_cause", "evidence", "upstream",
            "onset_probability", "suppliers_exposed", "skus_exposed", "single_source_skus", "annual_spend_exposed", "risk_weighted_spend",
            "suggested_action", "main_signal"]
    return alerts[cols].reset_index(drop=True)


def model_watchlist(world: World, det: Detection, model: OnsetModel, as_of=None, top: int = 10) -> pd.DataFrame:
    """Entities not yet at Watch that the onset model rates most likely to start a disruption within its horizon.

    Only entities at or above the model's alert threshold are listed. Empty
    when the date falls inside the model's training or calibration period.
    """
    date = det.as_of if as_of is None else pd.Timestamp(as_of).normalize()
    cols = ["entity", "name", "type", "onset_probability", "risk_index", "suppliers_exposed", "skus_exposed", "annual_spend_exposed"]
    if date < model.split_["test_from"]:
        return pd.DataFrame(columns=cols)
    p = model.predict(det, world.entities, dates=[date]).iloc[0]
    quiet = det.level.loc[date] == ""
    pick = p[quiet & (p >= model.alert_threshold_)].sort_values(ascending=False).head(top)
    exp = exposure_table(world.entities, world.skus).set_index("entity")
    out = pd.DataFrame({"entity": pick.index, "onset_probability": pick.to_numpy(), "risk_index": det.index.loc[date, pick.index].to_numpy()})
    for c in ("name", "type", "suppliers_exposed", "skus_exposed", "annual_spend_exposed"):
        out[c] = out["entity"].map(exp[c]).to_numpy()
    return out[cols]


def exposure_at_risk(alerts: pd.DataFrame, entities: pd.DataFrame, skus: pd.DataFrame) -> dict:
    """Totals behind today's alerts, counting each SKU-supplier row once even when several alerts cover it.

    Returns ``suppliers``, ``skus``, ``single_source_skus`` and
    ``annual_spend`` (and ``spend_share`` of all spend).
    """
    members = sorted({s for e in alerts["entity"] for s in exposed_suppliers(e, entities)}) if len(alerts) else []
    sources = skus.groupby("sku")["entity"].nunique() if len(skus) else pd.Series(dtype=int)
    mine = skus[skus["entity"].isin(members)]
    total = float(skus["annual_spend_usd"].sum()) if len(skus) else 0.0
    spend = float(mine["annual_spend_usd"].sum())
    return {"suppliers": len(members), "skus": int(mine["sku"].nunique()),
            "single_source_skus": int((mine["sku"].map(sources) == 1).sum()) if len(mine) else 0,
            "annual_spend": spend, "spend_share": spend / total if total else np.nan}
