"""Command line.

    early-warning sample --out data
    early-warning detect --data data --as-of 2026-05-15 --out alerts.csv
    early-warning digest --data data --as-of 2026-05-15 --out digest.html
    early-warning explain --data data --entity SUP-07 --as-of 2026-05-15
    early-warning evaluate --data data --report evaluation.html

``--data`` is a folder holding ``signals.csv`` and, optionally,
``entities.csv``, ``skus.csv`` and ``events.csv``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__, evaluate
from .data import SchemaError, World, generate_sample_world, load_world, save_world
from .detect import detect, explain
from .impact import build_alerts, model_watchlist
from .model import OnsetModel
from .report import build_digest, build_evaluation_report, digest_headline, evaluation_headline, write_html


def _show(df: pd.DataFrame, formatters: dict | None = None) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 70):
        print(df.to_string(index=False, formatters=formatters or {}, na_rep=""))


def _fit(world: World, det, quiet: bool = False) -> OnsetModel | None:
    """The onset model fitted on what was known as of the detection date, or None when there is too little to learn from."""
    if world.events.empty:
        return None
    try:
        return OnsetModel().fit(world, det)
    except ValueError as exc:
        if not quiet:
            print(f"(onset model not used: {exc})")
        return None


def _as_of(world: World, as_of: str | None) -> tuple[World, pd.Timestamp]:
    date = world.last_date if as_of is None else pd.Timestamp(as_of).normalize()
    if not world.first_date <= date <= world.last_date:
        raise ValueError(f"--as-of must be between {world.first_date:%Y-%m-%d} and {world.last_date:%Y-%m-%d}")
    return world.as_of(date), date


def _alerts(args) -> tuple[World, pd.Timestamp, pd.DataFrame, pd.DataFrame | None]:
    """Alerts as of a date, using only signals up to that date and only events that had begun by then."""
    world, date = _as_of(load_world(args.data), args.as_of)
    det = detect(world.signals)
    model = None if args.no_model else _fit(world, det)
    alerts = build_alerts(world, det, date, model)
    return world, date, alerts, model_watchlist(world, det, model, date) if model is not None else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="early-warning", description="Supply chain early warning: detect, explain and score alerts.")
    p.add_argument("--version", action="version", version=f"early-warning {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write the simulated world (signals, entities, SKUs, events) as CSV files")
    s.add_argument("--out", default="data", help="folder to write into")
    s.add_argument("--seed", type=int, default=7)

    for name, text in (("detect", "list the entities in alert as of a date"), ("digest", "write the daily alert digest as an HTML page")):
        d = sub.add_parser(name, help=text)
        d.add_argument("--data", required=True, help="folder with signals.csv (and optionally entities.csv, skus.csv, events.csv)")
        d.add_argument("--as-of", help="date to stand on, YYYY-MM-DD (default: the last date in the data); later data is ignored")
        d.add_argument("--no-model", action="store_true", help="skip the onset model")
        d.add_argument("--out", **({"help": "write the alerts to this CSV file"} if name == "detect" else
                                   {"default": "digest.html", "help": "HTML file to write"}))

    x = sub.add_parser("explain", help="show why one entity has the risk index it has")
    x.add_argument("--data", required=True)
    x.add_argument("--entity", required=True)
    x.add_argument("--as-of")

    e = sub.add_parser("evaluate", help="score every method against the known disruptions in events.csv")
    e.add_argument("--data", required=True)
    e.add_argument("--report", help="write the evaluation report to this HTML file")
    e.add_argument("--out", help="write the comparison table to this CSV file")
    e.add_argument("--seeds", type=int, default=0, help="also average the comparison over this many other simulated histories")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pc, one, signed = "{:.0%}".format, "{:.1f}".format, "{:+.1f}".format
    day = lambda d: f"{pd.Timestamp(d):%Y-%m-%d}"  # noqa: E731
    comp_fmt = {"detection_rate": pc, "median_days_from_onset": one, "median_days_gained": signed, "false_alarms_per_month": one,
                "precision": pc, "events": "{:.0f}".format}
    try:
        if args.command == "sample":
            world = generate_sample_world(seed=args.seed)
            save_world(world, args.out)
            print(f"Wrote signals.csv ({len(world.signals):,} rows, {world.first_date:%Y-%m-%d} to {world.last_date:%Y-%m-%d}), entities.csv "
                  f"({len(world.entities)}), skus.csv ({len(world.skus)}) and events.csv ({len(world.events)}) to {args.out}. "
                  "Simulated, not real data.")
        elif args.command in ("detect", "digest"):
            world, date, alerts, watchlist = _alerts(args)
            print(f"As of {day(date)}: {digest_headline(alerts, world)}")
            if args.command == "detect":
                if len(alerts):
                    print()
                    _show(alerts[["entity", "level", "risk_index", "since", "likely_cause", "evidence", "annual_spend_exposed"]],
                          {"risk_index": "{:.0f}".format, "since": day, "annual_spend_exposed": "${:,.0f}".format})
                if watchlist is not None and len(watchlist):
                    print("\nNot in alert, but rated likely to start within 14 days (onset model):")
                    _show(watchlist[["entity", "onset_probability", "risk_index", "annual_spend_exposed"]],
                          {"onset_probability": pc, "risk_index": "{:.0f}".format, "annual_spend_exposed": "${:,.0f}".format})
                if args.out:
                    alerts.to_csv(args.out, index=False)
                    print(f"\nWrote {args.out}")
            else:
                print(f"Wrote {write_html(build_digest(alerts, world, date, watchlist), args.out)}")
        elif args.command == "explain":
            world, date = _as_of(load_world(args.data), args.as_of)
            det = detect(world.signals)
            table = explain(det, args.entity, date)
            level = det.level.at[date, args.entity] or "no alert"
            print(f"{args.entity} on {day(date)}: risk index {det.index.at[date, args.entity]:.0f} ({level}).")
            _show(table[["label", "value_7d", "expected_7d", "z", "ewma", "cusum", "strength"]],
                  {"value_7d": "{:.3f}".format, "expected_7d": "{:.3f}".format, "z": signed, "ewma": signed, "cusum": one, "strength": "{:.2f}".format})
        elif args.command == "evaluate":
            world = load_world(args.data)
            if world.events.empty:
                raise ValueError("evaluate needs events.csv (known disruptions) in the data folder")
            det = detect(world.signals)
            model = _fit(world, det)
            full = evaluate.compare_methods(det, world.events, world.entities)
            print(evaluation_headline(full, det.first_usable, det.as_of))
            print(f"\nDetectors, {day(det.first_usable)} to {day(det.as_of)}:")
            _show(full.drop(columns=["detected", "events_compared", "false_alarms"]), comp_fmt)
            print("\nRisk index by alert level:")
            _show(evaluate.by_level(det, world.events, world.entities)[["level", "alerts", "precision", "recall", "median_days_from_onset",
                                                                       "false_alarms_per_month"]],
                  {"precision": pc, "recall": pc, "median_days_from_onset": one, "false_alarms_per_month": one})
            table = full
            if model is not None:
                table = evaluate.compare_methods(det, world.events, world.entities, model)
                s = evaluate.model_summary(model)
                print(f"\nAll methods on the onset model's test period, {day(model.split_['test_from'])} to {day(det.as_of)}:")
                _show(table.drop(columns=["detected", "events_compared", "false_alarms"]), comp_fmt)
                print(f"\nOnset model on {s['entity_days']:,} test entity-days: ROC AUC {s['roc_auc']:.3f} (risk index as the score: "
                      f"{s['index_roc_auc']:.3f}); Brier score {s['brier_score']:.4f} against {s['brier_baseline']:.4f} for the base rate.")
            if args.seeds:
                print(f"\nAverage over {args.seeds} other simulated histories (seeds 1 to {args.seeds}), each on its own test period:")
                _show(evaluate.across_seeds(range(1, args.seeds + 1)).drop(columns="histories"), comp_fmt)
            if args.out:
                table.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
            if args.report:
                print(f"Wrote {write_html(build_evaluation_report(world, det, model), args.report)}")
    except (SchemaError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
