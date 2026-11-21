"""What "normal" looks like for one signal of one entity, learned only from its own past.

For every day the baseline gives an expected value and a typical size of
day-to-day scatter. Both are built from medians, so a few bad days, a glitch
or the start of a disruption barely move them. Nothing here looks forward:
the numbers for a day use only the days before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_baseline(wide: pd.DataFrame, window: int = 91, season_weeks: int = 26, min_history: int = 42) -> dict[str, pd.DataFrame]:
    """Expected value, scatter and deviation score for every column of a daily table.

    ``wide`` has one row per calendar day and one column per series. Returns
    five tables of the same shape:

    ``level``     median of the previous ``window`` days (today excluded)
    ``expected``  ``level`` plus a weekday adjustment: the median, over the
                  previous ``season_weeks`` occurrences of today's weekday, of
                  how far that weekday sat from the level of its own day
    ``scale``     typical scatter: 1.4826 x the median absolute gap between
                  actual and expected over the previous ``window`` days. The
                  factor makes it comparable to a standard deviation when the
                  scatter is bell-shaped.
    ``z``         (actual - expected) / scale: how many typical scatters today
                  is away from normal. Blank until ``min_history`` days of
                  history exist, and on days with no reported value.

    ``persistence``  1 when one day's deviation says nothing about the next; larger
                  when bad days come in runs. It is the scatter of 7-day averages
                  of (actual - expected) over the previous ``window`` days, divided
                  by what that scatter would be for unrelated days. Values
                  below 1.25 are treated as 1. The detectors divide ``z`` by it, so that an ordinary
                  run of bad days in a slow-moving signal is not read as a shift.

    Set ``season_weeks=0`` to switch the weekday adjustment off.
    """
    level = wide.rolling(window, min_periods=min_history).median().shift(1)
    offset = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    if season_weeks:
        gap = wide - level
        weekday = np.asarray(wide.index.day_of_week)
        parts = [gap[weekday == k].rolling(season_weeks, min_periods=min(4, season_weeks)).median().shift(1) for k in range(7)]
        offset = pd.concat(parts).sort_index().fillna(0.0)
    expected = level + offset
    resid = wide - expected
    scale = 1.4826 * resid.abs().rolling(window, min_periods=min_history).median().shift(1)
    # A series that never moves would have zero scatter; keep a small floor so the score stays finite.
    scale = np.maximum(scale, 0.001 * level.abs() + 1e-9).where(scale.notna())
    # Persistence: how much bigger the scatter of 7-day averages is than it would be if days were unrelated.
    weekly = 1.4826 * resid.rolling(7, min_periods=4).mean().abs().rolling(window, min_periods=min_history).median().shift(1)
    persistence = (np.sqrt(7) * weekly / scale).fillna(1.0)
    persistence = persistence.where(persistence >= 1.25, 1.0)  # smaller values are what chance gives for unrelated days
    return {"level": level, "expected": expected, "scale": scale, "z": resid / scale, "persistence": persistence}
