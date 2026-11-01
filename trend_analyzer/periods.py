"""Project-wide non-overlapping research periods."""
from pathlib import Path

import pandas as pd

from .common import read_json


def period_settings():
    return read_json(Path(__file__).resolve().parents[1] / "config" / "periods.json")


def validate_periods(cfg):
    result = dict(cfg)
    result["old_end"] = result.get("old_end", str((pd.Timestamp(cfg["new_start"])-pd.Timedelta(days=1)).date()))
    result.setdefault("new_end", None)
    if pd.Timestamp(result["old_start"]) > pd.Timestamp(result["old_end"]):
        raise ValueError("Old period start must be on or before its end.")
    if pd.Timestamp(result["old_end"]) >= pd.Timestamp(result["new_start"]):
        raise ValueError("Old and new periods must not overlap: old end must be before new start.")
    if result["new_end"] and pd.Timestamp(result["new_start"]) > pd.Timestamp(result["new_end"]):
        raise ValueError("New period start must be on or before its end.")
    return result


def period_windows(calendar, cfg=None):
    cfg = validate_periods(cfg or period_settings())
    dates = pd.DatetimeIndex(calendar)
    return {"old":dates[(dates >= cfg["old_start"]) & (dates <= cfg["old_end"])],
            "new":dates[(dates >= cfg["new_start"]) & (dates <= cfg["new_end"] if cfg["new_end"] else True)]}


def period_caption(cfg=None):
    cfg = validate_periods(cfg or period_settings())
    return f"Old: {cfg['old_start']} → {cfg['old_end']} · New: {cfg['new_start']} → {cfg['new_end'] or 'latest saved completed session'}"
