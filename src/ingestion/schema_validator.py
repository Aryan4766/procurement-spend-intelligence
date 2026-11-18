"""Checks that every column named in a config exists in the file it refers to, with readable errors."""
from __future__ import annotations

import pandas as pd

from src.config.config_loader import ConfigError


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def source_required_columns(src: dict) -> list[str]:
    cols = [src["id_column"]]
    for logical, col in (src.get("fields") or {}).items():
        cols += _as_list(col)
    for logical, col in (src.get("context_fields") or {}).items():
        cols += _as_list(col)
    amt = src.get("amount") or {}
    for key in ("currency_column", "rate_column"):
        if amt.get(key):
            cols.append(amt[key])
    cols += _as_list(src.get("duplicate_key"))
    return list(dict.fromkeys(cols))


def validate_source_frame(src: dict, df: pd.DataFrame) -> None:
    missing = [c for c in source_required_columns(src) if c not in df.columns]
    if missing:
        raise ConfigError(
            f"Source '{src['name']}' ({src['file']}): configured column(s) not found: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if df[src["id_column"]].duplicated().any():
        n = int(df[src["id_column"]].duplicated().sum())
        raise ConfigError(f"Source '{src['name']}': id_column '{src['id_column']}' has {n} duplicated ids; "
                          "record ids must be unique so every input row appears exactly once in the output")


def validate_frame_columns(name: str, df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ConfigError(f"{name}: expected column(s) {missing} not found. Available: {list(df.columns)}")
