"""CSV ingestion: every source file is read as text and mapped onto the engine's logical schema."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.config_loader import ConfigError
from src.ingestion.schema_validator import validate_source_frame


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read a CSV with everything as string (no silent type coercion, blanks stay blank)."""
    if not path.exists():
        raise ConfigError(f"Input file not found: {path}")
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError as e:  # try next encoding
            last = e
    raise ConfigError(f"Cannot decode {path}: {last}")


def resolve_dir(cfg: dict, key: str, override: Optional[str]) -> Path:
    if override:  # command-line paths are relative to where the user runs the command
        return Path(override).expanduser().resolve()
    base = Path(cfg["_config_path"]).parent  # config paths are relative to the config file
    d = Path(cfg.get(key, cfg.get("data_dir", "."))).expanduser()
    return d if d.is_absolute() else (base / d).resolve()


def _first_nonblank(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        nxt = df[c].astype(str)
        out = out.where(out.str.strip() != "", nxt)
    return out


def _joined(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [df[c].astype(str).str.strip() for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = (out + " " + p).str.strip()
    return out


# logical fields that are free text and should be concatenated when several columns are given
_CONCAT_FIELDS = {"description"}


def load_source(src: dict, data_dir: Path) -> pd.DataFrame:
    """Return the raw frame (all columns, text) after validating the configured columns."""
    df = read_csv_text(data_dir / src["file"])
    validate_source_frame(src, df)
    return df


def standardise_source(src: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Map a raw source frame to logical columns + provenance columns (source, record_id, src_row)."""
    out = pd.DataFrame(index=raw.index)
    out["raw_id"] = raw[src["id_column"]].astype(str)
    out["record_id"] = out["raw_id"].map(lambda x: src["record_id_template"].format(id=x))
    out["source"] = src["name"]
    out["role"] = src.get("role", "standalone")
    out["src_row"] = range(len(raw))
    for logical, col in (src.get("fields") or {}).items():
        cols = col if isinstance(col, list) else [col]
        out[logical] = _joined(raw, cols) if logical in _CONCAT_FIELDS else _first_nonblank(raw, cols)
    for logical, col in (src.get("context_fields") or {}).items():
        cols = col if isinstance(col, list) else [col]
        out["ctx_" + logical] = _first_nonblank(raw, cols)
    for logical in ("description", "supplier_name", "supplier_id", "item_code", "po", "invoice_number", "status"):
        if logical not in out.columns:
            out[logical] = ""
    amt = src.get("amount") or {}
    out["_currency_raw"] = raw[amt["currency_column"]].astype(str).str.strip().str.upper() if amt.get("currency_column") else str(amt.get("currency", "")).upper()
    out["_rate_raw"] = raw[amt["rate_column"]].astype(str) if amt.get("rate_column") else ""
    if src.get("duplicate_key"):
        out["_dupkey"] = raw[src["duplicate_key"]].astype(str).agg("\x1f".join, axis=1)
    else:
        out["_dupkey"] = ""
    return out
