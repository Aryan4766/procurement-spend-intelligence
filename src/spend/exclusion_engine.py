"""Decides what counts as spend and assigns exactly one reason code to everything excluded.

Precedence (configurable, default): CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
A record's own state (cancelled / not approved / not a supplier purchase) always beats "already counted
elsewhere": a duplicate of an excluded record is excluded for the record's own reason.
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.text_normalizer import normalize_text

CANCELLED, NOT_APPROVED, NON_PROCUREMENT, DUPLICATE = "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"
EXCLUSION_CODES_DEFAULT = (CANCELLED, NOT_APPROVED, NON_PROCUREMENT, DUPLICATE)


def _status_codes(status: pd.Series, spec: dict) -> pd.Series:
    """Map raw status text to CANCELLED / NOT_APPROVED / '' (approved)."""
    norm = status.fillna("").astype(str).map(normalize_text)
    approved = {normalize_text(x) for x in spec.get("approved", [])}
    cancelled = {normalize_text(x) for x in spec.get("cancelled", [])}
    blank_ok = spec.get("blank_status", "approved") == "approved"
    out = pd.Series("", index=status.index, dtype=object)
    is_blank = norm.eq("")
    is_cancel = norm.isin(cancelled) | norm.map(lambda s: any(c in s.split() for c in cancelled) if s else False)
    is_ok = norm.isin(approved)
    out[is_cancel] = CANCELLED
    out[~is_cancel & ~is_ok & ~is_blank] = NOT_APPROVED  # rejected, on hold, submitted, disputed, unknown ...
    if not blank_ok:
        out[is_blank] = NOT_APPROVED
    return out


def apply_status(df: pd.DataFrame, status_specs: dict, default_spec: dict) -> pd.Series:
    out = pd.Series("", index=df.index, dtype=object)
    for source, grp in df.groupby("source"):
        spec = {**default_spec, **(status_specs.get(source) or {})}
        if "status" in grp.columns and grp["status"].astype(str).str.strip().ne("").any():
            out.loc[grp.index] = _status_codes(grp["status"], spec)
    return out


def _rule_mask(df: pd.DataFrame, rule: dict) -> Optional[pd.Series]:
    fields = rule.get("fields") or [rule.get("field")]
    fields = [f for f in fields if f in df.columns]
    if not fields:
        return None
    mask = pd.Series(False, index=df.index)
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    rx = re.compile(rule["regex"], flags) if rule.get("regex") else None
    values = {str(v).strip().casefold() for v in rule.get("values", [])} if rule.get("values") else None
    for f in fields:
        col = df[f].fillna("").astype(str)
        if rx is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mask |= col.str.contains(rx, na=False)
        if values is not None:
            mask |= col.str.strip().str.casefold().isin(values)
    return mask


def apply_non_procurement(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Return (flag, rule_name). Rules reference logical field names; a rule whose fields are absent is skipped."""
    flag = pd.Series(False, index=df.index)
    which = pd.Series("", index=df.index, dtype=object)
    for rule in rules:
        m = _rule_mask(df, rule)
        if m is None:
            continue
        if rule.get("sources"):
            m &= df["source"].isin(rule["sources"])
        new = m & ~flag
        which[new] = rule.get("name", "rule")
        flag |= m
    return flag, which


def apply_within_source_duplicates(df: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """Exact re-exports inside one source: identical configured duplicate_key. First occurrence is kept."""
    dup = pd.Series(False, index=df.index)
    has_key = df["_dupkey"].astype(str).ne("")
    cand = df[eligible & has_key]
    if cand.empty:
        return dup
    d = cand.duplicated(["source", "_dupkey"], keep="first")
    dup.loc[d[d].index] = True
    return dup


def combine(df: pd.DataFrame, status_code: pd.Series, np_flag: pd.Series, dup_flag: pd.Series,
            precedence: list[str]) -> pd.Series:
    """Resolve one reason per record following the configured precedence."""
    reasons = pd.Series("", index=df.index, dtype=object)
    cond = {CANCELLED: status_code.eq(CANCELLED), NOT_APPROVED: status_code.eq(NOT_APPROVED),
            NON_PROCUREMENT: np_flag, DUPLICATE: dup_flag}
    for code in reversed(precedence):  # lowest priority first so higher priority overwrites
        reasons[cond[code]] = code
    return reasons
