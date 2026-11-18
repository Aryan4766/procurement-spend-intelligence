"""Automated output validator: every rule of the output contract is checked and reported (never silently patched)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.output.output_generator import OUTPUT_COLUMNS
from src.taxonomy.taxonomy_loader import Taxonomy

VALID_CODES = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def validate_output(out: pd.DataFrame, input_ids: list[str], taxonomy: Taxonomy) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    problems: list[str] = []
    if list(out.columns) != OUTPUT_COLUMNS:
        problems.append(f"columns differ from contract: {list(out.columns)}")
        return problems
    ids = out["record_id"].astype(str)
    if ids.duplicated().any():
        problems.append(f"{int(ids.duplicated().sum())} duplicate record_id values")
    missing = set(input_ids) - set(ids)
    extra = set(ids) - set(input_ids)
    if missing:
        problems.append(f"{len(missing)} input records missing from the output (e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{len(extra)} output records not in the input (e.g. {sorted(extra)[:3]})")
    if not out["include"].isin([0, 1]).all():
        problems.append("include must be 0 or 1")
    inc = out["include"] == 1
    if (out.loc[inc, "exclusion_reason"].fillna("") != "").any():
        problems.append("included records must have a blank exclusion_reason")
    bad_codes = ~out.loc[~inc, "exclusion_reason"].isin(VALID_CODES)
    if bad_codes.any():
        problems.append(f"{int(bad_codes.sum())} excluded records without a valid exclusion code")
    if (out.loc[~inc, "spend"].astype(float) != 0).any():
        problems.append("excluded records must have spend == 0")
    conf = out["confidence"].astype(float)
    if ((conf < 0) | (conf > 1) | conf.isna()).any():
        problems.append("confidence must be within [0, 1]")
    if out["spend"].isna().any():
        problems.append("spend contains NaN")
    # category validity
    lvl_cols = ["l1", "l2", "l3", "l4"][: taxonomy.depth]
    valid_paths = {tuple(x for x in taxonomy.padded(i)) for i in range(taxonomy.n_leaves)}
    paths = list(zip(*[out.loc[inc, c].fillna("").astype(str) for c in lvl_cols]))
    bad = sum(1 for p in paths if tuple(p) not in valid_paths)
    if bad:
        problems.append(f"{bad} included records have a category path that is not in the taxonomy")
    for c in ["l1", "l2", "l3", "l4"][taxonomy.depth:]:
        if (out[c].fillna("") != "").any():
            problems.append(f"{c} must be blank (taxonomy has {taxonomy.depth} levels)")
    if out.loc[~inc, ["l1", "l2", "l3", "l4"]].fillna("").ne("").any().any():
        problems.append("excluded records must have blank categories")
    for c in ("supplier_key", "item_key"):
        if out[c].fillna("").eq("").all():
            problems.append(f"{c} is empty for every record")
    return problems


def reconcile_spend(txn: pd.DataFrame, out: pd.DataFrame) -> dict:
    """Totals that must hold: included spend equals the sum of included per-record spend; nothing is double counted."""
    inc = out["include"] == 1
    return {
        "records": int(len(out)),
        "included": int(inc.sum()),
        "excluded": int((~inc).sum()),
        "total_spend_included": round(float(out.loc[inc, "spend"].sum()), 2),
        "excluded_by_reason": out.loc[~inc, "exclusion_reason"].value_counts().to_dict(),
    }
