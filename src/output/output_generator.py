"""Builds the one-row-per-input-record output file in the exact required schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                  "l1", "l2", "l3", "l4", "confidence"]


def build_output(txn: pd.DataFrame, depth: int) -> pd.DataFrame:
    """`txn` must carry: record_id, include, exclusion_reason, supplier_key, item_key, spend_final,
    cat_l1..cat_l{depth}, confidence."""
    out = pd.DataFrame({
        "record_id": txn["record_id"].values,
        "include": txn["include"].astype(int).values,
        "exclusion_reason": txn["exclusion_reason"].fillna("").values,
        "supplier_key": txn["supplier_key"].fillna("").values,
        "item_key": txn["item_key"].fillna("").values,
        "spend": txn["spend_final"].astype(float).round(2).values,
    })
    for lvl in range(1, 5):
        col = f"cat_l{lvl}"
        out[f"l{lvl}"] = txn[col].fillna("").values if (lvl <= depth and col in txn.columns) else ""
    out["confidence"] = txn["confidence"].astype(float).clip(0, 1).round(4).values
    # excluded rows carry no category
    ex = out["include"] == 0
    out.loc[ex, ["l1", "l2", "l3", "l4"]] = ""
    out.loc[ex, "spend"] = 0.0
    out.loc[~ex, "exclusion_reason"] = ""
    return out[OUTPUT_COLUMNS]


def write_output(out: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8", float_format="%.2f", lineterminator="\n")
    return path
