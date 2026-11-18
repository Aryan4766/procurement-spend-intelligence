"""Supplier / item grouping quality vs the golden keys (company A), with and without the golden catch-all bucket.
Usage: python scripts/entity_resolution_report.py ../company_A  -> reports/entity_resolution_A.json"""
import src  # noqa: F401
import json
import sys
from pathlib import Path

import pandas as pd

from src.config.config_loader import load_and_validate
from src.evaluation.matching_metrics import pairwise_scores_excluding_bucket
from src.ingestion.data_loader import read_csv_text
from src.matching.item_resolver import resolve_items
from src.matching.supplier_resolver import inherit_from_primary, resolve_suppliers
from src.matching.transaction_matcher import match_transactions
from src.pipeline import apply_exclusions, finalise_inclusion, ingest_and_clean

cfg = load_and_validate("configs/company_a.yaml")
ddir = Path(sys.argv[1]).resolve()
txn = finalise_inclusion(cfg, match_transactions(apply_exclusions(cfg, ingest_and_clean(cfg, ddir)), cfg)[0])
sk, sd = resolve_suppliers(txn, cfg)
txn["supplier_key"] = inherit_from_primary(txn, sk).values
ik, idg = resolve_items(txn, txn["supplier_key"], cfg)
txn["item_key"] = inherit_from_primary(txn, ik).values
lab = read_csv_text(ddir / cfg["labels"]["file"])
res = {}
for kind, col in (("supplier", "supplier_key"), ("item", "item_key")):
    gold = dict(zip(lab["record_id"], lab[col]))
    sub = txn[txn.record_id.isin(gold)]
    res[kind] = pairwise_scores_excluding_bucket(sub[col], sub.record_id.map(gold))
res["supplier_diagnostics"] = {k: v for k, v in sd.items() if not k.startswith("_")}
res["item_diagnostics"] = idg
Path("reports/entity_resolution_A.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
print(json.dumps(res, indent=2, default=str))
