"""End-to-end orchestration: ingest -> clean -> exclude -> match -> resolve entities -> classify -> output."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.classification.hybrid_classifier import HybridClassifier
from src.config.config_loader import load_and_validate
from src.evaluation import classification_metrics as cm
from src.evaluation.matching_metrics import pairwise_scores, pairwise_scores_excluding_bucket
from src.evaluation.pipeline_validation import reconcile_spend, validate_output
from src.ingestion.data_loader import load_source, read_csv_text, resolve_dir, standardise_source
from src.matching.item_resolver import resolve_items
from src.matching.supplier_resolver import inherit_from_primary, resolve_suppliers
from src.matching.transaction_matcher import match_transactions
from src.output.output_generator import build_output, write_output
from src.preprocessing.text_normalizer import clean_description
from src.spend import exclusion_engine as ex
from src.spend.currency_converter import FxTable
from src.spend.financials import compute_spend, normalise_dates
from src.taxonomy.taxonomy_loader import load_taxonomy

CROSS_SOURCE_DUPLICATE_STATUSES = {"MATCHED_PRIMARY", "AGGREGATE_MATCH", "DUP_OF_PRIMARY", "EXPORT_DUPLICATE"}


class StageTimer:
    def __init__(self):
        self.times: dict[str, float] = {}

    def run(self, name: str, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.times[name] = round(time.perf_counter() - t0, 2)
        return out


def ingest_and_clean(cfg: dict, data_dir: Path) -> pd.DataFrame:
    """Load every source, map to the logical schema, normalise dates/amounts/currency. One row per input record."""
    fx = FxTable.from_config(cfg, data_dir)
    frames = []
    for src in cfg["sources"]:
        raw = load_source(src, data_dir)
        std = standardise_source(src, raw)
        std = normalise_dates(std, src)
        std = compute_spend(std, src, cfg, fx)
        frames.append(std)
    txn = pd.concat(frames, ignore_index=True, sort=False)
    for c in txn.columns:
        if txn[c].dtype == object:
            txn[c] = txn[c].fillna("")
    txn.index = range(len(txn))
    return txn


def apply_exclusions(cfg: dict, txn: pd.DataFrame) -> pd.DataFrame:
    """Status and non-procurement exclusions + exact in-source re-exports. Cross-source duplicates come later."""
    status_specs = {s["name"]: s.get("status") or {} for s in cfg["sources"]}
    status_code = ex.apply_status(txn, status_specs, cfg.get("status_defaults", {}))
    npc = cfg.get("non_procurement") or {}
    rules = (list(npc.get("default_rules", [])) if npc.get("use_defaults", True) else []) + list(npc.get("rules", []))
    np_flag, np_rule = ex.apply_non_procurement(txn, rules)
    eligible = status_code.eq("") & ~np_flag
    dup_flag = ex.apply_within_source_duplicates(txn, eligible)
    txn["status_code"] = status_code
    txn["np_flag"] = np_flag
    txn["np_rule"] = np_rule
    txn["dup_in_source"] = dup_flag
    return txn


def finalise_inclusion(cfg: dict, txn: pd.DataFrame) -> pd.DataFrame:
    """Combine every exclusion signal into one reason per record following the configured precedence."""
    dup_cross = txn["match_status"].isin(CROSS_SOURCE_DUPLICATE_STATUSES)
    dup = txn["dup_in_source"] | dup_cross
    txn["exclusion_reason"] = ex.combine(txn, txn["status_code"], txn["np_flag"], dup,
                                         (cfg.get("exclusion") or {}).get("precedence", list(ex.EXCLUSION_CODES_DEFAULT)))
    txn["include"] = txn["exclusion_reason"].eq("").astype(int)
    return txn


def opaque_keys(keys: pd.Series, prefix: str, unknown_label: str) -> pd.Series:
    """Replace readable keys (supplier names, description fragments) by stable opaque hashes: grouping is identical,
    but no company text leaks into the public output file."""
    import hashlib
    uniq = keys.astype(str).unique()
    mapping = {u: (prefix + "UNKNOWN" if u == unknown_label else prefix + hashlib.blake2b(u.encode("utf-8"), digest_size=5).hexdigest())
               for u in uniq}
    return keys.astype(str).map(mapping)


def exclusion_confidence(txn: pd.DataFrame) -> pd.Series:
    """Confidence in the exclusion decision itself (used as the confidence of excluded rows)."""
    c = pd.Series(1.0, index=txn.index)
    c[txn["exclusion_reason"].eq("NON_PROCUREMENT")] = 0.95
    c[txn["exclusion_reason"].eq("DUPLICATE") & txn["dup_in_source"]] = 0.98
    cross = txn["exclusion_reason"].eq("DUPLICATE") & txn["match_status"].isin(CROSS_SOURCE_DUPLICATE_STATUSES)
    score = txn["match_score"].astype(float).fillna(0.9).clip(0.7, 1.0)
    c[cross] = np.where(txn.loc[cross, "match_status"].eq("EXPORT_DUPLICATE"), 0.9, score[cross])
    return c


def text_for_classification(txn: pd.DataFrame, ccfg: dict) -> tuple[list[str], list[str]]:
    """Main text = cleaned description (falls back to configured context fields when the description says nothing);
    context text = main text + extra context fields (used by the supervised models)."""
    desc = txn["description"].fillna("").astype(str).map(clean_description)
    fb_fields = [f for f in ccfg.get("fallback_fields", []) if f in txn.columns]
    ctx_fields = [f for f in ccfg.get("context_fields", []) if f in txn.columns]
    fallback = pd.Series("", index=txn.index)
    for f in fb_fields:
        fallback = (fallback + " " + txn[f].fillna("").astype(str).map(clean_description)).str.strip()
    weak = desc.map(lambda t: len([w for w in t.split() if len(w) > 2 and not w.isdigit()]) == 0)
    main = desc.where(~weak, (desc + " " + fallback).str.strip())
    ctx = main.copy()
    for f in ctx_fields:
        ctx = (ctx + " " + txn[f].fillna("").astype(str).map(clean_description)).str.strip()
    return main.tolist(), ctx.tolist()


def embedding_texts(sub: pd.DataFrame, main: list[str], ccfg: dict) -> list[str]:
    """Text given to the semantic priors: optional metadata prefix ('service : ...') built from configured fields."""
    fields = [f for f in ccfg.get("embed_prefix_fields", []) if f in sub.columns]
    if not fields:
        return main
    pre = sub[fields[0]].fillna("").astype(str).map(clean_description)
    for f in fields[1:]:
        pre = (pre + " " + sub[f].fillna("").astype(str).map(clean_description)).str.strip()
    return [(f"{p} : {t}" if p else t) for p, t in zip(pre.tolist(), main)]


def load_labels(cfg: dict, labels_dir: Path, tax) -> Optional[dict]:
    lcfg = cfg.get("labels")
    if not lcfg:
        return None
    path = labels_dir / lcfg["file"]
    if not path.exists():
        return None
    df = read_csv_text(path)
    cols = lcfg["level_columns"]
    idx = []
    for row in zip(*[df[c].astype(str).str.strip() for c in cols]):
        p = tuple(x for x in row if x != "")
        idx.append(tax.leaf_ids.get(p, -1) if p else -1)
    df["_leaf"] = idx
    return {"leaf": dict(zip(df[lcfg["id_column"]], df["_leaf"])), "frame": df, "cfg": lcfg,
            "unmapped": int(((df["_leaf"] < 0) & (df[cols[0]].astype(str).str.strip() != "")).sum())}


def run_company(config_path: str, data_dir: Optional[str] = None, labels_dir: Optional[str] = None,
                out_path: Optional[str] = None, zero_label: bool = False, evaluate: bool = True,
                work_dir: Optional[str] = None, report_dir: Optional[str] = None, seed: int = 0) -> dict:
    timer = StageTimer()
    t_all = time.perf_counter()
    cfg = load_and_validate(config_path)
    base = Path(cfg["_config_path"]).parent
    ddir = resolve_dir(cfg, "data_dir", data_dir)
    ldir = resolve_dir(cfg, "labels_dir", labels_dir) if (labels_dir or cfg.get("labels_dir")) else ddir
    cache_dir = Path(work_dir) if work_dir else (base.parent / "models" / "cache")
    tax = timer.run("taxonomy", load_taxonomy, ddir / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
    txn = timer.run("ingest_clean", ingest_and_clean, cfg, ddir)
    txn = timer.run("exclusions", apply_exclusions, cfg, txn)
    txn, match_diag = timer.run("transaction_matching", match_transactions, txn, cfg)
    txn = finalise_inclusion(cfg, txn)
    sup_key, sup_diag = timer.run("supplier_resolution", resolve_suppliers, txn, cfg)
    txn["supplier_key"] = inherit_from_primary(txn, sup_key).values
    item_key, item_diag = timer.run("item_resolution", resolve_items, txn, txn["supplier_key"], cfg)
    txn["item_key"] = inherit_from_primary(txn, item_key).values
    if (cfg.get("output") or {}).get("key_style", "hashed") == "hashed":
        txn["supplier_key"] = opaque_keys(txn["supplier_key"], "SUP-", "unknown supplier")
        txn["item_key"] = opaque_keys(txn["item_key"], "ITM-", "unspecified item")

    # ---- classification of the included rows ----------------------------------------------------------------------------
    ccfg = cfg.get("classification") or {}
    inc = txn.index[txn["include"] == 1]
    main_txt, ctx_txt = text_for_classification(txn.loc[inc], ccfg)
    train_info = None if zero_label else load_labels(cfg, ldir, tax)
    truth_info = load_labels(cfg, ldir, tax) if evaluate else None
    y_train = y_eval = None
    if train_info is not None:
        y_train = np.array([train_info["leaf"].get(r, -1) for r in txn.loc[inc, "record_id"]], dtype=int)
    if truth_info is not None:
        y_eval = np.array([truth_info["leaf"].get(r, -1) for r in txn.loc[inc, "record_id"]], dtype=int)
    groups = pd.factorize(txn.loc[inc, "item_key"].astype(str) + "|" + pd.Series(main_txt, index=inc))[0]
    emb_txt = embedding_texts(txn.loc[inc], main_txt, ccfg)
    share_fields = [f for f in ((ccfg.get("evidence_sharing") or {}).get("fields") or []) if f in txn.columns]
    share_groups = None
    if share_fields:  # rows without any supplier evidence must not form one giant group: each becomes its own group
        keys = txn.loc[inc, share_fields].astype(str).agg("|".join, axis=1)
        unknown = keys.str.contains("UNKNOWN", regex=False)
        keys = keys.where(~unknown, "row-" + pd.Series(np.arange(len(keys)), index=keys.index).astype(str))
        share_groups = pd.factorize(keys)[0]
    clf = HybridClassifier(tax, cfg, cache_dir=cache_dir, seed=seed)
    res = timer.run("classification", clf.classify, main_txt, ctx_txt, groups, y_train, zero_label, y_eval, emb_txt, share_groups)

    txn["confidence"] = exclusion_confidence(txn)
    for lvl in range(1, 5):
        txn[f"cat_l{lvl}"] = ""
    pads = [tax.padded(int(i)) for i in res.leaf]
    for lvl in range(tax.depth):
        txn.loc[inc, f"cat_l{lvl + 1}"] = [p[lvl] for p in pads]
    txn.loc[inc, "confidence"] = res.confidence
    txn["spend_final"] = np.where(txn["include"] == 1, txn["spend_calc"].astype(float), 0.0)
    txn["spend_final"] = txn["spend_final"].fillna(0.0)
    n_unconverted = int(((txn["include"] == 1) & txn["spend_calc"].isna()).sum())

    out = build_output(txn, tax.depth)
    problems = validate_output(out, txn["record_id"].tolist(), tax)
    if problems:
        raise RuntimeError("Output validation failed:\n - " + "\n - ".join(problems))
    company = cfg["company_id"]
    out_file = Path(out_path) if out_path else base.parent / "outputs" / f"aryan_sharma__{company}.csv"
    timer.run("write_output", write_output, out, out_file)

    # ---- diagnostics (aggregate only: no raw records) ---------------------------------------------------------------------
    diag = {
        "company": company, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": int(len(txn)), "mode": res.mode, "zero_label_forced": bool(zero_label),
        "reconciliation": reconcile_spend(txn, out), "included_with_unresolved_amount_or_fx": n_unconverted,
        "financial_flags": txn["fin_flag"].value_counts().to_dict(), "tax_flags": txn["tax_flag"].value_counts().to_dict(),
        "date_flags": {c: txn[c].value_counts().to_dict() for c in txn.columns if c.endswith("_flag") and "date" in c},
        "non_procurement_rules_hit": txn.loc[txn["np_flag"], "np_rule"].value_counts().to_dict(),
        "matching": match_diag, "suppliers": {k: v for k, v in sup_diag.items() if not k.startswith("_")}, "items": item_diag,
        "classification": res.diagnostics, "taxonomy_warnings": tax.warnings, "stage_seconds": timer.times,
    }
    if truth_info is not None:
        lcfg, lf = truth_info["cfg"], truth_info["frame"]
        for kind, col_key, ours in (("supplier", "supplier_key_column", "supplier_key"), ("item", "item_key_column", "item_key")):
            if lcfg.get(col_key) and lcfg[col_key] in lf.columns:
                gold = dict(zip(lf[lcfg["id_column"]], lf[lcfg[col_key]]))
                sub = txn[txn["record_id"].isin(gold)]
                diag[f"{kind}_resolution_vs_golden_keys"] = pairwise_scores_excluding_bucket(sub[ours], sub["record_id"].map(gold))
    if y_eval is not None and (y_eval >= 0).any():
        m = y_eval >= 0
        spend_inc = txn.loc[inc, "spend_final"].to_numpy()
        rep = cm.classification_report(y_eval[m], res.leaf[m], tax, spend_inc[m], res.confidence[m])
        rep["labels_used_for_prediction"] = bool(y_train is not None and not zero_label)
        rep.update(cm.ambiguity_slice([t for t, k in zip(main_txt, m) if k], y_eval[m], res.leaf[m]))
        diag["classification_vs_labels"] = rep
        if report_dir:
            rd = Path(report_dir) / "private"
            rd.mkdir(parents=True, exist_ok=True)
            if res.mode == "zero_label" and res.features:  # development artefact used to fit the transferable confidence model
                np.savez(rd / f"conf_features_{company}.npz", correct=(y_eval[m] == res.leaf[m]).astype(float),
                         **{k: v[m] for k, v in res.features.items()})
            cm.per_category_table(y_eval[m], res.leaf[m], tax).to_csv(rd / f"per_category_{company}_{res.mode}.csv", index=False)
            cm.confusion_top_pairs(y_eval[m], res.leaf[m], tax).to_csv(rd / f"top_confusions_{company}_{res.mode}.csv", index=False)
    diag["total_seconds"] = round(time.perf_counter() - t_all, 2)
    diag["output_file"] = str(out_file)
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(report_dir) / f"run_{company}_{res.mode}.json", "w", encoding="utf-8") as fh:
            json.dump(diag, fh, indent=2, default=str)
    diag["_txn"] = txn
    diag["_out"] = out
    diag["_res"] = res
    diag["_tax"] = tax
    return diag
