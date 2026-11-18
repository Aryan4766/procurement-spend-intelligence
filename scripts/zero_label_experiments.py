"""Reproducible zero-label experiment harness (development use).

Every variant sees identical rows and is scored against the hidden labels AFTER prediction only.
Usage: python scripts/zero_label_experiments.py A|B [--proxy]   (appends a table to reports/optimization_experiments.md)
--proxy uses only e5-small + MiniLM (fast, for text-variant experiments that need re-embedding).
"""
import src  # noqa: F401  (torch first on Windows)
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import zero_label_classifier as zl
from src.classification.hybrid_classifier import DEFAULT_MODELS, HybridClassifier, decode_hierarchical
from src.config.config_loader import load_and_validate
from src.evaluation.classification_metrics import classification_report
from src.matching.supplier_resolver import inherit_from_primary, resolve_suppliers
from src.matching.transaction_matcher import match_transactions
from src.pipeline import apply_exclusions, finalise_inclusion, ingest_and_clean, load_labels, text_for_classification
from src.taxonomy.taxonomy_loader import load_taxonomy

CO = sys.argv[1].upper()
PROXY = "--proxy" in sys.argv
cfg = load_and_validate(f"configs/company_{CO.lower()}.yaml")
ddir = Path(f"../company_{CO}").resolve()
tax = load_taxonomy(ddir / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
txn = ingest_and_clean(cfg, ddir)
txn = apply_exclusions(cfg, txn)
txn, _ = match_transactions(txn, cfg)
txn = finalise_inclusion(cfg, txn)
sk, _ = resolve_suppliers(txn, cfg)
txn["supplier_key"] = inherit_from_primary(txn, sk).values
inc = txn.index[txn.include == 1]
sub = txn.loc[inc]
main, ctx = text_for_classification(sub, cfg["classification"])
lab = load_labels(cfg, ddir, tax)
y = np.array([lab["leaf"].get(r, -1) for r in sub["record_id"]])
spend = sub["spend_calc"].astype(float).fillna(0).values
m = y >= 0
models = [d for d in DEFAULT_MODELS if "e5-base" not in d["name"] and "bge" not in d["name"]] if PROXY else DEFAULT_MODELS
cfg["classification"]["embedding_models"] = models
clf = HybridClassifier(tax, cfg, cache_dir=Path("models/cache"))
ROWS = []


def prepare(texts):
    uniq, inv = np.unique(np.array(texts, dtype=object), return_inverse=True)
    S_emb, _, _ = clf._semantic_scores(uniq.tolist())
    S_lex = clf._lexical_scores(uniq.tolist())
    return zl.softmax(S_emb, 0.5)[inv], zl.softmax(S_lex, 0.5)[inv]


def selftrain(P_prior, texts, rounds=3):
    u, first, inv_c, cnt = np.unique(np.array(texts, dtype=object), return_index=True, return_inverse=True, return_counts=True)
    return zl.self_train(u.tolist(), P_prior[first], tax.n_leaves, rounds=rounds, top_frac=0.5, blend=0.7, weights=cnt.astype(float))[inv_c]


def group_ids(fields):
    cols = [sub[f].map(str).tolist() if f in sub.columns else [""] * len(sub) for f in fields]
    return pd.factorize(pd.Series(["|".join(t) for t in zip(*cols)]))[0]


def record(name, P, secs, note=""):
    leaf = decode_hierarchical(P, tax)
    r = classification_report(y[m], leaf[m], tax, spend[m])
    ROWS.append(dict(experiment=name, leaf=r["exact_leaf_accuracy"], L1=r["L1_accuracy"], L2=r["L2_accuracy"],
                     L3=r.get("L3_accuracy"), leaf_spend_w=r["exact_leaf_accuracy_spend_weighted"], L1_spend_w=r["L1_accuracy_spend_weighted"],
                     macro_f1=r["macro_f1"], seconds=round(secs, 1), note=note))
    print(ROWS[-1], flush=True)


def baseline(texts, ctxs):
    Pe, Pl = prepare(texts)
    return 0.5 * selftrain(Pl, ctxs) + 0.5 * selftrain(Pe, ctxs), Pe, Pl


t0 = time.time()
P0, Pe, Pl = baseline(main, ctx)
record("E1 baseline (lexical+embedding self-trained)" + (" [proxy ensemble]" if PROXY else ""), P0, time.time() - t0)
# E-group: evidence sharing between lines that structurally belong together (label-free)
for fields, strength in [(["supplier_key"], 0.5), (["supplier_key"], 0.8), (["supplier_key"], 1.0),
                         (["supplier_key", "ctx_spend_type"], 0.8), (["ctx_contract_number"], 0.8)]:
    if fields[-1] not in sub.columns and fields[-1] != "supplier_key":
        continue
    t0 = time.time()
    P = zl.smooth_by_group(P0, group_ids(fields), strength)
    record(f"E6 evidence sharing by {'+'.join(fields)} s={strength}", P, time.time() - t0)
# smoothing the PRIORS before self-training
for fields, strength in [(["supplier_key", "ctx_spend_type"], 0.8)]:
    t0 = time.time()
    g = group_ids(fields)
    Pe2, Pl2 = zl.smooth_by_group(Pe, g, strength), zl.smooth_by_group(Pl, g, strength)
    P = 0.5 * selftrain(Pl2, ctx) + 0.5 * selftrain(Pe2, ctx)
    record(f"E6b smooth priors by {'+'.join(fields)} s={strength} then self-train", P, time.time() - t0)
    P = zl.smooth_by_group(P, g, strength)
    record(f"E6c = E6b + smooth result", P, time.time() - t0)

# ---- text-representation experiments (re-embed; use --proxy for speed) --------------------------------------------------------
import re
def head_phrase(t):
    return re.split(r"[;,]", t, maxsplit=1)[0].strip() if t else t
def no_codes(t):
    return " ".join(w for w in t.split() if not (re.search(r"\d", w) and re.search(r"[a-z]", w) and len(w) > 5))
variants = {
    "E2 drop alphanumeric part codes": [no_codes(t) for t in main],
    "E3 head noun phrase repeated (head + full text)": [(head_phrase(t) + " . " + t).strip() for t in main],
    "E5 metadata prefix (spend type + material type)": [((a + " " + b + " : " + t).strip()) for t, a, b in zip(main, sub["ctx_spend_type"].map(str).str.lower(), sub["ctx_material_type"].map(str).str.lower())] if "ctx_spend_type" in sub.columns else main,
}
g_sup = group_ids(["supplier_key"])
for name, texts in variants.items():
    t0 = time.time()
    ctx_v = [t + c[len(m_):] if c.startswith(m_) else c for t, c, m_ in zip(texts, ctx, main)]
    P, _, _ = baseline(texts, ctx_v)
    record(name, P, time.time() - t0)
    record(name + " + supplier sharing 0.8", zl.smooth_by_group(P, g_sup, 0.8), 0.0)
pd.DataFrame(ROWS).to_csv(f"reports/private/experiments_{CO}{'_proxy' if PROXY else ''}.csv", index=False)
