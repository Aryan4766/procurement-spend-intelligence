"""Classification metrics: exact leaf accuracy, per-level accuracy, macro / weighted F1, per-category table, confusion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.taxonomy.taxonomy_loader import Taxonomy


def level_accuracy(true_idx: np.ndarray, pred_idx: np.ndarray, tax: Taxonomy, weights: np.ndarray | None = None) -> dict:
    out = {}
    w = np.ones(len(true_idx)) if weights is None else weights
    for lvl in range(tax.depth):
        # a level-k match requires every shallower level to agree as well (a path prefix match)
        ok = np.array([tax.padded(int(a))[: lvl + 1] == tax.padded(int(b))[: lvl + 1] for a, b in zip(true_idx, pred_idx)])
        out[f"L{lvl + 1}_accuracy"] = float(ok.mean())
        out[f"L{lvl + 1}_accuracy_spend_weighted"] = float((ok * w).sum() / w.sum()) if w.sum() else None
    return out


def classification_report(true_idx: np.ndarray, pred_idx: np.ndarray, tax: Taxonomy, spend: np.ndarray | None = None,
                          confidence: np.ndarray | None = None) -> dict:
    true_idx, pred_idx = np.asarray(true_idx), np.asarray(pred_idx)
    rep = {
        "n": int(len(true_idx)),
        "exact_leaf_accuracy": float((true_idx == pred_idx).mean()),
        "macro_f1": float(f1_score(true_idx, pred_idx, average="macro", labels=np.unique(true_idx), zero_division=0)),
        "weighted_f1": float(f1_score(true_idx, pred_idx, average="weighted", labels=np.unique(true_idx), zero_division=0)),
    }
    w = None if spend is None else np.abs(spend)
    rep.update(level_accuracy(true_idx, pred_idx, tax, w))
    if w is not None and w.sum() > 0:
        rep["exact_leaf_accuracy_spend_weighted"] = float(((true_idx == pred_idx) * w).sum() / w.sum())
    if confidence is not None:
        correct = (true_idx == pred_idx).astype(float)
        rep["mean_confidence"] = float(np.mean(confidence))
        rep["expected_calibration_error"] = expected_calibration_error(confidence, correct)
        hi = confidence >= 0.8
        rep["share_conf_ge_0.8"] = float(hi.mean())
        rep["accuracy_conf_ge_0.8"] = float(correct[hi].mean()) if hi.any() else None
        rep["accuracy_conf_lt_0.5"] = float(correct[confidence < 0.5].mean()) if (confidence < 0.5).any() else None
    return rep


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(conf)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.any():
            ece += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def per_category_table(true_idx: np.ndarray, pred_idx: np.ndarray, tax: Taxonomy) -> pd.DataFrame:
    rows = []
    for c in np.unique(true_idx):
        t, p = true_idx == c, pred_idx == c
        tp = int((t & p).sum())
        prec = tp / p.sum() if p.sum() else 0.0
        rec = tp / t.sum() if t.sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"category": tax.path_text(int(c)), "support": int(t.sum()), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def confusion_top_pairs(true_idx: np.ndarray, pred_idx: np.ndarray, tax: Taxonomy, top: int = 15) -> pd.DataFrame:
    m = true_idx != pred_idx
    df = pd.DataFrame({"true": [tax.path_text(int(i)) for i in true_idx[m]], "pred": [tax.path_text(int(i)) for i in pred_idx[m]]})
    return df.value_counts().head(top).reset_index(name="count")


def ambiguity_slice(texts: list[str], true_idx: np.ndarray, pred_idx: np.ndarray, max_tokens: int = 2) -> dict:
    """Performance on very short / generic descriptions (<= max_tokens informative tokens) versus the rest."""
    short = np.array([len([w for w in t.split() if len(w) > 2 and not w.isdigit()]) <= max_tokens for t in texts])
    out = {"short_descriptions_share": float(short.mean())}
    if short.any():
        out["short_accuracy"] = float((true_idx[short] == pred_idx[short]).mean())
    if (~short).any():
        out["long_accuracy"] = float((true_idx[~short] == pred_idx[~short]).mean())
    return out
