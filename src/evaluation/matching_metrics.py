"""Entity-resolution metrics against golden keys: pairwise precision / recall and cluster purity / completeness."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pairs(x: np.ndarray) -> float:
    return float((x * (x - 1) // 2).sum())


def pairwise_scores(pred_key: pd.Series, true_key: pd.Series) -> dict:
    """Pair-counting precision/recall/F1 between two clusterings of the same rows (rows with blank truth are skipped)."""
    df = pd.DataFrame({"p": pred_key.values, "t": true_key.values})
    df = df[df["t"].astype(str) != ""]
    if df.empty:
        return {}
    both = df.groupby(["p", "t"]).size().to_numpy()
    pa = df.groupby("p").size().to_numpy()
    ta = df.groupby("t").size().to_numpy()
    tp, pp, tt = _pairs(both), _pairs(pa), _pairs(ta)
    prec = tp / pp if pp else 1.0
    rec = tp / tt if tt else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ct = pd.crosstab(df["p"], df["t"])
    return {
        "rows": int(len(df)), "pairwise_precision": round(prec, 4), "pairwise_recall": round(rec, 4), "pairwise_f1": round(f1, 4),
        "cluster_purity": round(float(ct.max(axis=1).sum() / ct.values.sum()), 4),
        "cluster_completeness": round(float(ct.max(axis=0).sum() / ct.values.sum()), 4),
        "predicted_groups": int(df["p"].nunique()), "true_groups": int(df["t"].nunique()),
    }


def pairwise_scores_excluding_bucket(pred_key: pd.Series, true_key: pd.Series, bucket_share: float = 0.1) -> dict:
    """Same scores, but without the golden catch-all bucket (a golden key covering > bucket_share of the labelled rows
    is a deliberate merge of unrelated entities and cannot be reconstructed from the data)."""
    t = true_key.astype(str)
    t = t[t != ""]
    top = t.value_counts()
    out = {"all_rows": pairwise_scores(pred_key, true_key)}
    if len(top) and top.iloc[0] / len(t) > bucket_share:
        keep = true_key.astype(str) != top.index[0]
        out["without_catch_all_bucket"] = pairwise_scores(pred_key[keep], true_key[keep])
        out["catch_all_bucket_share_of_labelled_rows"] = round(float(top.iloc[0] / len(t)), 4)
    return out
