"""Fit the transferable zero-label confidence model on development companies' hidden-label runs.

Usage: python -m src.evaluation.fit_confidence reports/private/conf_features_A.npz reports/private/conf_features_B.npz
Prints the coefficients to paste into configs/common.yaml (classification.confidence_coefficients).
Companies are pooled with equal weight so the larger company does not dominate.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation.classification_metrics import expected_calibration_error

FEATURES = ("top", "margin", "agree", "info")


def fit(paths: list[str]) -> dict:
    Xs, ys, ws = [], [], []
    for p in paths:
        z = np.load(p)
        X = np.stack([z[f] for f in FEATURES], axis=1)
        Xs.append(X)
        ys.append(z["correct"])
        ws.append(np.full(len(X), 1.0 / len(X)))
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    lr = LogisticRegression(C=10.0, max_iter=1000).fit(X, y, sample_weight=w * len(w))
    coefs = {"intercept": float(lr.intercept_[0]), **{f: float(c) for f, c in zip(FEATURES, lr.coef_[0])}}
    if len(paths) > 1:  # leave-one-company-out: fit on the others, score the held-out company (honest transfer estimate)
        for k, p in enumerate(paths):
            tr = [i for i in range(len(paths)) if i != k]
            Xt, yt = np.vstack([Xs[i] for i in tr]), np.concatenate([ys[i] for i in tr])
            wt = np.concatenate([np.full(len(Xs[i]), 1.0 / len(Xs[i])) for i in tr])
            m = LogisticRegression(C=10.0, max_iter=1000).fit(Xt, yt, sample_weight=wt * len(wt))
            conf = m.predict_proba(Xs[k])[:, 1]
            print(f"leave-one-out {p}: accuracy={ys[k].mean():.3f} mean_conf={conf.mean():.3f} ECE={expected_calibration_error(conf, ys[k]):.3f}")
    for p, Xi, yi in zip(paths, Xs, ys):
        conf = lr.predict_proba(Xi)[:, 1]
        auc = roc_auc_score(yi, conf) if 0 < yi.mean() < 1 else float("nan")
        print(f"{p}: n={len(yi)} accuracy={yi.mean():.3f} mean_conf={conf.mean():.3f} ECE={expected_calibration_error(conf, yi):.3f} ranking AUC={auc:.3f}")
        edges = np.linspace(0, 1, 6)
        rel = []
        for a, b in zip(edges[:-1], edges[1:]):
            mk = (conf >= a) & (conf <= b if b == 1 else conf < b)
            if mk.any():
                rel.append(f"[{a:.1f},{b:.1f}): n={int(mk.sum())} conf={conf[mk].mean():.2f} acc={yi[mk].mean():.2f}")
        print("   reliability: " + " | ".join(rel))
    return coefs


if __name__ == "__main__":
    c = fit(sys.argv[1:])
    print("confidence_coefficients:", {k: round(v, 3) for k, v in c.items()})
