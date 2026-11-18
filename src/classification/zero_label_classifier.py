"""Zero-label classification: place a description into a taxonomy it has never seen, using only the taxonomy text.

Stage 1  (semantic prior)   cosine similarity between the description embedding and the embedding of every leaf
                            path ("L1 > L2 > L3"), blended with similarities to the individual level names; turned
                            into a probability with a softmax whose temperature is fixed (not tuned on labels).
Stage 2  (self-training)    the confident part of stage 1 is used as pseudo-labels to fit a TF-IDF linear model on the
                            company's OWN descriptions. It learns that 'RODANCHR' and 'ROD ANCHR' behave like the
                            words the embedding model understood. Blended back with the prior. No external labels.
Stage 3  (consistency)      lines of the same item share evidence (mean log-probability over the item cluster).
"""
from __future__ import annotations

import numpy as np

from src.classification.text_classifier import TfidfLinearClassifier
from src.taxonomy.taxonomy_loader import Taxonomy


def softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = S / T
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def label_texts(tax: Taxonomy) -> dict[str, list[str]]:
    """Texts derived ONLY from the taxonomy, for each leaf."""
    path = [" > ".join(p) for p in tax.paths]
    last = [p[-1] for p in tax.paths]
    last2 = [(p[-2] + " " + p[-1]) if len(p) > 1 and p[-2] != p[-1] else p[-1] for p in tax.paths]
    return {"path": path, "last": last, "last2": last2}


def semantic_scores(X: np.ndarray, label_vecs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    S = 0.0
    tot = sum(weights.values())
    for k, w in weights.items():
        S = S + (w / tot) * (X @ label_vecs[k].T)
    return S


def self_train(texts: list[str], P0: np.ndarray, n_classes: int, rounds: int = 2, top_frac: float = 0.6,
               blend: float = 0.5, weights: np.ndarray | None = None, seed: int = 0, min_per_class: int = 3) -> np.ndarray:
    """Transductive refinement using only the model's own confident predictions as pseudo-labels."""
    P = P0.copy()
    n = len(texts)
    w_rows = np.ones(n) if weights is None else weights
    for _ in range(rounds):
        pred = P.argmax(1)
        srt = np.sort(P, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        score = margin
        chosen = np.zeros(n, dtype=bool)
        # class-balanced selection: top fraction per predicted class so rare classes are not starved
        for c in np.unique(pred):
            idx = np.nonzero(pred == c)[0]
            k = max(min_per_class, int(np.ceil(top_frac * len(idx))))
            chosen[idx[np.argsort(-score[idx])[:k]]] = True
        clf = TfidfLinearClassifier(n_classes, C=10.0, seed=seed, kind="sgd")
        clf.fit([texts[i] for i in np.nonzero(chosen)[0]], pred[chosen], sample_weight=w_rows[chosen])
        Pt = clf.predict_proba(texts)
        P = (1 - blend) * P0 + blend * Pt
        P = P / P.sum(axis=1, keepdims=True)
    return P


def smooth_by_group(P: np.ndarray, groups: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Mix each row's log-probability with the mean log-probability of its group (same item / same description)."""
    L = np.log(np.clip(P, 1e-9, 1.0))
    out = L.copy()
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) > 1:
            out[idx] = (1 - strength) * L[idx] + strength * L[idx].mean(axis=0, keepdims=True)
    Z = np.exp(out - out.max(axis=1, keepdims=True))
    return Z / Z.sum(axis=1, keepdims=True)


def share_parent_evidence(P: np.ndarray, groups: np.ndarray, tax, strength: float = 1.0, level: int = 0) -> np.ndarray:
    """Group-level evidence for the PARENT only: rows of one group (e.g. one supplier) vote on the level-`level` branch with
    their mean branch marginals; the leaf choice inside the branch stays the row's own. P' ~ P * M_group[branch(leaf)]^strength."""
    pref = tax.prefix_index(level)
    branch_of_leaf = np.zeros(tax.n_leaves, dtype=int)
    cols = []
    for j, (_, idx) in enumerate(pref.items()):
        branch_of_leaf[idx] = j
        cols.append(P[:, idx].sum(1))
    marg = np.stack(cols, axis=1)  # rows x branches
    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    gmean = np.zeros_like(marg)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        gmean[idx] = marg[idx].mean(axis=0, keepdims=True)
    out = P * np.power(np.clip(gmean[:, branch_of_leaf], 1e-6, 1.0), strength)
    return out / out.sum(axis=1, keepdims=True)
