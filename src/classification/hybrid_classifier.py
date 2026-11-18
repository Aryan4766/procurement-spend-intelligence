"""Hybrid classifier: semantic zero-label prior + (optional) supervised models, hierarchical decoding, calibrated confidence.

Modes (chosen at run time from the data, never from the company's identity):
  zero_label   no labels (or --zero-label): ensemble semantic prior + label-free self-training on description+context
  supervised   labels available: TF-IDF / embedding linear models trained on them, blended with the prior with a weight
               chosen on cross-fitted (out-of-fold) predictions - no line is ever predicted by a model that saw it
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.classification import zero_label_classifier as zl
from src.classification.semantic_classifier import Embedder, EmbeddingUnavailable
from src.classification.text_classifier import TfidfLinearClassifier
from src.taxonomy.taxonomy_loader import Taxonomy

DEFAULT_MODELS = [  # chosen on development data (DECISIONS.md): multilingual e5 (base+small) + two English encoders
    {"name": "intfloat/multilingual-e5-base", "query_prefix": "query: ", "doc_prefix": "passage: "},
    {"name": "intfloat/multilingual-e5-small", "query_prefix": "query: ", "doc_prefix": "passage: "},
    {"name": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "BAAI/bge-small-en-v1.5"},
]

# Confidence model for runs without labels: logistic regression on [top prob, margin, signal agreement,
# text informativeness]. The coefficients are fitted on the development companies' hidden-label runs
# (configs/common.yaml: classification.confidence_coefficients) and reused on a company that has no labels.
DEFAULT_CONF_COEFS = {"intercept": -1.2, "top": 2.0, "margin": 2.0, "agree": 0.8, "info": 0.6}


@dataclass
class ClassificationResult:
    proba: np.ndarray
    leaf: np.ndarray
    confidence: np.ndarray
    mode: str
    diagnostics: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)


def decode_hierarchical(P: np.ndarray, tax: Taxonomy) -> np.ndarray:
    """Top-down decoding: pick the best level-1 branch by summed leaf probability, then the best branch below it, ...
    The result is always a leaf of the supplied tree, so impossible parent/child combinations cannot occur."""
    n = P.shape[0]
    allowed = np.ones((n, tax.n_leaves), dtype=bool)
    for lvl in range(tax.depth - 1):
        pref = tax.prefix_index(lvl)
        keys = list(pref.keys())
        masked = np.where(allowed, P, 0.0)
        marg = np.stack([masked[:, idx].sum(1) for idx in pref.values()], axis=1)
        best = marg.argmax(1)
        new_allowed = np.zeros_like(allowed)
        for j, k in enumerate(keys):
            rows = best == j
            if rows.any():
                new_allowed[np.ix_(rows, pref[k])] = True
        # leaves shorter than this level stay allowed only if their branch was chosen (they are inside pref[k])
        allowed &= new_allowed
    return np.where(allowed, P, -1.0).argmax(1)


def confidence_features(P: np.ndarray, leaf: np.ndarray, agree: np.ndarray, info: np.ndarray) -> dict:
    srt = np.sort(P, axis=1)
    return {"top": P[np.arange(len(P)), leaf], "margin": srt[:, -1] - srt[:, -2], "agree": agree, "info": info}


def apply_confidence(feats: dict, coefs: dict) -> np.ndarray:
    z = coefs["intercept"] + sum(coefs[k] * feats[k] for k in ("top", "margin", "agree", "info"))
    return 1.0 / (1.0 + np.exp(-z))


def informativeness(texts: list[str]) -> np.ndarray:
    """0 for blank / numeric-only text, up to 1 for a description with several informative tokens."""
    out = np.zeros(len(texts))
    for i, t in enumerate(texts):
        toks = [w for w in t.split() if len(w) > 2 and not w.isdigit()]
        out[i] = min(len(toks), 4) / 4.0
    return out


class HybridClassifier:
    def __init__(self, taxonomy: Taxonomy, cfg: dict, cache_dir: Optional[Path] = None, seed: int = 0):
        self.tax = taxonomy
        self.ccfg = cfg.get("classification") or {}
        self.seed = seed
        self.cache_dir = cache_dir
        self.models = self.ccfg.get("embedding_models", DEFAULT_MODELS)
        if isinstance(self.models, str):
            self.models = [{"name": self.models}]
        self.embedding_ok = True

    # ------------------------------------------------------------------------------------------------------------
    def _semantic_scores(self, uniq: list[str]):
        """Z-scored mean of cosine-similarity matrices over the ensemble. Returns (S, primary_embeddings, used_models)."""
        weights = self.ccfg.get("label_text_weights", {"path": 0.6, "last2": 0.4})
        lt = zl.label_texts(self.tax)
        mats, primary, used = [], None, []
        for spec in self.models:
            name = spec["name"] if isinstance(spec, dict) else spec
            if str(name).lower() in ("none", "lexical", ""):
                continue
            try:
                emb = Embedder(name, cache_dir=self.cache_dir, query_prefix=spec.get("query_prefix", "") if isinstance(spec, dict) else "",
                               doc_prefix=spec.get("doc_prefix", "") if isinstance(spec, dict) else "")
                X = emb.encode(uniq, kind="query")
                L = {k: emb.encode(v, kind="doc") for k, v in lt.items()}
                emb.save()
            except EmbeddingUnavailable:
                continue
            S = zl.semantic_scores(X, L, weights)
            mats.append((S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9))
            used.append(name)
            if primary is None:
                primary = X
        if not mats:
            self.embedding_ok = False
            return self._lexical_scores(uniq), None, []
        return np.mean(mats, axis=0), primary, used

    def _lexical_scores(self, texts: list[str]) -> np.ndarray:
        """Fallback when no embedding model is available: char-n-gram TF-IDF cosine against the taxonomy text, z-scored."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        lt = zl.label_texts(self.tax)["path"]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts + lt)
        S = (vec.transform(texts) @ vec.transform(lt).T).toarray()
        return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)

    # ------------------------------------------------------------------------------------------------------------
    def classify(self, texts: list[str], ctx_texts: list[str], groups: np.ndarray, labels: Optional[np.ndarray] = None,
                 zero_label: bool = False, eval_labels: Optional[np.ndarray] = None,
                 embed_texts: Optional[list] = None, share_groups: Optional[np.ndarray] = None) -> ClassificationResult:
        """texts: main description text per row; ctx_texts: description + context (supplier, account...) per row;
        groups: item key per row (consistency + leakage-free folds); labels: leaf index or -1 (ignored in zero_label mode);
        eval_labels: hidden leaf indices used ONLY after prediction, for a diagnostic accuracy."""
        t0 = time.perf_counter()
        n, L = len(texts), self.tax.n_leaves
        diag: dict = {"rows": n, "leaves": L}
        emb_src = embed_texts if embed_texts is not None else texts  # metadata-prefixed text for the priors (optional)
        uniq, inv = np.unique(np.array(emb_src, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        S_u, X_primary, used = self._semantic_scores(uniq)
        diag.update({"embedding_models_used": used, "embedding_available": bool(used)})
        temperature = float(self.ccfg.get("temperature", 0.5))
        P_prior = zl.softmax(S_u, temperature)[inv]
        info = informativeness(texts)
        blank = np.array([len(t.strip()) == 0 for t in texts])
        comp: dict = {"prior": P_prior}

        supervised = labels is not None and not zero_label and int((labels >= 0).sum()) >= int(self.ccfg.get("min_labels", 200))
        if supervised:
            P, sup_diag, comp_s = self._supervised(ctx_texts, groups, labels, X_primary, inv, P_prior)
            diag.update(sup_diag)
            comp.update(comp_s)
            mode = "supervised+semantic"
            agree = (comp["sup"].argmax(1) == comp["prior"].argmax(1)).astype(float)
        else:
            P = P_prior
            agree = np.ones(n)
            if self.ccfg.get("self_train", True):
                # Two independent priors, each refined by label-free self-training on description + context
                # (supplier, account...) and then averaged: the lexical prior is high-precision on lines that share words with
                # the category names, the embedding prior covers paraphrases; measured best on both development companies.
                u_ctx, first, inv_c, cnt = np.unique(np.array(ctx_texts, dtype=object), return_index=True, return_inverse=True, return_counts=True)
                kw = dict(rounds=int(self.ccfg.get("self_train_rounds", 3)), top_frac=float(self.ccfg.get("self_train_top_frac", 0.5)),
                          blend=float(self.ccfg.get("self_train_blend", 0.7)), weights=cnt.astype(float), seed=self.seed)
                S_lex = self._lexical_scores(uniq)
                P_lex = zl.softmax(S_lex, temperature)[inv]
                st_lex = zl.self_train(u_ctx.tolist(), P_lex[first], L, **kw)[inv_c]
                if used:
                    st_emb = zl.self_train(u_ctx.tolist(), P_prior[first], L, **kw)[inv_c]
                    P = 0.5 * st_lex + 0.5 * st_emb
                    agree = (st_lex.argmax(1) == st_emb.argmax(1)).astype(float)
                else:
                    P = st_lex
                comp["self_trained"] = P
            mode = "zero_label"
        share = self.ccfg.get("evidence_sharing") or {}
        if not supervised and share_groups is not None and float(share.get("strength", 0.0)) > 0:
            # lines that structurally belong together (same supplier...) vote on the PARENT branch only; the leaf inside the
            # branch stays the line's own (full-leaf sharing hurt B: -16 pt; parent-only helped both A and B)
            P = zl.share_parent_evidence(P, share_groups, self.tax, float(share["strength"]), level=0)
            diag["evidence_sharing"] = {"fields": share.get("fields"), "strength": share["strength"]}
        if float(self.ccfg.get("item_smoothing", 0.0)) > 0 and groups is not None:
            P = zl.smooth_by_group(P, groups, float(self.ccfg["item_smoothing"]))
        leaf = decode_hierarchical(P, self.tax) if self.ccfg.get("hierarchical_decode", True) else P.argmax(1)
        if blank.any():  # no text at all: most frequent predicted leaf, flagged by a near-zero confidence below
            fallback = np.bincount(leaf[~blank], minlength=L).argmax() if (~blank).any() else 0
            leaf = np.where(blank, fallback, leaf)
        feats = confidence_features(P, leaf, agree, info)
        coefs = {**DEFAULT_CONF_COEFS, **(self.ccfg.get("confidence_coefficients") or {})}
        conf = apply_confidence(feats, coefs)
        conf = np.where(blank, np.minimum(conf, 0.05), conf)
        if supervised:
            conf = self._calibrate(conf, leaf, labels, diag)
        diag["seconds"] = round(time.perf_counter() - t0, 2)
        if eval_labels is not None and (eval_labels >= 0).any():
            m = eval_labels >= 0
            diag["diagnostic_leaf_accuracy_vs_hidden_labels"] = float((leaf[m] == eval_labels[m]).mean())
        return ClassificationResult(P, leaf, conf, mode, diag, comp, feats)

    # ------------------------------------------------------------------------------------------------------------
    class _EmbLR:
        def __init__(self, clf, classes, L):
            self.clf, self.classes, self.L = clf, classes, L

        def predict(self, X):
            out = np.zeros((len(X), self.L))
            if self.clf is None:
                out[:, self.classes[0]] = 1.0
            else:
                out[:, self.clf.classes_] = self.clf.predict_proba(X)
            return out

    def _emb_lr(self, X, y, L):
        classes = np.unique(y)
        if len(classes) == 1:
            return self._EmbLR(None, classes, L)
        clf = LogisticRegression(C=float(self.ccfg.get("embedding_lr_C", 10.0)), max_iter=300, random_state=self.seed)
        clf.fit(X, y)
        return self._EmbLR(clf, classes, L)

    def _supervised(self, ctx_texts, groups, labels, X_primary, inv, P_prior):
        n, L = len(ctx_texts), self.tax.n_leaves
        lab_idx = np.nonzero(labels >= 0)[0]
        y = labels
        folds = min(int(self.ccfg.get("cv_folds", 5)), max(2, len(np.unique(groups[lab_idx]))))
        gkf = GroupKFold(n_splits=folds)
        P_tfidf, P_emb = np.zeros((n, L)), np.zeros((n, L))
        X_rows = X_primary[inv] if X_primary is not None else None
        for tr, va in gkf.split(lab_idx, groups=groups[lab_idx]):
            tr_i, va_i = lab_idx[tr], lab_idx[va]
            clf = TfidfLinearClassifier(L, seed=self.seed).fit([ctx_texts[i] for i in tr_i], y[tr_i])
            P_tfidf[va_i] = clf.predict_proba([ctx_texts[i] for i in va_i])
            if X_rows is not None:
                P_emb[va_i] = self._emb_lr(X_rows[tr_i], y[tr_i], L).predict(X_rows[va_i])
        rest = np.nonzero(labels < 0)[0]
        if len(rest):  # models trained on all labels predict the lines that have none
            clf = TfidfLinearClassifier(L, seed=self.seed).fit([ctx_texts[i] for i in lab_idx], y[lab_idx])
            P_tfidf[rest] = clf.predict_proba([ctx_texts[i] for i in rest])
            if X_rows is not None:
                P_emb[rest] = self._emb_lr(X_rows[lab_idx], y[lab_idx], L).predict(X_rows[rest])
        P_sup = P_tfidf if X_rows is None else 0.5 * P_tfidf + 0.5 * P_emb
        best_w, best_ll = 1.0, 1e9
        for w in np.linspace(0, 1, 11):
            Pb = (1 - w) * P_prior + w * P_sup
            ll = -np.log(np.clip(Pb[lab_idx, y[lab_idx]], 1e-9, 1)).mean()
            if ll < best_ll:
                best_ll, best_w = ll, float(w)
        P = (1 - best_w) * P_prior + best_w * P_sup
        d = {"supervised_weight": best_w, "labelled_rows": int(len(lab_idx)), "cv_folds": folds,
             "oof_accuracy_supervised_only": float((P_sup[lab_idx].argmax(1) == y[lab_idx]).mean()),
             "oof_accuracy_prior_only": float((P_prior[lab_idx].argmax(1) == y[lab_idx]).mean()),
             "oof_accuracy_blend": float((P[lab_idx].argmax(1) == y[lab_idx]).mean())}
        return P, d, {"sup": P_sup, "tfidf": P_tfidf, "emb": P_emb}

    def _calibrate(self, conf, leaf, labels, diag):
        """Isotonic calibration of the confidence on out-of-fold correctness (labelled rows only)."""
        from sklearn.isotonic import IsotonicRegression
        m = labels >= 0
        correct = (leaf[m] == labels[m]).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(conf[m], correct)
        diag["confidence_calibration"] = "isotonic regression on out-of-fold correctness"
        return iso.predict(conf)
