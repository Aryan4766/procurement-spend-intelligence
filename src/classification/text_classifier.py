"""TF-IDF (word + character n-grams) with a linear classifier. Used for supervised training and for the
label-free self-training step. Character n-grams make it robust to OCR noise, abbreviations and many languages."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier


class TfidfLinearClassifier:
    def __init__(self, n_classes: int, C: float = 20.0, max_features: int = 300000, seed: int = 0, kind: str = "logreg"):
        """kind='logreg' (multinomial, accurate; supervised mode) or 'sgd' (fast one-vs-rest log-loss; self-training)."""
        self.n_classes = n_classes
        self.C = C
        self.seed = seed
        self.kind = kind
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w[\w.\-/]*\b", max_features=max_features)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                    max_features=max_features)
        self.clf = None
        self.classes_: np.ndarray | None = None

    def _fit_transform(self, texts):
        return sp.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return sp.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts: list[str], y: np.ndarray, sample_weight: np.ndarray | None = None,
            extra: sp.spmatrix | None = None) -> "TfidfLinearClassifier":
        X = self._fit_transform(texts)
        if extra is not None:
            X = sp.hstack([X, extra]).tocsr()
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            self.clf = None
            return self
        if self.kind == "sgd":
            self.clf = SGDClassifier(loss="log_loss", alpha=1.0 / (self.C * max(1, X.shape[0])) * 8, max_iter=25, tol=None,
                                     random_state=self.seed)
        else:
            self.clf = LogisticRegression(C=self.C, max_iter=400, solver="lbfgs", random_state=self.seed)
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, texts: list[str], extra: sp.spmatrix | None = None) -> np.ndarray:
        out = np.zeros((len(texts), self.n_classes), dtype=np.float64)
        if self.clf is None:
            out[:, self.classes_[0]] = 1.0
            return out
        X = self.transform(texts)
        if extra is not None:
            X = sp.hstack([X, extra]).tocsr()
        P = self.clf.predict_proba(X)
        P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
        out[:, self.clf.classes_] = P
        return out
