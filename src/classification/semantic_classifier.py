"""Local sentence-embedding wrapper with an on-disk cache, plus cosine-similarity retrieval over taxonomy paths.

torch is imported first on purpose: on Windows, importing numpy/pandas before torch can break torch's DLL loading.
If the embedding model cannot be loaded the caller falls back to a lexical (TF-IDF) similarity - the pipeline
never depends on network access at run time once the model is cached.
"""
from __future__ import annotations

try:  # noqa: SIM105 - must precede numpy / pandas on Windows
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch missing or broken: lexical fallback is used
    torch = None

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingUnavailable(RuntimeError):
    pass


class Embedder:
    """encode(texts) -> L2-normalised float32 matrix. Results are cached on disk per (model, text)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[Path] = None, batch_size: int = 64,
                 query_prefix: str = "", doc_prefix: str = ""):
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self._model = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self._cache_file: Optional[Path] = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            slug = model_name.replace("/", "__")
            self._cache_file = self.cache_dir / f"{slug}.npz"
            if self._cache_file.exists():
                try:
                    z = np.load(self._cache_file, allow_pickle=False)
                    self._cache = {k: v for k, v in zip(z["keys"].tolist(), z["vecs"])}
                except Exception:
                    self._cache = {}

    def _load(self):
        if self._model is not None:
            return self._model
        if torch is None:
            raise EmbeddingUnavailable("torch is not importable")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:  # network/model missing
            raise EmbeddingUnavailable(f"cannot load embedding model '{self.model_name}': {e}") from e
        return self._model

    @staticmethod
    def _h(model: str, text: str) -> str:
        return hashlib.blake2b((model + "\x00" + text).encode("utf-8"), digest_size=12).hexdigest()

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        keys = [self._h(self.model_name, prefix + t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            uniq: dict[str, str] = {}
            for i in missing:
                uniq.setdefault(keys[i], prefix + texts[i])
            model = self._load()
            order = list(uniq.keys())
            vecs = model.encode([uniq[k] for k in order], batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            for k, v in zip(order, vecs):
                self._cache[k] = v.astype(np.float32)
            self._dirty = True
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys])

    def save(self) -> None:
        if self._dirty and self._cache_file is not None:
            keys = np.array(list(self._cache.keys()))
            vecs = np.vstack(list(self._cache.values()))
            np.savez(self._cache_file, keys=keys, vecs=vecs)
            self._dirty = False


def cosine_scores(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y.T
