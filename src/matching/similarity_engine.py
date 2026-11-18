"""String-similarity primitives (RapidFuzz) used by transaction matching and entity resolution.

All scores are returned on a 0..1 scale. Blank inputs yield `None` ("no evidence"), never 0, so that missing
data does not count as a disagreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from rapidfuzz import fuzz, process


def name_similarity(a: str, b: str) -> Optional[float]:
    """Similarity of two compacted supplier names (spaces/punctuation already removed)."""
    if not a or not b:
        return None
    if a == b:
        return 1.0
    r = fuzz.ratio(a, b) / 100.0
    # containment ("oakent" in "oakentmining") is weaker evidence than edit similarity
    if len(a) >= 5 and len(b) >= 5 and (a in b or b in a):
        r = max(r, 0.85)
    return r


def text_similarity(a: str, b: str) -> Optional[float]:
    """Similarity of two normalised free-text descriptions, tolerant to OCR noise and word re-ordering."""
    if not a or not b:
        return None
    if a == b:
        return 1.0
    return max(fuzz.ratio(a, b), 0.95 * fuzz.token_set_ratio(a, b), 0.95 * fuzz.token_sort_ratio(a, b)) / 100.0


def quantity_equal(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-6) -> Optional[bool]:
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return None
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def amount_close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def weighted_score(parts: dict[str, tuple[Optional[float], float]]) -> float:
    """parts: name -> (value in 0..1 or None, weight). Missing evidence is excluded, remaining weights renormalised."""
    num = den = 0.0
    for v, w in parts.values():
        if v is None:
            continue
        num += v * w
        den += w
    return num / den if den else 0.0


def cluster_by_threshold(strings: list[str], threshold: float, blocker=None, scorer=fuzz.ratio) -> list[int]:
    """Single-linkage clustering of `strings` (unique, normalised) at `threshold` (0..100) using blocking.

    `blocker(s) -> key` restricts comparisons to strings sharing a key (avoids O(n^2) over everything).
    Returns the cluster label for each input string.
    """
    n = len(strings)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    blocks: dict = {}
    for i, s in enumerate(strings):
        keys = blocker(s) if blocker else [""]
        for k in (keys if isinstance(keys, (list, tuple, set)) else [keys]):
            blocks.setdefault(k, []).append(i)
    for idxs in blocks.values():
        if len(idxs) < 2:
            continue
        sub = [strings[i] for i in idxs]
        mat = process.cdist(sub, sub, scorer=scorer, score_cutoff=threshold, dtype=np.uint8, workers=1)
        ii, jj = np.nonzero(np.triu(mat >= threshold, k=1))
        for a, b in zip(ii, jj):
            union(idxs[a], idxs[b])
    return [find(i) for i in range(n)]
