"""Supplier entity resolution: one real supplier -> one supplier_key, however its name is typed.

Evidence (strongest first), combined with union-find so every merge is explainable:
  1. the same vendor id (scope configurable: global, or per source system)
  2. identical normalised name after legal-suffix, punctuation, spacing and OCR-digit normalisation
  3. fuzzy name similarity above a high threshold on names long enough to make 1-edit typos meaningful
  4. names on lines linked to the same PO (PO belongs to one supplier) AND moderately similar names
Names are never merged on a weak similarity alone, and different vendor ids are never merged without a
name-level reason: over-merging silently distorts spend per supplier, so the default is conservative.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.preprocessing.text_normalizer import compact, normalize_supplier_name

_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "6": "g", "$": "s", "|": "l"})


def supplier_name_key(name) -> str:
    """Compact comparison key: 'MEL DRIN (PTY) LTD' and 'MELDRIN' -> 'meldrin'. Digits that are OCR look-alikes map to letters."""
    base = compact(normalize_supplier_name(name))
    return base.translate(_OCR)


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # deterministic root: lexicographically smaller
        if ra < rb:
            self.p[rb] = ra
        else:
            self.p[ra] = rb
        return True


def resolve_suppliers(txn: pd.DataFrame, cfg: dict) -> tuple[pd.Series, dict]:
    """Return (supplier_key per row, diagnostics)."""
    scfg = ((cfg.get("entity_resolution") or {}).get("supplier") or {})
    threshold = float(scfg.get("name_threshold", 92))
    min_len = int(scfg.get("min_len_fuzzy", 7))
    po_sim = float(scfg.get("po_link_similarity", 75))
    scope = scfg.get("vendor_id_scope", "global")

    names = txn["supplier_name"].fillna("").astype(str) if "supplier_name" in txn.columns else pd.Series("", index=txn.index)
    ids = txn["supplier_id"].fillna("").astype(str).str.strip() if "supplier_id" in txn.columns else pd.Series("", index=txn.index)
    systems = txn["system"].fillna("").astype(str) if "system" in txn.columns else pd.Series("", index=txn.index)
    nkey = names.map(supplier_name_key)
    n_blank_name = nkey.eq("")
    id_node = pd.Series("", index=txn.index, dtype=object)
    has_id = ids.ne("") & ~ids.str.fullmatch(r"0+")
    id_node[has_id] = "I:" + (systems[has_id] + ":" if scope == "per_system" else "") + ids[has_id]
    name_node = pd.Series("", index=txn.index, dtype=object)
    name_node[~n_blank_name] = "N:" + nkey[~n_blank_name]

    uf = _UF()
    n_rows_id_name = 0
    # 1. same vendor id / name <-> id linkage observed on the same line
    pairs = pd.DataFrame({"i": id_node, "n": name_node})
    both = pairs[(pairs.i != "") & (pairs.n != "")].drop_duplicates()
    for i, n in zip(both["i"], both["n"]):
        uf.union(i, n)
        n_rows_id_name += 1
    for node in set(id_node[id_node != ""]) | set(name_node[name_node != ""]):
        uf.find(node)

    # 3. fuzzy merging of distinct names (all-pairs if small, prefix blocks if large)
    uniq = sorted(set(nkey[~n_blank_name]))
    merges_fuzzy = 0
    uncertain: list[tuple[str, str, float]] = []
    blocks: dict[str, list[str]] = defaultdict(list)
    if len(uniq) <= 4000:
        blocks[""] = uniq
    else:
        for u in uniq:
            blocks[u[:2]].append(u)
    for blk in blocks.values():
        if len(blk) < 2:
            continue
        mat = process.cdist(blk, blk, scorer=fuzz.ratio, dtype=np.uint8, workers=1)
        ii, jj = np.nonzero(np.triu(mat >= threshold - 10, k=1))
        for a, b in zip(ii, jj):
            sa, sb = blk[a], blk[b]
            score = float(mat[a, b])
            if min(len(sa), len(sb)) < min_len and score < 100:
                continue
            if re.sub(r"[a-z]", "", sa) != re.sub(r"[a-z]", "", sb):  # digit patterns must agree
                continue
            if score >= threshold:
                if uf.union("N:" + sa, "N:" + sb):
                    merges_fuzzy += 1
            else:
                uncertain.append((sa, sb, score))

    # 4. PO-linked names
    merges_po = 0
    if "po_norm" in txn.columns:
        po = txn["po_norm"]
        by_po = pd.DataFrame({"po": po, "n": name_node})[(po != "") & (name_node != "")].drop_duplicates()
        for _, grp in by_po.groupby("po"):
            ns = grp["n"].tolist()
            if len(ns) < 2:
                continue
            base = ns[0]
            for other in ns[1:]:
                if uf.find(base) == uf.find(other):
                    continue
                if fuzz.ratio(base[2:], other[2:]) >= po_sim:
                    merges_po += uf.union(base, other)

    # canonical label per cluster: most frequent readable normalised name, else the vendor id
    node_for_row = name_node.where(name_node != "", id_node)
    root = node_for_row.map(lambda x: uf.find(x) if x else "")
    label_counts: dict[str, Counter] = defaultdict(Counter)
    for r, nm in zip(root, names.map(normalize_supplier_name)):
        if r and nm:
            label_counts[r][nm] += 1
    id_labels: dict[str, str] = {}
    for r, i in zip(root, id_node):
        if r and i and r not in id_labels:
            id_labels[r] = i[2:]
    label: dict[str, str] = {}
    for r in set(root[root != ""]):
        if label_counts.get(r):
            label[r] = label_counts[r].most_common(1)[0][0]
        else:
            label[r] = "vendor " + id_labels.get(r, r)
    key = root.map(lambda r: label.get(r, "unknown supplier") if r else "unknown supplier")

    ident = node_for_row[node_for_row != ""].nunique()
    diag = {
        "supplier_identities_before": int(ident),
        "distinct_name_keys": int(len(uniq)),
        "distinct_vendor_ids": int(id_node[id_node != ""].nunique()),
        "suppliers_after": int(key.nunique()),
        "merges_vendor_id_name_links": int(n_rows_id_name),
        "merges_fuzzy_name": int(merges_fuzzy),
        "merges_po_linked": int(merges_po),
        "uncertain_pairs_not_merged": int(len(uncertain)),
        "rows_without_any_supplier_evidence": int((key == "unknown supplier").sum()),
        "uncertain_examples_count": min(len(uncertain), 20),
    }
    txn_uncertain = sorted(uncertain, key=lambda x: -x[2])[:20]
    diag["_uncertain_pairs"] = txn_uncertain  # kept out of public reports (names are company data)
    return key, diag


def inherit_from_primary(txn: pd.DataFrame, keys: pd.Series, link_col: str = "match_to") -> pd.Series:
    """Secondary lines that duplicate a primary line inherit the primary line's key (same purchase, same supplier)."""
    if link_col not in txn.columns:
        return keys
    rid = pd.Series(keys.values, index=txn["record_id"].values)
    out = keys.copy()
    link = txn[link_col].fillna("")
    first = link.str.split("|").str[0]
    m = first.ne("") & txn["match_status"].isin(["MATCHED_PRIMARY", "AGGREGATE_MATCH", "DUP_OF_PRIMARY"])
    out[m] = first[m].map(rid.to_dict()).fillna(keys[m])
    return out
