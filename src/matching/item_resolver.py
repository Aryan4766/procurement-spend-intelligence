"""Item entity resolution: one real item -> one item_key, even when descriptions are typed differently.

Conservative by construction (similar descriptions do not prove identical products):
  1. An item/material code is authoritative: the same code is one item; two different (non-generic) codes are
     never merged on description similarity (the labelled data shows identical wording under different codes is
     usually a different catalogue item: another pack size, grade or contract).
  2. A code is ignored when it is a placeholder: it carries many unrelated descriptions (one 'misc' code on thousands
     of lines).
  3. Lines without a usable code are grouped by description: near-identical text (RapidFuzz ratio) AND identical numeric
     specifications (sizes, ratings, lengths must agree: '37,5mm' and '53mm' are different products) within the same
     supplier; identical long descriptions may also merge across suppliers.
  4. A code-less group that matches a coded item (same supplier, same specs, near-identical text) joins it.
  5. Nothing to go on (no code, blank / generic text) -> 'unspecified item'.
OCR-style letter/digit confusions inside words ('RE5IN', 'B0OM') are repaired before comparison.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.preprocessing.text_normalizer import normalize_text

_OCR_DIGITS = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b", "6": "g", "4": "a"})
_GENERIC = {"product", "products", "parts", "part", "item", "items", "material", "materials", "misc", "other",
            "various", "service", "services", "general", "sundry", "goods", "spare", "spares", "n a", "na", "none"}


def repair_ocr(text: str) -> str:
    """'re5in roof b0lt 25mm' -> 'resin roof bolt 25mm'. Only digit-poor, letter-rich tokens are touched."""
    out = []
    for tok in text.split():
        letters = sum(c.isalpha() for c in tok)
        digits = sum(c.isdigit() for c in tok)
        if letters >= 3 and 0 < digits <= 2 and not re.search(r"\d{2}", tok) and not re.fullmatch(r"[a-z]{1,2}\d{1,3}[a-z]?", tok):
            tok = tok.translate(_OCR_DIGITS)
        out.append(tok)
    return " ".join(out)


def description_key(text) -> str:
    return repair_ocr(normalize_text(text))


def spec_signature(desc_key: str) -> tuple:
    """All numeric runs, in order-insensitive form. '3,5M,8' and '3.5M,8' agree; '3,5M,6' does not."""
    return tuple(sorted(re.findall(r"\d+", desc_key)))


def _is_generic(desc_key: str) -> bool:
    return desc_key in _GENERIC or len(desc_key) < 4


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def resolve_items(txn: pd.DataFrame, supplier_keys: pd.Series, cfg: dict) -> tuple[pd.Series, dict]:
    icfg = ((cfg.get("entity_resolution") or {}).get("item") or {})
    thr = float(icfg.get("desc_threshold", 93))
    use_code = bool(icfg.get("use_item_code", True))
    generic_cluster_limit = int(icfg.get("generic_code_min_distinct_descriptions", 8))
    cross_supplier_min_len = int(icfg.get("cross_supplier_min_len", 12))
    n = len(txn)
    codes = txn["item_code"].fillna("").astype(str).str.strip() if "item_code" in txn.columns else pd.Series("", index=txn.index)
    codes = codes.where(~codes.str.fullmatch(r"[0\W_]*"), "")
    if not use_code:
        codes = pd.Series("", index=txn.index)
    dkey = txn["description"].fillna("").astype(str).map(description_key)
    sup = supplier_keys.astype(str)

    # --- generic (placeholder) codes: many distinct, mutually unrelated descriptions ------------------------------
    generic_codes: set[str] = set()
    tmp = pd.DataFrame({"code": codes, "d": dkey})
    tmp = tmp[(tmp["code"] != "") & (tmp["d"] != "")]
    for code, grp in tmp.groupby("code"):
        ds = grp["d"].value_counts().index.tolist()
        if len(ds) < generic_cluster_limit:
            continue
        top = ds[:200]
        mat = process.cdist(top, top, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=1)
        unrelated = (mat < 60).mean()
        if unrelated > 0.5:
            generic_codes.add(code)
    eff_code = codes.where(~codes.isin(generic_codes), "")

    # --- nodes: unique (supplier, description) for lines without an effective code ---------------------------------
    nocode = eff_code.eq("") & dkey.ne("")
    node_df = pd.DataFrame({"sup": sup[nocode], "d": dkey[nocode]}).drop_duplicates().reset_index(drop=True)
    node_id = {(s, d): i for i, (s, d) in enumerate(zip(node_df["sup"], node_df["d"]))}
    uf = _UF(len(node_df))
    n_fuzzy = n_cross = 0
    first_by_desc: dict[str, int] = {}
    for i, d in enumerate(node_df["d"]):
        if _is_generic(d) or len(d) < cross_supplier_min_len:
            continue
        if d in first_by_desc:
            uf.union(first_by_desc[d], i)
            n_cross += 1
        else:
            first_by_desc[d] = i
    for s, grp in node_df.groupby("sup"):
        idx = grp.index.to_list()
        if len(idx) < 2:
            continue
        descs = grp["d"].tolist()
        by_sig: dict[tuple, list[int]] = defaultdict(list)
        for k, dd in enumerate(descs):
            by_sig[spec_signature(dd)].append(k)
        for ks in by_sig.values():
            if len(ks) < 2:
                continue
            for a0 in range(0, len(ks), 3000):
                part = ks[a0:a0 + 3000]
                chunk = [descs[k] for k in part]
                mat = process.cdist(chunk, chunk, scorer=fuzz.ratio, dtype=np.uint8, workers=1, score_cutoff=thr)
                ii, jj = np.nonzero(np.triu(mat >= thr, k=1))
                for a, b in zip(ii, jj):
                    if _is_generic(chunk[a]) or _is_generic(chunk[b]):
                        continue
                    uf.union(idx[part[a]], idx[part[b]])
                    n_fuzzy += 1

    # --- attach code-less description groups to a coded item of the same supplier ------------------------------------
    coded = pd.DataFrame({"sup": sup, "d": dkey, "code": eff_code})
    coded = coded[(coded["code"] != "") & (coded["d"] != "")]
    coded_by_sup: dict[str, tuple[list[str], list[str], list[tuple]]] = {}
    for s, grp in coded.groupby("sup"):
        cnt = grp.groupby(["d", "code"]).size().reset_index(name="n").sort_values("n", ascending=False)
        ds = cnt["d"].tolist()
        coded_by_sup[s] = (ds, cnt["code"].tolist(), [spec_signature(x) for x in ds])
    cluster_code: dict[int, str] = {}
    n_attached = 0
    for s, grp in node_df.groupby("sup"):
        if s not in coded_by_sup:
            continue
        ds, cs, sigs = coded_by_sup[s]
        for i, d in zip(grp.index, grp["d"]):
            if _is_generic(d):
                continue
            sg = spec_signature(d)
            best = process.extract(d, ds, scorer=fuzz.ratio, limit=5, score_cutoff=thr)
            for _, score, pos in best:
                if sigs[pos] == sg:
                    cl = uf.find(i)
                    cluster_code.setdefault(cl, cs[pos])
                    n_attached += 1
                    break

    # --- final identity per row ----------------------------------------------------------------------------------------------
    ident: list[tuple[str, str]] = []
    for code, d, s in zip(eff_code, dkey, sup):
        if code:
            ident.append(("K", code))
        elif d and not _is_generic(d):
            cl = uf.find(node_id[(s, d)])
            ident.append(("K", cluster_code[cl]) if cl in cluster_code else ("D", str(cl)))
        else:
            ident.append(("U", ""))

    desc_votes: dict[tuple, Counter] = defaultdict(Counter)
    for it, d in zip(ident, dkey):
        if it[0] == "D" and d:
            desc_votes[it][d] += 1
    label: dict[tuple, str] = {}
    for it in set(ident):
        if it[0] == "K":
            label[it] = it[1]
        elif it[0] == "D":
            d = desc_votes[it].most_common(1)[0][0]
            label[it] = "desc-" + hashlib.md5(d.encode("utf-8")).hexdigest()[:8] + " " + d[:28]
    key = pd.Series([label.get(it, "unspecified item") for it in ident], index=txn.index)
    diag = {
        "rows": int(n), "distinct_description_keys": int(dkey[dkey != ""].nunique()), "distinct_codes": int(codes[codes != ""].nunique()),
        "generic_codes_ignored": int(len(generic_codes)), "fuzzy_description_merges": int(n_fuzzy),
        "cross_supplier_exact_merges": int(n_cross), "codeless_groups_attached_to_coded_items": int(n_attached),
        "items_after": int(key.nunique()), "rows_unspecified_item": int((key == "unspecified item").sum()),
    }
    return key, diag
