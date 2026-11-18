"""Cross-system transaction matching: count every real purchase exactly once.

Evidence model (nothing is decided on description similarity alone):
  amount   net amounts equal within a rounding tolerance (gross->net conversion differs by a cent or two)
  po       normalised PO core identical (formatting like 'PO#', 'PO ', '/10.0' suffix ignored)
  supplier fuzzy similarity of normalised supplier names
  quantity equal quantities
  text     fuzzy line-text similarity (supporting evidence only)
  date     within a window (only when both dates exist)

A candidate pair is only eligible if the AMOUNT agrees AND (same PO OR (supplier agrees AND (quantity agrees OR
text agrees))). Eligible pairs are scored, then assigned greedily one-to-one by descending score (ERP precedence:
the primary-source line is kept, the secondary-source line becomes DUPLICATE).
Extra stages: (a) one secondary line that settles several primary lines of the same PO (subset-sum),
(b) secondary lines that replicate a purchase already represented by a primary line (or by another secondary line):
    same amount + quantity + supplier + PO. The invoice NUMBER is deliberately not required to be equal: the data
    shows re-exports carrying a different invoice number / date format for the same purchase.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from src.matching.similarity_engine import (amount_close, name_similarity, quantity_equal, text_similarity,
                                            weighted_score)
from src.preprocessing.text_normalizer import (compact, normalize_invoice_number, normalize_reference,
                                               normalize_supplier_name, normalize_text)

WEIGHTS = {"amount": 0.35, "po": 0.25, "supplier": 0.15, "quantity": 0.10, "text": 0.10, "date": 0.05}


def prepare_match_columns(txn: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    po_rx = mcfg.get("po_regex")
    txn["po_norm"] = txn["po"].map(lambda x: normalize_reference(x, po_rx)) if "po" in txn.columns else ""
    txn["inv_norm"] = txn["invoice_number"].map(normalize_invoice_number)
    txn["sup_norm"] = txn["supplier_name"].map(lambda x: compact(normalize_supplier_name(x)))
    txn["desc_norm"] = txn["description"].map(normalize_text)
    q = txn["quantity"] if "quantity" in txn.columns else pd.Series("", index=txn.index)
    txn["qty_num"] = pd.to_numeric(q.astype(str).str.replace(",", ""), errors="coerce")
    return txn


def _pair_score(s: dict, p: dict, tol_abs: float, tol_rel: float, window_days: int) -> tuple[float, dict]:
    """Return (score, evidence)."""
    amt_ok = amount_close(s["net"], p["net"], tol_abs, tol_rel)
    po_same = bool(s["po"] and p["po"] and s["po"] == p["po"])
    po_diff = bool(s["po"] and p["po"] and s["po"] != p["po"])
    sup = name_similarity(s["sup"], p["sup"])
    qty_eq = quantity_equal(s["qty"], p["qty"])
    txt = text_similarity(s["desc"], p["desc"])
    date = None
    if pd.notna(s["date"]) and pd.notna(p["date"]):
        gap = abs((s["date"] - p["date"]).days)
        date = 1.0 if gap <= window_days else max(0.0, 1.0 - (gap - window_days) / 730.0)
    ev = {"amount": amt_ok, "po_same": po_same, "po_diff": po_diff, "sup": sup, "qty_eq": qty_eq, "txt": txt}
    parts = {
        "amount": (1.0 if amt_ok else 0.0, WEIGHTS["amount"]),
        "po": ((1.0 if po_same else 0.0) if (s["po"] and p["po"]) else None, WEIGHTS["po"]),
        "supplier": (sup, WEIGHTS["supplier"]),
        "quantity": (None if qty_eq is None else float(qty_eq), WEIGHTS["quantity"]),
        "text": (txt, WEIGHTS["text"]),
        "date": (date, WEIGHTS["date"]),
    }
    return weighted_score(parts), ev


def _eligible(ev: dict, sup_min: float, txt_min: float) -> bool:
    if not ev["amount"]:
        return False
    if ev["po_same"]:
        return True
    sup_ok = ev["sup"] is not None and ev["sup"] >= sup_min
    qty_ok = ev["qty_eq"] is True
    txt_ok = ev["txt"] is not None and ev["txt"] >= txt_min
    if ev["po_diff"]:  # a different PO is strong counter-evidence: demand supplier, quantity and text together
        return sup_ok and qty_ok and txt_ok
    return sup_ok and (qty_ok or txt_ok)


def _subset_sums(vals: np.ndarray) -> np.ndarray:
    sums = np.zeros(1)
    for v in vals:
        sums = np.concatenate([sums, sums + v])
    return sums


def dedupe_secondary_exports(sec: pd.DataFrame, tol_abs: float, tol_rel: float, sup_min: float,
                             txt_min: float, date_days: int = 7) -> dict[int, int]:
    """The same purchase present on several secondary lines. Returns {duplicate_index: kept_index}.

    Two lines are the same purchase when: amounts equal (tolerance), quantities not different, suppliers not
    different, texts similar, and the PO agrees (both present and equal). If a PO is missing on either side the pair
    additionally needs the same normalised invoice number or dates within `date_days`.
    """
    if sec.empty:
        return {}
    ids = sec.index.to_numpy()
    net = sec["net_local"].to_numpy(dtype=float)
    order = np.argsort(net, kind="mergesort")
    parent = {int(i): int(i) for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cols = {c: sec[c].to_numpy() for c in ("po_norm", "inv_norm", "sup_norm", "desc_norm", "qty_num")}
    dates = sec["date_parsed"].to_numpy() if "date_parsed" in sec.columns else np.full(len(sec), np.datetime64("NaT", "ns"))
    n = len(sec)
    for a in range(n):
        i = order[a]
        if np.isnan(net[i]):
            continue
        for b in range(a + 1, n):
            j = order[b]
            if np.isnan(net[j]):
                continue
            tol = max(tol_abs, tol_rel * max(abs(net[i]), abs(net[j])))
            if net[j] - net[i] > tol:
                break
            if quantity_equal(cols["qty_num"][i], cols["qty_num"][j]) is False:
                continue
            sv = name_similarity(cols["sup_norm"][i], cols["sup_norm"][j])
            if sv is not None and sv < sup_min:
                continue
            tv = text_similarity(cols["desc_norm"][i], cols["desc_norm"][j])
            if tv is not None and tv < txt_min:
                continue
            pi, pj = cols["po_norm"][i], cols["po_norm"][j]
            if pi and pj:
                if pi != pj:
                    continue
            else:
                same_inv = bool(cols["inv_norm"][i]) and cols["inv_norm"][i] == cols["inv_norm"][j]
                close_date = (not np.isnat(dates[i])) and (not np.isnat(dates[j])) and \
                    abs((dates[i] - dates[j]) / np.timedelta64(1, "D")) <= date_days
                if not (same_inv or close_date):
                    continue
            ri, rj = find(int(ids[i])), find(int(ids[j]))
            if ri != rj:
                parent[rj] = ri
    groups: dict[int, list[int]] = defaultdict(list)
    for i in ids:
        groups[find(int(i))].append(int(i))
    out: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        sub = sec.loc[members].sort_values(["has_po", "src_row"], ascending=[False, True])
        keep = int(sub.index[0])
        for m in members:
            if m != keep:
                out[m] = keep
    return out


def match_transactions(txn: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Annotate `txn` with match_status / match_to / match_score and return (txn, diagnostics).

    match_status values: '' (not a secondary row), 'MATCHED_PRIMARY', 'AGGREGATE_MATCH', 'DUP_OF_PRIMARY'
    (replicates a primary line already consumed by a better secondary match), 'EXPORT_DUPLICATE' (replicates another
    secondary line), 'AMBIGUOUS' (weak evidence: kept and flagged), 'UNMATCHED' (genuinely only in the secondary system).
    """
    mcfg = cfg.get("matching") or {}
    txn["match_status"] = ""
    txn["match_to"] = ""
    txn["match_score"] = np.nan
    diag: dict = {"enabled": bool(mcfg.get("enabled"))}
    if not mcfg.get("enabled"):
        return txn, diag
    txn = prepare_match_columns(txn, mcfg)
    tol_abs = float(mcfg.get("amount_tolerance_abs", 0.05))
    tol_rel = float(mcfg.get("amount_tolerance_rel", 0.0002))
    sup_min = float(mcfg.get("supplier_match_min", 0.8))
    txt_min = float(mcfg.get("text_match_min", 0.55))
    accept = float(mcfg.get("accept_score", 0.7))
    ambig = float(mcfg.get("ambiguous_score", 0.5))
    window = int(mcfg.get("date_window_days", 365))
    max_subset = int(mcfg.get("max_subset_lines", 14))

    live = txn["status_code"].eq("") & ~txn["np_flag"] & ~txn["dup_in_source"]
    prim = txn[(txn["role"] == "primary") & live]
    sec_all = txn[(txn["role"] == "secondary")]
    sec = txn[(txn["role"] == "secondary") & live].copy()
    sec["has_po"] = sec["po_norm"].ne("")
    txn.loc[sec_all.index, "match_status"] = "UNMATCHED"

    p_idx = prim.index.to_numpy()
    p_net = prim["net_local"].to_numpy(dtype=float)
    order = np.argsort(p_net, kind="mergesort")
    p_net_sorted = p_net[order]
    po_map: dict[str, list[int]] = defaultdict(list)
    for pos, po in enumerate(prim["po_norm"].tolist()):
        if po:
            po_map[po].append(pos)
    P = {
        "net": p_net, "po": prim["po_norm"].tolist(), "sup": prim["sup_norm"].tolist(),
        "qty": prim["qty_num"].to_numpy(dtype=float), "desc": prim["desc_norm"].tolist(),
        "date": prim["date_parsed"].tolist() if "date_parsed" in prim.columns else [pd.NaT] * len(prim),
    }
    s_net = sec["net_local"].to_numpy(dtype=float)
    s_po, s_sup = sec["po_norm"].tolist(), sec["sup_norm"].tolist()
    s_qty, s_desc = sec["qty_num"].to_numpy(dtype=float), sec["desc_norm"].tolist()
    s_date = sec["date_parsed"].tolist() if "date_parsed" in sec.columns else [pd.NaT] * len(sec)
    s_index = sec.index.to_numpy()

    # ---- stage 1: eligible (secondary, primary) edges; the amount must agree, so candidates = amount window --------
    edges: list[tuple[float, int, int]] = []
    for k in range(len(sec)):
        if np.isnan(s_net[k]):
            continue
        tol = max(tol_abs, tol_rel * abs(s_net[k]))
        lo = np.searchsorted(p_net_sorted, s_net[k] - tol, side="left")
        hi = np.searchsorted(p_net_sorted, s_net[k] + tol, side="right")
        if hi <= lo:
            continue
        s = {"net": s_net[k], "po": s_po[k], "sup": s_sup[k], "qty": s_qty[k], "desc": s_desc[k], "date": s_date[k]}
        for pos in order[lo:hi].tolist():
            p = {key: P[key][pos] for key in P}
            score, ev = _pair_score(s, p, tol_abs, tol_rel, window)
            if _eligible(ev, sup_min, txt_min):
                edges.append((score, k, pos))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    p_taken: dict[int, int] = {}
    s_done: dict[int, tuple[int, float]] = {}
    ambiguous_edges: dict[int, tuple[float, int]] = {}
    for score, k, pos in edges:
        if k in s_done or pos in p_taken:
            continue
        if score >= accept:
            s_done[k] = (pos, score)
            p_taken[pos] = k
        elif score >= ambig and k not in ambiguous_edges:
            ambiguous_edges[k] = (score, pos)
    for k, (pos, score) in s_done.items():
        i = s_index[k]
        txn.at[i, "match_status"] = "MATCHED_PRIMARY"
        txn.at[i, "match_to"] = txn.at[p_idx[pos], "record_id"]
        txn.at[i, "match_score"] = round(score, 4)
    n_single = len(s_done)

    # ---- stage 2: one secondary line settling several primary lines of the same PO ---------------------------
    n_agg = 0
    for k in range(len(sec)):
        if k in s_done or not s_po[k] or np.isnan(s_net[k]) or s_po[k] not in po_map:
            continue
        free = [pos for pos in po_map[s_po[k]] if pos not in p_taken]
        if len(free) < 2:
            continue
        free = [pos for pos in free if (name_similarity(s_sup[k], P["sup"][pos]) or 1.0) >= sup_min]
        if len(free) < 2:
            continue
        tol = max(tol_abs * 2, tol_rel * abs(s_net[k]))
        chosen: Optional[list[int]] = None
        if len(free) <= max_subset:
            vals = np.array([P["net"][pos] for pos in free])
            sums = _subset_sums(vals)
            hits = np.nonzero(np.abs(sums - s_net[k]) <= tol)[0]
            best = None
            for h in hits:
                members = [j for j in range(len(free)) if (int(h) >> j) & 1]
                if len(members) < 2:
                    continue
                qsum = sum(P["qty"][free[j]] for j in members)
                q_ok = quantity_equal(s_qty[k], qsum, rel=1e-3) is not False
                key = (not q_ok, len(members))
                if best is None or key < best[0]:
                    best = (key, members)
            if best is not None and not best[0][0]:
                chosen = [free[j] for j in best[1]]
        elif abs(sum(P["net"][pos] for pos in free) - s_net[k]) <= tol:
            chosen = free
        if chosen:
            i = s_index[k]
            s_done[k] = (chosen[0], 0.9)
            for pos in chosen:
                p_taken[pos] = k
            txn.at[i, "match_status"] = "AGGREGATE_MATCH"
            txn.at[i, "match_to"] = "|".join(txn.at[p_idx[pos], "record_id"] for pos in chosen)
            txn.at[i, "match_score"] = 0.9
            n_agg += 1

    # ---- stage 3: lines replicating a primary line that a better secondary line already consumed ----------------
    n_dup_primary = 0
    for score, k, pos in edges:
        if k in s_done or score < accept:
            continue
        i = s_index[k]
        if txn.at[i, "match_status"] == "UNMATCHED" and pos in p_taken:
            txn.at[i, "match_status"] = "DUP_OF_PRIMARY"
            txn.at[i, "match_to"] = txn.at[p_idx[pos], "record_id"]
            txn.at[i, "match_score"] = round(score, 4)
            n_dup_primary += 1

    # ---- stage 4: ambiguous leftovers are kept (and flagged) rather than forced -------------------------------------
    n_amb = 0
    for k, (score, pos) in ambiguous_edges.items():
        i = s_index[k]
        if txn.at[i, "match_status"] != "UNMATCHED":
            continue
        txn.at[i, "match_status"] = "AMBIGUOUS"
        txn.at[i, "match_to"] = txn.at[p_idx[pos], "record_id"]
        txn.at[i, "match_score"] = round(score, 4)
        n_amb += 1

    # ---- stage 5: the same secondary purchase exported more than once ---------------------------------------------------
    rest = sec[txn.loc[sec.index, "match_status"].isin(["UNMATCHED", "AMBIGUOUS"])]
    exp_dups = dedupe_secondary_exports(rest, tol_abs, tol_rel, sup_min, txt_min)
    for dup_i, kept_i in exp_dups.items():
        txn.at[dup_i, "match_status"] = "EXPORT_DUPLICATE"
        txn.at[dup_i, "match_to"] = txn.at[kept_i, "record_id"]

    st = txn.loc[sec_all.index, "match_status"].value_counts().to_dict()
    ms = txn.loc[txn.match_status == "MATCHED_PRIMARY", "match_score"].astype(float)
    diag.update({
        "primary_live_rows": int(len(prim)), "secondary_rows": int(len(sec_all)), "secondary_live_rows": int(len(sec)),
        "status_counts": st, "single_line_matches": n_single, "aggregate_matches": n_agg, "ambiguous": n_amb,
        "dup_of_consumed_primary": n_dup_primary, "export_duplicates": len(exp_dups),
        "primary_lines_consumed": len(p_taken),
        "matched_score_quantiles_5_25_50": [float(ms.quantile(q)) for q in (0.05, 0.25, 0.5)] if n_single else [],
    })
    return txn, diag
