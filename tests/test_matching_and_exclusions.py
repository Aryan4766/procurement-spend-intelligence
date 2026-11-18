"""Matching, exclusions, entity resolution and taxonomy tests on small synthetic data (no confidential records)."""
import numpy as np
import pandas as pd
import pytest

from src.config.config_loader import ConfigError, validate_config
from src.matching.item_resolver import description_key, resolve_items, spec_signature
from src.matching.supplier_resolver import resolve_suppliers, supplier_name_key
from src.matching.transaction_matcher import match_transactions
from src.spend import exclusion_engine as ex
from src.taxonomy.taxonomy_loader import load_taxonomy

MCFG = {"matching": {"enabled": True, "po_regex": r"(\d{8,12})", "amount_tolerance_abs": 0.05, "amount_tolerance_rel": 0.0002,
                     "supplier_match_min": 0.8, "text_match_min": 0.55, "accept_score": 0.7, "ambiguous_score": 0.5}}


def _row(rid, source, role, po, net, qty, sup, desc, inv="", date=pd.NaT, status=""):
    return dict(record_id=rid, source=source, role=role, po=po, net_local=net, quantity=qty, supplier_name=sup,
                description=desc, invoice_number=inv, date_parsed=date, status_code=status, np_flag=False,
                dup_in_source=False, src_row=0)


def _txn(rows):
    df = pd.DataFrame(rows)
    df.index = range(len(df))
    return df


def test_matching_counts_each_purchase_once_and_one_to_one():
    rows = [
        _row("ERP:1", "erp", "primary", "4510000001", 100.0, 5, "ACME (PTY) LTD", "VALVE BALL 2IN"),
        _row("ERP:2", "erp", "primary", "4510000001", 100.0, 5, "ACME (PTY) LTD", "VALVE BALL 2IN"),  # a second identical receipt
        _row("TOOL:1", "tool", "secondary", "PO#4510000001", 100.0, 5, "ACME PTY LTD", "VALVE BALL 2IN", "INV/2024/111"),
        _row("TOOL:2", "tool", "secondary", "4510000001/10.0", 100.01, 5, "ACME  (PTY) LTD", "VALVE BALL 2lN", "SI-2024-222"),
        _row("TOOL:3", "tool", "secondary", "PO 4510000001", 100.0, 5, "ACME PTY LTD", "VALVE BALL 2IN", "333"),  # third copy: no ERP left
        _row("TOOL:4", "tool", "secondary", "", 777.0, 1, "OTHER LTD", "SOMETHING ELSE", "444"),  # only in the tool
    ]
    out, diag = match_transactions(_txn(rows), MCFG)
    st = dict(zip(out.record_id, out.match_status))
    first_three = [st["TOOL:1"], st["TOOL:2"], st["TOOL:3"]]
    assert sorted(first_three) == ["DUP_OF_PRIMARY", "MATCHED_PRIMARY", "MATCHED_PRIMARY"]  # two ERP lines, three copies
    assert st["TOOL:4"] == "UNMATCHED"                # genuine tool-only spend is kept
    assert len(set(out.loc[out.match_status == "MATCHED_PRIMARY", "match_to"])) == 2  # one-to-one


def test_description_alone_never_matches():
    rows = [
        _row("ERP:1", "erp", "primary", "4510000001", 500.0, 5, "ACME LTD", "VALVE BALL 2IN"),
        _row("TOOL:1", "tool", "secondary", "", 900.0, 5, "ZED LTD", "VALVE BALL 2IN", "1"),
    ]
    out, _ = match_transactions(_txn(rows), MCFG)
    assert out.loc[1, "match_status"] == "UNMATCHED"


def test_amount_alone_never_matches():
    rows = [
        _row("ERP:1", "erp", "primary", "4510000001", 500.0, 5, "ACME LTD", "VALVE BALL 2IN"),
        _row("TOOL:1", "tool", "secondary", "4510000099", 500.0, 9, "ZED LTD", "PAPER A4 WHITE", "1"),
    ]
    out, _ = match_transactions(_txn(rows), MCFG)
    assert out.loc[1, "match_status"] == "UNMATCHED"


def test_aggregate_invoice_settles_several_receipts_of_one_po():
    rows = [
        _row("ERP:1", "erp", "primary", "4510000002", 96048.0, 2000, "IVES LTD", "BIT RCK DIA 25"),
        _row("ERP:2", "erp", "primary", "4510000002", 48024.0, 1000, "IVES LTD", "BIT RCK DIA 25"),
        _row("TOOL:1", "tool", "secondary", "4510000002", 144072.0, 3000, "IVES LTD", "BIT RCK DIA 25", "9"),
    ]
    out, diag = match_transactions(_txn(rows), MCFG)
    assert out.loc[2, "match_status"] == "AGGREGATE_MATCH" and diag["aggregate_matches"] == 1
    assert set(out.loc[2, "match_to"].split("|")) == {"ERP:1", "ERP:2"}


def test_secondary_export_duplicate_with_different_invoice_number():
    rows = [
        _row("TOOL:1", "tool", "secondary", "4510000003", 250.0, 2, "ACME LTD", "GLOVES LEATHER", "SI-2022-065598"),
        _row("TOOL:2", "tool", "secondary", "PO#4510000003", 250.0, 2, "ACME LTD", "GL0VES LEATHER", "TAX INV 322915"),
    ]
    out, _ = match_transactions(_txn(rows), MCFG)
    assert sorted(out.match_status) == ["EXPORT_DUPLICATE", "UNMATCHED"]


def test_repeated_monthly_charge_without_po_is_not_a_duplicate():
    rows = [
        _row("TOOL:1", "tool", "secondary", "", 1000.0, 1, "CLEAN CO", "CLEANING SERVICE", "11", pd.Timestamp("2024-01-31")),
        _row("TOOL:2", "tool", "secondary", "", 1000.0, 1, "CLEAN CO", "CLEANING SERVICE", "12", pd.Timestamp("2024-02-29")),
    ]
    out, _ = match_transactions(_txn(rows), MCFG)
    assert set(out.match_status) == {"UNMATCHED"}


# -------------------------------------------------------------------------------------------------- exclusions
def test_status_mapping():
    s = pd.Series(["Approved", "PAID", "Cancelled", "On Hold", "Rejected", "Submitted", "Disputed", ""])
    codes = ex._status_codes(s, {"approved": ["approved", "paid"], "cancelled": ["cancelled"]})
    assert list(codes) == ["", "", "CANCELLED", "NOT_APPROVED", "NOT_APPROVED", "NOT_APPROVED", "NOT_APPROVED", ""]


def test_precedence_of_exclusion_reasons():
    df = pd.DataFrame({"x": range(3)})
    status = pd.Series(["CANCELLED", "", ""])
    npf = pd.Series([True, True, False])
    dup = pd.Series([True, True, True])
    r = ex.combine(df, status, npf, dup, ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"])
    assert list(r) == ["CANCELLED", "NON_PROCUREMENT", "DUPLICATE"]


def test_non_procurement_rules_and_missing_fields_are_skipped():
    df = pd.DataFrame({"source": ["a"] * 4, "account_desc": ["IC Payables", "I/C Payable", "AP - Other", "Intercompany - CZ"],
                       "description": ["x", "State & Local Taxes -", "Coffee Supplies", ""]})
    rules = [{"name": "ic", "fields": ["account_desc", "not_mapped"], "regex": r"(?<![A-Za-z])(I/C|IC\d{0,3}|inter[ -]?co(mpany)?)(?![A-Za-z])"},
             {"name": "tax", "fields": ["description"], "regex": r"^\s*state\s*(&|and)?\s*local\s+tax"}]
    flag, which = ex.apply_non_procurement(df, rules)
    assert list(flag) == [True, True, False, True]


def test_in_source_reexport_keeps_first():
    df = pd.DataFrame({"source": ["s"] * 3, "_dupkey": ["k1", "k1", "k2"]})
    d = ex.apply_within_source_duplicates(df, pd.Series([True] * 3))
    assert list(d) == [False, True, False]


# -------------------------------------------------------------------------------------------------- suppliers / items
def test_supplier_resolution_merges_typing_variants_but_not_different_suppliers():
    names = ["ZORVEX TECHNOLOGIES (PTY) LTD", "ZORVEX  TECHNOLOGIES (PTY) LTD", "ZORVEX TECHNOLOGIES PTY LTD", "MEL DRIN (PTY) LTD",
             "MELDRIN (PTY) LTD", "TARVEN EQUIPMENT (PTY) LTD", "TARMON EQUIPMENT (PTY) LTD", "BORL0NT (PTY) LTD", "BORLONT (PTY) LTD"]
    txn = pd.DataFrame({"supplier_name": names, "supplier_id": [""] * len(names)})
    key, diag = resolve_suppliers(txn, {"entity_resolution": {"supplier": {}}})
    assert key.iloc[0] == key.iloc[1] == key.iloc[2]
    assert key.iloc[3] == key.iloc[4]
    assert key.iloc[7] == key.iloc[8]
    assert key.iloc[5] != key.iloc[6]                       # similar but different suppliers stay apart
    assert key.iloc[0] != key.iloc[3]


def test_supplier_vendor_id_links_names_and_scope():
    txn = pd.DataFrame({"supplier_name": ["ALPHA SUPPLY", "", "BETA TRADING", ""], "supplier_id": ["V1", "V1", "V2", "V2"],
                        "system": ["S1", "S2", "S1", "S2"]})
    key, _ = resolve_suppliers(txn, {"entity_resolution": {"supplier": {"vendor_id_scope": "global"}}})
    assert key.iloc[0] == key.iloc[1] and key.iloc[2] == key.iloc[3] and key.iloc[0] != key.iloc[2]
    key2, _ = resolve_suppliers(txn.assign(supplier_name=["", "", "", ""]), {"entity_resolution": {"supplier": {"vendor_id_scope": "per_system"}}})
    assert key2.iloc[0] != key2.iloc[1]                      # per-system scope never merges ids of different ERPs


def test_item_spec_guard_and_ocr_repair():
    assert description_key("RE5IN, ROOF B0LT") == description_key("RESIN, ROOF BOLT")
    assert spec_signature(description_key("STONE 37,5MM")) != spec_signature(description_key("STONE 53MM"))
    txn = pd.DataFrame({"item_code": ["", "", "", ""], "description": ["STONE CRUSHED 37,5MM", "STONE  CRUSHED 37,5MM", "STONE CRUSHED 53MM", ""]})
    key, _ = resolve_items(txn, pd.Series(["s"] * 4), {"entity_resolution": {"item": {}}})
    assert key.iloc[0] == key.iloc[1] and key.iloc[0] != key.iloc[2] and key.iloc[3] == "unspecified item"


def test_item_codes_are_authoritative():
    txn = pd.DataFrame({"item_code": ["M1", "M2", "M1"], "description": ["BOLT ROOF 20MM", "BOLT ROOF 20MM", "BOLT  ROOF 20 MM"]})
    key, _ = resolve_items(txn, pd.Series(["s"] * 3), {"entity_resolution": {"item": {}}})
    assert key.iloc[0] == key.iloc[2] and key.iloc[0] != key.iloc[1]


# -------------------------------------------------------------------------------------------------- taxonomy / config
def test_taxonomy_loading_variable_depth(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("l1,l2,l3,l4\nX,Y,Z,W\nX,Y,Z2,\nA,B,,\nA,C,,\n", encoding="utf-8")
    tax = load_taxonomy(p, ["l1", "l2", "l3", "l4"])
    assert tax.n_leaves == 4 and tax.depth == 4
    assert tax.padded(0) == ("X", "Y", "Z", "W") and tax.padded(2) == ("A", "B", "", "")
    assert tax.is_valid_path(("A", "B", "", ""))
    assert not tax.is_valid_path(("A", "Q", "", ""))


def test_taxonomy_rejects_gap(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("l1,l2,l3\nX,,Z\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_taxonomy(p, ["l1", "l2", "l3"])


def test_config_validation_errors_are_readable():
    with pytest.raises(ConfigError) as e:
        validate_config({"company_id": "Z", "reporting_currency": "USD", "taxonomy": {"file": "t.csv", "level_columns": ["a"]},
                         "sources": [{"name": "s", "file": "f.csv", "id_column": "id", "record_id_template": "{id}", "fields": {}}]})
    assert "amount" in str(e.value)
