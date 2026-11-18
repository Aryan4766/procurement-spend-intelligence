"""End-to-end test on a tiny synthetic two-system company ("Company S") and a one-file company ("Company T").
Proves the pipeline is configuration-driven: no code differs between them, only YAML."""
import textwrap

import pandas as pd
import pytest

from src.config.config_loader import ConfigError
from src.evaluation.pipeline_validation import validate_output
from src.pipeline import run_company

TAXONOMY = "l1,l2\nFood,Fruit\nFood,Dairy\nTools,Hand tools\nTools,Power tools\nServices,Cleaning\n"


def _write(tmp, name, text):
    (tmp / name).write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


@pytest.fixture()
def company_s(tmp_path):
    _write(tmp_path, "tax.csv", TAXONOMY)
    _write(tmp_path, "gr.csv", """\
        gid,po,mat,txt,qty,val,vend
        1,4510000001,M1,apples red,10,1000.00,Fruity Ltd
        2,4510000002,M2,milk 2l,20,400.00,Dairy Co
        3,4510000003,M3,hammer steel,5,250.00,Tools Inc
        4,4510000004,M4,drill cordless,2,800.00,Tools Inc
        """)
    _write(tmp_path, "inv.csv", """\
        iid,inv,sup,poref,txt,qty,gross,vat,st
        T1,INV/1,FRUITY PTY LTD,PO#4510000001,apples red,10,"1,150.00",15%,Paid
        T2,INV/2,DAIRY CO,4510000002/10.0,milk 2l,20,460.00,,Approved
        T3,INV/3,Cleanly Services,,office cleaning january,1,1150.00,15,Approved
        T4,INV/4,Cleanly Services,,office cleaning february,1,1150.00,15,Cancelled
        T5,INV/5,Cleanly Services,,office cleaning march,1,1150.00,15,On Hold
        T6,INV/6,Tools Inc,4510000003,hammer steel,5,250.00,0,Paid
        T7,INV/7,Tools Inc,4510000003,hammer steel,5,250.00,0,Paid
        """)
    _write(tmp_path, "cfg.yaml", """\
        company_id: S
        reporting_currency: ZAR
        data_dir: .
        taxonomy: {file: tax.csv, level_columns: [l1, l2]}
        sources:
          - name: erp
            role: primary
            file: gr.csv
            id_column: gid
            record_id_template: "ERP:{id}"
            fields: {description: txt, supplier_name: vend, item_code: mat, quantity: qty, po: po, amount: val}
            amount: {basis: net, currency: ZAR}
          - name: tool
            role: secondary
            file: inv.csv
            id_column: iid
            record_id_template: "TOOL:{id}"
            fields: {description: txt, supplier_name: sup, quantity: qty, po: poref, invoice_number: inv, amount: gross, vat_rate: vat, status: st}
            amount: {basis: gross, vat_default: 0.15, currency: ZAR}
            status: {approved: [approved, paid], cancelled: [cancelled]}
        matching: {enabled: true, po_regex: '(\\d{8,12})', amount_tolerance_abs: 0.05, supplier_match_min: 0.8, text_match_min: 0.55}
        classification: {embedding_models: none, self_train: false}
        """)
    return tmp_path


def test_two_system_company_end_to_end(company_s):
    d = run_company(str(company_s / "cfg.yaml"), out_path=str(company_s / "out.csv"), evaluate=False, zero_label=True)
    out = d["_out"].set_index("record_id")
    assert len(out) == 11 and not out.index.duplicated().any()           # every input record exactly once
    assert out.loc["ERP:1", "include"] == 1 and out.loc["ERP:1", "spend"] == 1000.00
    assert out.loc["TOOL:T1", "exclusion_reason"] == "DUPLICATE" and out.loc["TOOL:T1", "spend"] == 0
    assert out.loc["TOOL:T2", "exclusion_reason"] == "DUPLICATE"        # blank VAT -> 15 % -> 400.00 net == ERP net
    assert out.loc["TOOL:T3", "include"] == 1 and out.loc["TOOL:T3", "spend"] == 1000.00   # tool-only, net of 15 % VAT
    assert out.loc["TOOL:T4", "exclusion_reason"] == "CANCELLED"
    assert out.loc["TOOL:T5", "exclusion_reason"] == "NOT_APPROVED"
    assert out.loc["TOOL:T6", "exclusion_reason"] == "DUPLICATE"        # zero-rated: net == gross == ERP value
    assert out.loc["TOOL:T7", "exclusion_reason"] == "DUPLICATE"        # exported twice / consumed ERP line
    assert out["supplier_key"].nunique() == 4                           # Fruity, Dairy, Tools, Cleanly
    assert out.loc["ERP:1", "supplier_key"] == out.loc["TOOL:T1", "supplier_key"]
    assert out.loc["TOOL:T6", "item_key"] == out.loc["ERP:3", "item_key"]
    # total spend: 1000 + 400 + 250 + 800 (ERP) + 1000 (tool only) = 3450
    assert round(out.loc[out.include == 1, "spend"].sum(), 2) == 3450.00
    assert validate_output(d["_out"], list(out.index), d["_tax"]) == []
    # the run report states which semantic models were actually used (none here: lexical mode was configured)
    assert d["classification"]["embedding_available"] is False and d["classification"]["embedding_models_used"] == []


def test_categories_exactly_match_taxonomy(company_s):
    d = run_company(str(company_s / "cfg.yaml"), out_path=str(company_s / "out.csv"), evaluate=False, zero_label=True)
    out, tax = d["_out"], d["_tax"]
    valid = {tax.padded(i) for i in range(tax.n_leaves)}
    inc = out[out.include == 1]
    assert {(a, b) for a, b in zip(inc.l1, inc.l2)} <= valid
    assert (out.loc[out.include == 0, ["l1", "l2"]] == "").all().all()
    assert ((out.confidence >= 0) & (out.confidence <= 1)).all()


def test_new_company_needs_only_config(tmp_path):
    _write(tmp_path, "tax.csv", "a,b,c,d\nX,Y,Z,W\nX,Y,Z2,\nQ,R,,\n")
    _write(tmp_path, "lines.csv", """\
        id,desc,amt,ccy,rate,dt
        1,widget z,100.5,EUR,0.5,2024-02-03
        2,widget z2,50,USD,,2024-02-04
        3,gadget r,10,EUR,,2024-02-03
        """)
    _write(tmp_path, "fx.csv", "currency,month,units_per_usd\nEUR,2024-02,0.8\n")
    _write(tmp_path, "cfg.yaml", """\
        company_id: C
        reporting_currency: USD
        data_dir: .
        taxonomy: {file: tax.csv, level_columns: [a, b, c, d]}
        fx: {file: fx.csv, currency_column: currency, month_column: month, rate_column: units_per_usd, semantics: units_per_reporting}
        sources:
          - name: ap
            file: lines.csv
            id_column: id
            record_id_template: "{id}"
            fields: {description: desc, amount: amt, date: dt}
            amount: {basis: net, currency_column: ccy, rate_column: rate, fx_date_fields: [date]}
        classification: {embedding_models: none, self_train: false}
        """)
    d = run_company(str(tmp_path / "cfg.yaml"), out_path=str(tmp_path / "o.csv"), evaluate=False, zero_label=True)
    out = d["_out"].set_index("record_id")
    assert out.loc["1", "spend"] == 201.00        # 100.5 EUR / 0.5 (line rate) = 201 USD
    assert out.loc["2", "spend"] == 50.00         # USD unchanged
    assert out.loc["3", "spend"] == 12.50         # no line rate -> monthly 0.8 -> 12.5
    assert list(out.columns)[-5:] == ["l1", "l2", "l3", "l4", "confidence"] or "l4" in out.columns


def test_missing_column_gives_readable_error(tmp_path):
    _write(tmp_path, "tax.csv", "a\nX\n")
    _write(tmp_path, "lines.csv", "id,desc\n1,x\n")
    _write(tmp_path, "cfg.yaml", """\
        company_id: C
        reporting_currency: USD
        data_dir: .
        taxonomy: {file: tax.csv, level_columns: [a]}
        sources:
          - name: ap
            file: lines.csv
            id_column: id
            record_id_template: "{id}"
            fields: {description: desc, amount: amount_missing}
            amount: {basis: net, currency: USD}
        """)
    with pytest.raises(ConfigError) as e:
        run_company(str(tmp_path / "cfg.yaml"), evaluate=False, zero_label=True)
    assert "amount_missing" in str(e.value)
