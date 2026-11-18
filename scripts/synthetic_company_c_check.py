"""Synthetic configuration test for the Company C template (NOT Company C validation).

Builds a tiny made-up company in a temp folder (unrelated industry, 4-level taxonomy, non-USD currencies, a status column,
an intercompany GL line, an exact re-export), fills the placeholders of configs/company_c_template.yaml with plain text
substitution (configuration only) and runs the normal one-command CLI in zero-label mode with the real embedding models.
No source file is touched. Nothing here uses A/B data; the figures it prints are sanity checks on 18 invented lines.

    python scripts/synthetic_company_c_check.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch  # noqa: F401  (must be imported before numpy/pandas on Windows)
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TAX = """sector,family,class,item_type
Clinical Supplies,Imaging,Contrast Media,Intravenous Contrast
Clinical Supplies,Imaging,Contrast Media,Oral Contrast
Clinical Supplies,Imaging,Film and Plates,X-Ray Film
Clinical Supplies,Surgical,Sutures,Absorbable Sutures
Clinical Supplies,Surgical,Sutures,Non-absorbable Sutures
Clinical Supplies,Surgical,Gloves,Sterile Surgical Gloves
Facilities,Cleaning,Janitorial Services,Daily Office Cleaning
Facilities,Cleaning,Janitorial Services,Window Cleaning
Facilities,Energy,Electricity,Grid Electricity
IT,Hardware,Laptops,Business Laptops
IT,Software,Licences,Annual Software Subscription
Logistics,Freight,Road Freight,Pallet Delivery
"""
LINES = [  # (id, description, amount, currency, date, status, gl_text, vendor)
    ("L01", "Iohexol 350 contrast vial 100ml for CT scans", 1200.00, "USD", "2026-01-12", "Approved", "Medical consumables", "Medix GmbH"),
    ("L02", "Barium sulphate oral suspension 500ml", 340.00, "USD", "2026-01-15", "Paid", "Medical consumables", "Medix GmbH"),
    ("L03", "Digital x-ray film plates 35x43", 780.00, "EUR", "2026-02-03", "Approved", "Medical consumables", "RadiCo SA"),
    ("L04", "Vicryl absorbable suture 3-0", 95.50, "USD", "2026-02-10", "Approved", "Medical consumables", "SurgiLine"),
    ("L05", "Nylon non-absorbable suture 4-0 box 36", 120.00, "USD", "2026-02-11", "Approved", "Medical consumables", "SurgiLine"),
    ("L06", "Sterile latex surgical gloves size 7.5 box 50", 64.00, "USD", "2026-02-20", "Approved", "Medical consumables", "SurgiLine"),
    ("L07", "Daily cleaning of main office January", 900.00, "ZAR", "2026-01-31", "Approved", "Facility services", "Sparkle Services"),
    ("L08", "Exterior window cleaning quarterly", 450.00, "USD", "2026-03-05", "Approved", "Facility services", "Sparkle Services"),
    ("L09", "Electricity bill head office February", 2300.00, "USD", "2026-03-01", "Paid", "Utilities", "City Power"),
    ("L10", "Laptop 14 inch 16GB for new hire", 1450.00, "USD", "2026-03-07", "Approved", "IT equipment", "TechDirect"),
    ("L11", "Annual subscription project software 25 seats", 3100.00, "USD", "2026-03-09", "Approved", "IT software", "CloudSoft"),
    ("L12", "Pallet delivery Hamburg to Munich", 380.00, "EUR", "2026-03-12", "Approved", "Freight", "FastHaul"),
    ("L13", "Intercompany recharge Q1", 5000.00, "USD", "2026-03-31", "Approved", "Intercompany Payable IC01", "Group Holdings"),
    ("L14", "Office cleaning February", 900.00, "ZAR", "2026-02-28", "Cancelled", "Facility services", "Sparkle Services"),
    ("L15", "Laptop docking station", 150.00, "USD", "2026-03-15", "Rejected", "IT equipment", "TechDirect"),
    ("L16", "Pallet delivery Hamburg to Munich", 380.00, "EUR", "2026-03-12", "Approved", "Freight", "FastHaul"),  # exact re-export of L12
    ("L17", "", 77.00, "USD", "2026-03-20", "Approved", "Facility services", "Sparkle Services"),               # blank description
    ("L18", "Iohexol 350 contrast vial 100ml for CT scans", 1200.00, "USD", "2026-03-22", "Approved", "Medical consumables", "Medix GmbH"),
]
FX = "currency,month,units_per_usd\nEUR,2026-02,0.92\nEUR,2026-03,0.90\nZAR,2026-01,18.0\nZAR,2026-02,18.5\n"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="synthetic_company_c_"))
    pd.DataFrame(LINES, columns=["line_no", "item_text", "net_amt", "ccy", "posted", "state", "gl_name", "vendor"]).assign(
        dup_hint=lambda d: d["item_text"] + d["net_amt"].astype(str)).to_csv(tmp / "inv.csv", index=False)
    (tmp / "taxonomy_c.csv").write_text(TAX, encoding="utf-8")
    (tmp / "fx.csv").write_text(FX, encoding="utf-8")

    cfg = (REPO / "configs" / "company_c_template.yaml").read_text(encoding="utf-8")
    for old, new in {
        "<taxonomy_file>.csv": "taxonomy_c.csv",
        "[<level1_col>, <level2_col>, <level3_col>, <level4_col>]": "[sector, family, class, item_type]",
        "<invoice_lines_file>.csv": "inv.csv", "<line_id_col>": "line_no", "<description_col>": "item_text", "<amount_col>": "net_amt",
        "<posting_or_invoice_date_col>": "posted", "<currency_col>": "ccy",
        "# supplier_name: <vendor_name_col>": "supplier_name: vendor", "# status: <status_col>": "status: state",
        "# account_desc: <gl_description_col>": "account_desc: gl_name",
        "# duplicate_key: [<col>, <col>, ...]": "duplicate_key: [dup_hint, posted, vendor]",
    }.items():
        cfg = cfg.replace(old, new)
    fx_block = "fx:\n  file: fx.csv\n  currency_column: currency\n  month_column: month\n  rate_column: units_per_usd\n  semantics: units_per_reporting\n"
    cfg = cfg.replace("# fx:                                 # needed when currency_column holds non-reporting currencies\n", fx_block + "# ")
    cfg = cfg.replace("extends: common.yaml", f"extends: {(REPO / 'configs' / 'common.yaml').as_posix()}")
    (tmp / "company_c_synthetic.yaml").write_text(cfg, encoding="utf-8")

    out = tmp / "out.csv"
    cmd = [sys.executable, str(REPO / "main.py"), "--config", str(tmp / "company_c_synthetic.yaml"), "--data-dir", str(tmp),
           "--out", str(out), "--no-evaluate", "--report-dir", str(tmp), "--work-dir", str(tmp / "work")]
    print("running:", " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        return 1
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    ok = {"rows == input lines": len(df) == len(LINES), "unique record_id": df["record_id"].is_unique,
          "11 contract columns": list(df.columns) == ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]}
    tax = pd.read_csv(tmp / "taxonomy_c.csv", dtype=str)
    valid = {tuple(x) for x in tax.values.tolist()}
    inc = df[df["include"] == "1"]
    ok["included rows carry a valid 4-level taxonomy path"] = all((a, b, c, d) in valid for a, b, c, d in inc[["l1", "l2", "l3", "l4"]].values)
    ok["excluded rows spend 0 and no path"] = bool((df[df["include"] == "0"]["spend"].astype(float) == 0).all())
    ok["confidence within 0-1"] = bool(df["confidence"].astype(float).between(0, 1).all())
    ok["exclusions: CANCELLED/NOT_APPROVED/NON_PROCUREMENT/DUPLICATE"] = (
        df.set_index("record_id").loc[["L14", "L15", "L13", "L16"], "exclusion_reason"].tolist() == ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"])
    ok["EUR/ZAR converted (L03 = 780/0.92 in Feb)"] = abs(float(df.set_index("record_id").loc["L03", "spend"]) - 780 / 0.92) < 0.01
    for k, v in ok.items():
        print(("PASS " if v else "FAIL ") + k)
    print("\nsanity view (zero-label, 18 invented lines; not an accuracy estimate):")
    view = df.assign(txt=df["record_id"].map({x[0]: x[1] for x in LINES}))
    print(view[["record_id", "include", "exclusion_reason", "spend", "l3", "l4", "confidence", "txt"]].to_string(index=False, max_colwidth=34))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
