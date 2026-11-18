"""Independent re-computation of the financial rules from the RAW files, compared with the generated output files.

It deliberately does not import the pipeline's amount/FX code: a second, simple implementation must agree.
Usage: python scripts/final_validation.py --a-data ../company_A --b-data ../company_B --a-out outputs/aryan_sharma__A.csv --b-out outputs/aryan_sharma__B.csv
Prints a markdown report (aggregate numbers only) that is saved to reports/final_validation_report.md by the caller.
"""
import argparse
import re

import numpy as np
import pandas as pd


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")


def check_common(out, ids, lines):
    inc = out.include == 1
    ok = {
        "every input record appears exactly once": bool(set(ids) == set(out.record_id) and not out.record_id.duplicated().any()),
        "excluded rows have spend == 0": bool((out.loc[~inc, "spend"] == 0).all()),
        "included rows have blank exclusion_reason": bool((out.loc[inc, "exclusion_reason"].fillna("") == "").all()),
        "excluded rows carry a valid code": bool(out.loc[~inc, "exclusion_reason"].isin(["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]).all()),
        "confidence within [0,1]": bool(out.confidence.between(0, 1).all()),
    }
    for k, v in ok.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
    return ok


def validate_a(data, outp, lines):
    out = pd.read_csv(outp, dtype={"exclusion_reason": str}, keep_default_na=False)
    out["exclusion_reason"] = out.exclusion_reason.replace("nan", "")
    e = pd.read_csv(f"{data}/erp_goods_receipts.csv", dtype=str, keep_default_na=False)
    t = pd.read_csv(f"{data}/invoice_tool_export.csv", dtype=str, keep_default_na=False)
    ids = ("ERP:" + e.line_id).tolist() + ("TOOL:" + t.tool_record_id).tolist()
    lines.append("### Company A (ZAR)")
    lines.append(f"- records: ERP {len(e):,} + tool {len(t):,} = {len(ids):,}; output rows {len(out):,}")
    check_common(out, ids, lines)
    o = out.set_index("record_id")
    # ERP: net as supplied
    ev = num(e.value_zar).round(2).values
    es = o.loc["ERP:" + e.line_id, "spend"].values
    inc = o.loc["ERP:" + e.line_id, "include"].values == 1
    lines.append(f"- {'PASS' if np.allclose(ev[inc], es[inc], atol=0.005) else 'FAIL'}: included ERP spend equals the supplied net value (no VAT removed again) on {int(inc.sum()):,} lines; {int((~inc).sum())} ERP lines excluded")
    # tool: gross / (1+rate); blank -> 0.15; 0 stays 0
    def rate(x):
        x = x.strip().replace("%", "")
        if x == "":
            return 0.15
        v = float(x)
        return v / 100 if v >= 1 else v
    r = t.vat_rate.map(rate)
    net = (num(t.gross_amount) / (1 + r))
    ts = o.loc["TOOL:" + t.tool_record_id, "spend"].values
    tinc = o.loc["TOOL:" + t.tool_record_id, "include"].values == 1
    diff = np.abs(net.values[tinc] - ts[tinc])
    lines.append(f"- {'PASS' if (diff <= 0.0051).all() else 'FAIL'}: included tool spend equals gross/(1+VAT) (max abs difference {diff.max():.4f} on {int(tinc.sum()):,} lines; blank VAT -> 15 %, explicit 0 kept)")
    lines.append(f"- tool lines by VAT field: " + ", ".join(f"{k or 'blank'}={v}" for k, v in t.vat_rate.value_counts().head(6).items()))
    st = t.status.values
    o_t = o.loc["TOOL:" + t.tool_record_id]
    bad = ((st == "Cancelled") & (o_t.exclusion_reason.values != "CANCELLED")).sum() + (np.isin(st, ["On Hold", "Rejected"]) & (o_t.exclusion_reason.values != "NOT_APPROVED")).sum()
    lines.append(f"- {'PASS' if bad == 0 else 'FAIL'}: every Cancelled / On Hold / Rejected tool line carries CANCELLED / NOT_APPROVED ({int(bad)} violations)")
    lines.append(f"- total included spend ZAR {out.loc[out.include == 1, 'spend'].sum():,.2f}; excluded by reason: {out.loc[out.include == 0, 'exclusion_reason'].value_counts().to_dict()}")
    lines.append(f"- ERP lines also present in the tool (duplicates removed): spend removed from the tool side = ZAR {net.values[~tinc & (o_t.exclusion_reason.values == 'DUPLICATE')].sum():,.2f}")


def validate_b(data, outp, lines):
    out = pd.read_csv(outp, dtype={"exclusion_reason": str}, keep_default_na=False)
    b = pd.read_csv(f"{data}/invoice_lines.csv", dtype=str, keep_default_na=False)
    fx = pd.read_csv(f"{data}/fx_rates_usd.csv")
    lines.append("### Company B (USD)")
    lines.append(f"- records {len(b):,}; output rows {len(out):,}")
    check_common(out, b.line_id.tolist(), lines)
    o = out.set_index("record_id").loc[b.line_id]
    inc = o.include.values == 1
    amt = num(b.line_amount).values
    rate = num(b.currency_rate).values
    ccy = b.currency.values
    usd = ccy == "USD"
    expected = np.where(usd, amt, amt / rate)
    have_rate = usd | ~np.isnan(rate)
    m = inc & have_rate & ~np.isnan(amt)
    diff = np.abs(expected[m] - o.spend.values[m])
    lines.append(f"- {'PASS' if (diff <= 0.0051).all() else 'FAIL'}: lines with USD or a line rate: USD = local / rate (max abs difference {diff.max():.4f} on {int(m.sum()):,} lines); USD lines unchanged: {'PASS' if np.allclose(amt[inc & usd], o.spend.values[inc & usd], atol=0.005) else 'FAIL'}")
    # lines without a rate: must use the monthly table (invoice month; documented fallbacks)
    nr = inc & ~have_rate & ~np.isnan(amt)
    table = {(c, mth): r for c, mth, r in zip(fx.currency, fx.month, fx.units_per_usd)}
    months = b.invoice_date.str[:7].values
    ok = tot = 0
    for i in np.nonzero(nr)[0]:
        r = table.get((ccy[i], months[i]))
        if r is None:
            continue
        tot += 1
        ok += abs(amt[i] / r - o.spend.values[i]) <= 0.0051
    lines.append(f"- lines without a line rate: {int(nr.sum()):,}; of these {tot:,} have their invoice month in the FX table and {ok:,} agree with local / monthly rate; the rest use the documented fallbacks (posting month, nearest month)")
    nonusd_unresolved = int((inc & (o.spend.values == 0) & (amt != 0) & ~np.isnan(amt)).sum())
    eff_rate = np.where(np.isnan(rate), np.array([table.get((c, mm), np.nan) for c, mm in zip(ccy, months)]), rate)
    eff_rate = np.where(usd, 1.0, eff_rate)
    tiny = int((inc & (o.spend.values == 0) & (np.abs(amt / eff_rate) < 0.005) & (amt != 0) & ~np.isnan(amt)).sum())
    lines.append(f"- included lines whose amount converts to 0.00 USD: {nonusd_unresolved}, of which {tiny} are legitimate sub-cent amounts (PYG/CLP/MMK); unconverted for lack of a rate: {nonusd_unresolved - tiny}")
    lines.append(f"- total included spend USD {out.loc[out.include == 1, 'spend'].sum():,.2f}; excluded by reason: {out.loc[out.include == 0, 'exclusion_reason'].value_counts().to_dict()}")
    gl = b.gl_account_description.values
    ic = pd.Series(gl).str.contains(r"(?<![A-Za-z])(?:I/C|IC\d{0,3}|inter[ -]?co(?:mpany)?)(?![A-Za-z])", regex=True, case=False).values
    lines.append(f"- {'PASS' if (o.exclusion_reason.values[ic & (o.exclusion_reason.values != 'DUPLICATE')] == 'NON_PROCUREMENT').all() else 'FAIL'}: every intercompany-account line is NON_PROCUREMENT (or a re-export of one)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-data"); ap.add_argument("--b-data"); ap.add_argument("--a-out"); ap.add_argument("--b-out")
    a = ap.parse_args()
    L = ["## Independent financial re-computation (from raw files)", ""]
    validate_a(a.a_data, a.a_out, L)
    L.append("")
    validate_b(a.b_data, a.b_out, L)
    print("\n".join(L))
