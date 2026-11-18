"""Per-source financial normalisation: parse amounts and dates, remove VAT where gross, convert to reporting currency."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.amount_normalizer import (parse_amount, parse_rate_fraction, quantize_money, remove_tax)
from src.preprocessing.date_normalizer import flag_out_of_range, parse_dates
from src.spend.currency_converter import FxTable, convert_to_reporting


def normalise_dates(std: pd.DataFrame, src: dict) -> pd.DataFrame:
    """Parse every configured date field; flag (never drop) missing / unparseable / implausible values."""
    sanity = src.get("date_sanity") or {}
    formats = src.get("date_formats", "auto")
    date_fields = [f for f in ("date", "invoice_date", "secondary_date") if f in std.columns]
    for f in date_fields:
        fmts = (formats.get(f, formats.get("default", "auto")) if isinstance(formats, dict) else formats)
        parsed, flag = parse_dates(std[f], fmts)
        std[f + "_parsed"] = parsed
        std[f + "_flag"] = flag
    ref = sanity.get("reference_field", "date")
    gap = sanity.get("max_gap_days")
    if gap and (ref + "_parsed") in std.columns:
        for f in date_fields:
            if f == ref:
                continue
            bad = flag_out_of_range(std[f + "_parsed"], std[ref + "_parsed"], int(gap))
            std.loc[bad, f + "_flag"] = "OUT_OF_RANGE"
    return std


def compute_spend(std: pd.DataFrame, src: dict, cfg: dict, fx: FxTable) -> pd.DataFrame:
    """Add: amount_local (gross or net as supplied), vat_rate, net_local, fx_rate, spend_calc, fin_flag."""
    amt = src.get("amount") or {}
    reporting = cfg["reporting_currency"].upper()
    basis = amt.get("basis", "net")
    default_vat = Decimal(str(amt["vat_default"])) if amt.get("vat_default") is not None else None
    semantics = (cfg.get("fx") or {}).get("semantics", "units_per_reporting")
    fx_date_fields = amt.get("fx_date_fields", ["invoice_date", "date"])

    n = len(std)
    raw_amounts = std["amount"].tolist()
    vat_raw = std["vat_rate"].tolist() if "vat_rate" in std.columns else [""] * n
    ccys = std["_currency_raw"].tolist()
    rates_raw = std["_rate_raw"].tolist()
    month_cols = []
    for f in fx_date_fields:
        if f + "_parsed" in std.columns:
            ok = std[f + "_flag"].eq("OK")
            m = std[f + "_parsed"].dt.strftime("%Y-%m").where(ok, None)
            month_cols.append(m.tolist())

    gross_l, net_l, vat_l, rep_l, rate_l, flag_l, taxflag_l = [], [], [], [], [], [], []
    for i in range(n):
        a = parse_amount(raw_amounts[i])
        if a is None:
            gross_l.append(None); net_l.append(None); vat_l.append(None); rep_l.append(None); rate_l.append(None)
            flag_l.append("AMOUNT_UNPARSEABLE" if str(raw_amounts[i]).strip() else "AMOUNT_BLANK")
            taxflag_l.append("")
            continue
        if basis == "gross":
            rate, tf = parse_rate_fraction(vat_raw[i], default_vat)
            net = remove_tax(a, rate)
            vat_l.append(rate)
        else:
            net, tf = a, "NET_AS_SUPPLIED"
            vat_l.append(None)
        taxflag_l.append(tf)
        gross_l.append(a)
        net_l.append(net)
        lr = parse_amount(rates_raw[i]) if rates_raw[i] != "" else None
        months = [mc[i] for mc in month_cols]
        rep, used, flag = convert_to_reporting(net, ccys[i], reporting, lr, fx, months, semantics)
        rep_l.append(quantize_money(rep))
        rate_l.append(used)
        flag_l.append(flag)
    std["amount_supplied"] = [None if x is None else float(x) for x in gross_l]
    std["vat_fraction"] = [None if x is None else float(x) for x in vat_l]
    std["net_local"] = [None if x is None else float(quantize_money(x)) for x in net_l]
    std["fx_rate"] = [None if x is None else float(x) for x in rate_l]
    std["spend_calc"] = [None if x is None else float(x) for x in rep_l]
    std["fin_flag"] = flag_l
    std["tax_flag"] = taxflag_l
    return std
