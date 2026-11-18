"""Robust parsing of monetary amounts and tax rates into exact Decimals."""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional

import pandas as pd

CENT = Decimal("0.01")
_NON_NUM = re.compile(r"[^\d,.\-+()]")
_NULLS = {"", "nan", "none", "null", "n/a", "na", "-"}


def parse_amount(value) -> Optional[Decimal]:
    """Parse '1,234.56', '1.234,56', '1 234,56', '(12.50)', '12.50-', 'R 1200' -> Decimal.

    Returns None for blank / unparseable input. It never raises and never invents a number.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _NULLS:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = _NON_NUM.sub("", s).replace("(", "").replace(")", "")
    if s.endswith("-"):  # trailing minus (SAP style)
        neg, s = True, s[:-1]
    if s.startswith("-"):
        neg, s = True, s[1:]
    s = s.lstrip("+").replace("-", "")
    if s == "":
        return None
    if "," in s and "." in s:
        # the right-most separator is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        if s.count(",") > 1 or len(tail) == 3:  # 1,234 / 1,234,567 -> thousands separators
            s = s.replace(",", "")
        else:  # 12,5 / 12,50 -> decimal comma
            s = head.replace(",", "") + "." + tail
    elif s.count(".") > 1:  # 1.234.567 -> thousands separators
        s = s.replace(".", "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return -d if neg else d


def parse_amount_series(series: pd.Series) -> pd.Series:
    return series.map(parse_amount)


def parse_rate_fraction(value, default: Optional[Decimal]) -> tuple[Optional[Decimal], str]:
    """Parse a tax rate to a fraction: '15', '0.15', '15%', '15.0' -> 0.15 ; blank -> default.

    Returns (fraction, flag): flag is 'OK', 'DEFAULTED' (blank -> default) or 'INVALID' (-> default).
    An explicit 0 is a real zero-rated line and stays 0 (it is NOT treated as blank).
    """
    s = "" if value is None else str(value).strip()
    if s.lower() in _NULLS:
        return default, "DEFAULTED"
    pct = "%" in s
    d = parse_amount(s.replace("%", ""))
    if d is None or d < 0:
        return default, "INVALID"
    if pct or d >= 1:  # '15' and '15%' are percentages; '0.15' is already a fraction
        d = d / Decimal(100)
    if d >= 1:
        return default, "INVALID"
    return d, "OK"


def quantize_money(d: Optional[Decimal]) -> Optional[Decimal]:
    return None if d is None else d.quantize(CENT, rounding=ROUND_HALF_UP)


def remove_tax(gross: Optional[Decimal], rate: Optional[Decimal]) -> Optional[Decimal]:
    """net = gross / (1 + rate), kept at full precision (quantised once, by the caller)."""
    if gross is None or rate is None:
        return None
    return gross / (Decimal(1) + rate)
