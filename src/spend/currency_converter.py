"""FX handling. Convention (from the brief): rate = units of the line currency per 1 unit of reporting currency.

    reporting_amount = local_amount / rate

A rate is never invented: line rate -> monthly table rate (invoice month, then fallbacks, all flagged) -> unresolved.
"""
from __future__ import annotations

import bisect
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.data_loader import read_csv_text
from src.preprocessing.amount_normalizer import parse_amount


class FxTable:
    def __init__(self, rates: dict[tuple[str, str], Decimal], semantics: str = "units_per_reporting"):
        self.rates = rates
        self.semantics = semantics
        self._months: dict[str, list[str]] = {}
        for (ccy, month) in rates:
            self._months.setdefault(ccy, []).append(month)
        for ccy in self._months:
            self._months[ccy].sort()

    @classmethod
    def from_config(cls, cfg: dict, data_dir: Path) -> "FxTable":
        fx = cfg.get("fx")
        if not fx or not fx.get("file"):
            return cls({}, (fx or {}).get("semantics", "units_per_reporting"))
        df = read_csv_text(data_dir / fx["file"])
        rates: dict[tuple[str, str], Decimal] = {}
        for ccy, month, rate in zip(df[fx["currency_column"]], df[fx["month_column"]], df[fx["rate_column"]]):
            d = parse_amount(rate)
            m = str(month).strip()[:7]
            if d is not None and d > 0:
                rates[(str(ccy).strip().upper(), m)] = d
        return cls(rates, fx.get("semantics", "units_per_reporting"))

    def lookup(self, ccy: str, month: Optional[str]) -> tuple[Optional[Decimal], str]:
        """Return (rate, flag). flag: 'FX_TABLE', 'FX_NEAREST_MONTH' or 'FX_MISSING'."""
        months = self._months.get(ccy)
        if not months:
            return None, "FX_MISSING"
        if month and (ccy, month) in self.rates:
            return self.rates[(ccy, month)], "FX_TABLE"
        if not month:
            return None, "FX_MISSING"
        i = bisect.bisect_left(months, month)
        cands = [months[j] for j in (i - 1, i) if 0 <= j < len(months)]
        best = min(cands, key=lambda m: abs((int(m[:4]) * 12 + int(m[5:7])) - (int(month[:4]) * 12 + int(month[5:7]))))
        return self.rates[(ccy, best)], "FX_NEAREST_MONTH"


def convert_to_reporting(local: Optional[Decimal], ccy: str, reporting: str, line_rate: Optional[Decimal],
                         fx: FxTable, months: list[Optional[str]], semantics: str) -> tuple[Optional[Decimal], Optional[Decimal], str]:
    """Return (reporting_amount, rate_used, fx_flag)."""
    if local is None:
        return None, None, "NO_AMOUNT"
    if ccy == reporting or ccy == "":
        return local, Decimal(1), "SAME_CCY" if ccy else "NO_CCY_ASSUMED_REPORTING"
    rate, flag = None, "LINE_RATE"
    if line_rate is not None and line_rate > 0:
        rate = line_rate
    else:
        for m in months:
            r, fl = fx.lookup(ccy, m)
            if r is not None:
                rate, flag = r, fl
                if fl == "FX_TABLE":
                    break
        if rate is None:
            return None, None, "FX_MISSING"
    amt = local / rate if semantics == "units_per_reporting" else local * rate
    return amt, rate, flag
