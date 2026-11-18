"""Date parsing: configured formats first, deterministic inference second, never silent guessing."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

_BLANK = {"", "nan", "none", "null", "nat", "0000-00-00"}

CANDIDATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%m-%d-%Y",
    "%d-%b-%y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d/%m/%y", "%Y-%m-%d %H:%M:%S",
]


def _try(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series, format=fmt, errors="coerce")


def infer_formats(series: pd.Series, candidates: Iterable[str] = CANDIDATE_FORMATS, min_share: float = 0.01) -> list[str]:
    """Pick the formats that parse a meaningful share of the non-blank values (most productive first).

    Day-first vs month-first is only resolved when the data can discriminate: a value such as
    27/03/2024 can only be day-first. When both orders fit every value, the day-first candidate
    comes first in `candidates` and `ambiguous_share` reports how undecidable the column was.
    """
    s = series.astype(str).str.strip()
    s = s[~s.str.lower().isin(_BLANK)]
    if s.empty:
        return []
    s = s.sample(min(len(s), 5000), random_state=0)
    scored = []
    for fmt in candidates:
        share = _try(s, fmt).notna().mean()
        if share >= min_share:
            scored.append((fmt, share))
    scored.sort(key=lambda x: -x[1])
    return [f for f, _ in scored]


def parse_dates(series: pd.Series, formats: Optional[list] = None) -> tuple[pd.Series, pd.Series]:
    """Parse with an ordered list of strptime formats (or infer them when None/'auto').

    Each value is parsed by the first format that succeeds. Returns (datetime series, flag series):
    flag is 'OK', 'MISSING' (blank) or 'UNPARSEABLE' (non-blank but no format fits, including
    impossible dates such as 2024-13-45). Rows are never dropped.
    """
    raw = series.astype(str).str.strip()
    blank = raw.str.lower().isin(_BLANK)
    if not formats or formats == "auto":
        formats = infer_formats(raw)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    todo = ~blank
    for fmt in formats:
        if not todo.any():
            break
        parsed = _try(raw[todo], fmt)
        ok = parsed.notna()
        out.loc[parsed.index[ok]] = parsed[ok]
        todo = todo & out.isna()
    flag = pd.Series("OK", index=series.index)
    flag[blank] = "MISSING"
    flag[(~blank) & out.isna()] = "UNPARSEABLE"
    return out, flag


def ambiguous_share(series: pd.Series) -> float:
    """Share of 'nn/nn/yyyy'-shaped values where both leading parts are <= 12 (order undecidable)."""
    m = series.astype(str).str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}$").dropna()
    if m.empty:
        return 0.0
    m = m.astype(int)
    return float(((m[0] <= 12) & (m[1] <= 12)).mean())


def flag_out_of_range(dates: pd.Series, reference: pd.Series, max_gap_days: int) -> pd.Series:
    """True where |date - reference| is implausibly large (e.g. invoice date 2018 for a 2024 posting)."""
    gap = (dates - reference).abs().dt.days
    return (gap > max_gap_days).fillna(False)


def to_iso(dates: pd.Series) -> pd.Series:
    return dates.dt.strftime("%Y-%m-%d").fillna("")
