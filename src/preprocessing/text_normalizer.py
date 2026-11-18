"""Text normalisation helpers shared by matching, entity resolution and classification."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

DEFAULT_LEGAL_SUFFIXES = {
    "pty", "ltd", "limited", "proprietary", "cc", "sa", "inc", "incorporated", "llc", "llp", "gmbh", "bv", "nv",
    "ag", "sarl", "sas", "srl", "spa", "ltda", "co", "corp", "corporation", "plc", "pvt", "private",
}


def fold(text) -> str:
    """Casefold + strip accents (NFKD) while keeping non-latin scripts intact."""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def normalize_text(text, keep_chars: str = "") -> str:
    """Lower-case, fold accents, replace punctuation by spaces, collapse whitespace. Digits are kept."""
    if text is None:
        return ""
    t = fold(text)
    if keep_chars:
        t = re.sub(r"[^\w\s" + re.escape(keep_chars) + "]", " ", t)
    else:
        t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t.replace("_", " ")).strip()


def normalize_supplier_name(name, legal_suffixes: Iterable[str] = DEFAULT_LEGAL_SUFFIXES) -> str:
    """'ZORVEX  TECHNOLOGIES (PTY) LTD.' -> 'zorvex technologies'. Legal-form tokens are dropped."""
    toks = normalize_text(name).split()
    suf = set(legal_suffixes)
    kept = [t for t in toks if t not in suf]
    return " ".join(kept) if kept else " ".join(toks)


def compact(text) -> str:
    """Remove all non-alphanumerics so 'MEL DRIN' and 'MELDRIN' compare equal."""
    return re.sub(r"[\W_]+", "", fold(text))


def normalize_reference(ref, regex: str | None = None) -> str:
    """PO / reference normalisation. 'PO#4510000018/10.0' -> '4510000018'.

    With a regex the first capture group (else the whole match) is returned. Otherwise all
    non-alphanumerics are removed and leading zeros stripped from purely numeric references.
    """
    if ref is None:
        return ""
    s = str(ref).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""
    if regex:
        m = re.search(regex, s)
        if not m:
            return ""
        return (m.group(1) if m.groups() else m.group(0)).lstrip("0")
    c = re.sub(r"[\W_]+", "", s).lower()
    return c.lstrip("0") if c.isdigit() else c


def normalize_invoice_number(ref) -> str:
    """Invoice numbers are typed 'INV/2024/437590', 'IN0914136', 'SI-2023-504376', 'TAX INV 74677', '437590'.

    The identifying core is the LAST digit run; prefixes and embedded years are formatting.
    """
    if ref is None:
        return ""
    s = str(ref).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""
    runs = re.findall(r"\d+", s)
    if not runs:
        return compact(s)
    return runs[-1].lstrip("0") or "0"


def normalize_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(normalize_text)


def clean_description(text, expand: dict | None = None) -> str:
    """Description clean-up for ML / similarity: keep numbers, units and specs; drop only separators."""
    t = normalize_text(text, keep_chars=".-/%")
    if expand:
        t = " ".join(expand.get(w, w) for w in t.split())
    return t
