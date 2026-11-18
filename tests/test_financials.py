"""Unit tests: amounts, tax removal, FX, dates, references."""
from decimal import Decimal

import pandas as pd
import pytest

from src.preprocessing.amount_normalizer import parse_amount, parse_rate_fraction, quantize_money, remove_tax
from src.preprocessing.date_normalizer import ambiguous_share, parse_dates
from src.preprocessing.text_normalizer import (compact, normalize_invoice_number, normalize_reference,
                                               normalize_supplier_name)
from src.spend.currency_converter import FxTable, convert_to_reporting


@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", "1234.56"), ("1.234,56", "1234.56"), ("1 234,56", "1234.56"), ("(12.50)", "-12.50"),
    ("12.50-", "-12.50"), ("512,428.09", "512428.09"), ("2656.33", "2656.33"), ("-13827.64", "-13827.64"),
    ("1,234", "1234"), ("12,5", "12.5"), ("R 1200", "1200"),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["", "  ", "n/a", "abc", None])
def test_parse_amount_blank(raw):
    assert parse_amount(raw) is None


@pytest.mark.parametrize("raw,frac,flag", [
    ("15", "0.15", "OK"), ("0.15", "0.15", "OK"), ("15%", "0.15", "OK"), ("15.0", "0.15", "OK"),
    ("0", "0", "OK"), ("", "0.15", "DEFAULTED"), ("garbage", "0.15", "INVALID"),
])
def test_vat_rate_parsing(raw, frac, flag):
    d, f = parse_rate_fraction(raw, Decimal("0.15"))
    assert d == Decimal(frac) and f == flag


def test_net_from_gross_exact():
    net = quantize_money(remove_tax(Decimal("115.00"), Decimal("0.15")))
    assert net == Decimal("100.00")
    net = quantize_money(remove_tax(Decimal("8721.47"), Decimal("0.15")))
    assert net == Decimal("7583.89")


def test_zero_rated_line_is_not_taxed():
    assert quantize_money(remove_tax(Decimal("500.00"), Decimal("0"))) == Decimal("500.00")


def test_fx_conversion_units_per_usd():
    fx = FxTable({("EUR", "2024-03"): Decimal("0.5")})
    # 100 EUR at 0.5 EUR per USD -> 200 USD
    amt, rate, flag = convert_to_reporting(Decimal("100"), "EUR", "USD", Decimal("0.5"), fx, [], "units_per_reporting")
    assert amt == Decimal("200") and flag == "LINE_RATE"
    # no line rate -> monthly table
    amt, rate, flag = convert_to_reporting(Decimal("100"), "EUR", "USD", None, fx, ["2024-03"], "units_per_reporting")
    assert amt == Decimal("200") and flag == "FX_TABLE"
    # usd unchanged
    amt, _, flag = convert_to_reporting(Decimal("100"), "USD", "USD", Decimal("9"), fx, [], "units_per_reporting")
    assert amt == Decimal("100") and flag == "SAME_CCY"


def test_fx_missing_is_flagged_not_invented():
    fx = FxTable({("EUR", "2024-03"): Decimal("0.5")})
    amt, rate, flag = convert_to_reporting(Decimal("100"), "XXX", "USD", None, fx, ["2024-03"], "units_per_reporting")
    assert amt is None and flag == "FX_MISSING"


def test_fx_nearest_month_is_flagged():
    fx = FxTable({("EUR", "2024-01"): Decimal("0.9"), ("EUR", "2024-06"): Decimal("0.8")})
    amt, rate, flag = convert_to_reporting(Decimal("90"), "EUR", "USD", None, fx, ["2024-02"], "units_per_reporting")
    assert flag == "FX_NEAREST_MONTH" and rate == Decimal("0.9")


def test_dates_multiple_formats_and_flags():
    s = pd.Series(["19/01/2024", "2022-01-29", "18-Sep-21", "", "2024-13-45", "garbage"])
    d, f = parse_dates(s, ["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%y"])
    assert d.iloc[0] == pd.Timestamp("2024-01-19") and d.iloc[1] == pd.Timestamp("2022-01-29")
    assert d.iloc[2] == pd.Timestamp("2021-09-18")
    assert list(f) == ["OK", "OK", "OK", "MISSING", "UNPARSEABLE", "UNPARSEABLE"]


def test_ambiguous_date_share_reported():
    assert ambiguous_share(pd.Series(["02/03/2024", "27/03/2024"])) == 0.5


@pytest.mark.parametrize("raw", ["4510000018", "PO4510000018", "PO 4510000018", "po_4510000018", "PO#4510000018",
                                 "4510000018/10.0"])
def test_po_normalisation(raw):
    assert normalize_reference(raw, r"(\d{8,12})") == "4510000018"


def test_invoice_number_normalisation():
    assert normalize_invoice_number("INV/2024/437590") == "437590"
    assert normalize_invoice_number("IN0914136") == "914136"
    assert normalize_invoice_number("TAX INV 74677") == "74677"


def test_supplier_name_normalisation():
    assert compact(normalize_supplier_name("ZORVEX  TECHNOLOGIES (PTY) LTD.")) == "zorvextechnologies"
    assert compact(normalize_supplier_name("MEL DRIN (PTY) LTD")) == compact(normalize_supplier_name("MELDRIN"))
