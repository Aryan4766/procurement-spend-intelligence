"""The output validator must reject every violation of the output contract."""
import pandas as pd
import pytest

from src.evaluation.pipeline_validation import validate_output
from src.output.output_generator import OUTPUT_COLUMNS, build_output
from src.taxonomy.taxonomy_loader import load_taxonomy


@pytest.fixture()
def tax(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,c\nX,Y,Z\nX,Q,\n", encoding="utf-8")
    return load_taxonomy(p, ["a", "b", "c"])


def good():
    return pd.DataFrame([
        ["R1", 1, "", "S1", "I1", 10.5, "X", "Y", "Z", "", 0.9],
        ["R2", 1, "", "S1", "I2", -3.0, "X", "Q", "", "", 0.4],
        ["R3", 0, "DUPLICATE", "S2", "I1", 0.0, "", "", "", "", 0.95],
    ], columns=OUTPUT_COLUMNS)


def test_valid_file_passes(tax):
    assert validate_output(good(), ["R1", "R2", "R3"], tax) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.drop(index=2), "missing"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate record_id"),
    (lambda d: d.assign(include=[1, 1, 2]), "include must be"),
    (lambda d: d.assign(exclusion_reason=["", "NOT_A_CODE", "DUPLICATE"]), "blank exclusion_reason"),
    (lambda d: d.assign(spend=[10.5, -3.0, 5.0]), "spend == 0"),
    (lambda d: d.assign(confidence=[0.9, 1.4, 0.95]), "confidence"),
    (lambda d: d.assign(l3=["Wrong", "", ""]), "not in the taxonomy"),
    (lambda d: d.assign(l2=["Y", "Bogus", ""]), "not in the taxonomy"),
    (lambda d: d.assign(exclusion_reason=["", "", ""]), "valid exclusion code"),
    (lambda d: d.assign(l1=["X", "X", "X"]), "blank categories"),
])
def test_validator_catches_violation(tax, mutate, needle):
    problems = validate_output(mutate(good()), ["R1", "R2", "R3"], tax)
    assert any(needle in p for p in problems), problems


def test_build_output_blanks_excluded_rows():
    txn = pd.DataFrame({"record_id": ["a", "b"], "include": [1, 0], "exclusion_reason": ["x", "DUPLICATE"],
                        "supplier_key": ["s", "s"], "item_key": ["i", "i"], "spend_final": [12.346, 9.0],
                        "cat_l1": ["X", "X"], "cat_l2": ["Y", "Y"], "cat_l3": ["Z", "Z"], "confidence": [1.2, 0.9]})
    out = build_output(txn, 3)
    assert out.loc[0, "exclusion_reason"] == "" and out.loc[0, "confidence"] == 1.0 and out.loc[0, "spend"] == 12.35
    assert out.loc[1, "spend"] == 0 and out.loc[1, "l1"] == "" and out.loc[1, "include"] == 0
    assert list(out.columns) == OUTPUT_COLUMNS
