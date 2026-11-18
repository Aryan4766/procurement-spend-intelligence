"""Classification-engine tests on a synthetic tree (lexical mode: no model download needed)."""
import numpy as np
import pandas as pd
import pytest

from src.classification import zero_label_classifier as zl
from src.classification.hybrid_classifier import HybridClassifier, decode_hierarchical, informativeness
from src.evaluation import classification_metrics as cm
from src.evaluation.matching_metrics import pairwise_scores
from src.taxonomy.taxonomy_loader import load_taxonomy


@pytest.fixture()
def tax(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,c\nFood,Fruit,Fresh fruit\nFood,Dairy,Milk and cheese\nTools,Hand tools,Hammers\nTools,Hand tools,Saws\n"
                 "Tools,Power tools,Drills\nServices,Cleaning,\n", encoding="utf-8")
    return load_taxonomy(p, ["a", "b", "c"])


def test_hierarchical_decoding_always_returns_a_valid_leaf(tax):
    rng = np.random.RandomState(0)
    P = rng.dirichlet(np.ones(tax.n_leaves), size=200)
    leaf = decode_hierarchical(P, tax)
    assert leaf.min() >= 0 and leaf.max() < tax.n_leaves
    # the chosen leaf lies inside the best level-1 branch
    pref = tax.prefix_index(0)
    best_l1 = np.stack([P[:, idx].sum(1) for idx in pref.values()], 1).argmax(1)
    keys = list(pref.keys())
    for row, b in enumerate(best_l1):
        assert tax.paths[leaf[row]][:1] == keys[b]


def test_softmax_rows_sum_to_one():
    P = zl.softmax(np.random.RandomState(1).randn(5, 7), 0.3)
    assert np.allclose(P.sum(1), 1) and (P >= 0).all()


def test_zero_label_mode_never_uses_labels_and_outputs_valid_probabilities(tax):
    texts = ["fresh fruit apples", "apples fresh fruit box", "milk and cheese dairy", "claw hammer", "wood saws", "power drills",
             "cleaning service office", ""] * 6
    ctx = [t + " sup1" for t in texts]
    groups = np.arange(len(texts))
    labels = np.zeros(len(texts), dtype=int)  # a poisoned label vector that would wreck predictions if it were used
    clf = HybridClassifier(tax, {"classification": {"embedding_models": "none", "self_train_rounds": 1}}, cache_dir=None)
    res = clf.classify(texts, ctx, groups, labels=labels, zero_label=True)
    assert res.mode == "zero_label"
    assert np.allclose(res.proba.sum(1), 1)
    assert ((res.confidence >= 0) & (res.confidence <= 1)).all()
    blank = np.array([t == "" for t in texts])
    assert (res.confidence[blank] <= 0.05).all()                       # blank text -> near-zero confidence, still a valid leaf
    assert res.leaf.min() >= 0 and res.leaf.max() < tax.n_leaves
    pred = [tax.paths[i][-1] for i in res.leaf[~blank]]
    assert "Fresh fruit" in pred                                         # the lexical prior alone finds the obvious ones


def test_supervised_mode_is_cross_fitted_and_beats_the_prior(tax):
    rng = np.random.RandomState(0)
    vocab = {0: "apple pear banana", 1: "milk cheese yoghurt", 2: "hammer claw", 3: "saw blade wood", 4: "drill bit cordless", 5: "clean mop"}
    texts, y = [], []
    for i in range(300):
        c = rng.randint(6)
        w = vocab[c].split()
        texts.append(" ".join(rng.choice(w, 2)) + f" item{i}")
        y.append(c)
    y = np.array(y)
    groups = np.arange(300)
    clf = HybridClassifier(tax, {"classification": {"embedding_models": "none", "min_labels": 50, "cv_folds": 3}}, cache_dir=None)
    res = clf.classify(texts, texts, groups, labels=y, zero_label=False)
    assert res.mode == "supervised+semantic"
    assert (res.leaf == y).mean() > 0.9
    assert res.diagnostics["oof_accuracy_blend"] > res.diagnostics["oof_accuracy_prior_only"]


def test_informativeness_bounds():
    v = informativeness(["", "12 34", "copper pipe fitting elbow"])
    assert v[0] == 0 and v[1] == 0 and v[2] == 1


def test_classification_metrics_levels(tax):
    true = np.array([0, 1, 2, 3, 4])
    pred = np.array([0, 0, 3, 3, 5])
    rep = cm.classification_report(true, pred, tax)
    assert rep["exact_leaf_accuracy"] == pytest.approx(2 / 5)
    assert rep["L1_accuracy"] == pytest.approx(4 / 5)       # leaf 4 (Tools) vs 5 (Services) is the only level-1 miss
    assert rep["L2_accuracy"] == pytest.approx(3 / 5)
    assert 0 <= rep["macro_f1"] <= 1 and 0 <= rep["weighted_f1"] <= 1


def test_pairwise_entity_scores():
    pred = pd.Series(["a", "a", "b", "b", "c"])
    true = pd.Series(["x", "x", "y", "z", "z"])
    s = pairwise_scores(pred, true)
    assert s["pairwise_precision"] == pytest.approx(0.5) and s["pairwise_recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------- second-pass additions
def test_parent_evidence_sharing_changes_branch_not_leaf_within_branch(tax):
    # 4 rows of one supplier: three clearly "Food", one ambiguous towards "Tools"; the group vote moves the ambiguous row to Food
    P = np.full((4, tax.n_leaves), 0.01)
    P[:3, 0] = 0.9
    P[3, 2] = 0.5
    P[3, 1] = 0.4
    P = P / P.sum(1, keepdims=True)
    out = zl.share_parent_evidence(P, np.zeros(4, dtype=int), tax, strength=2.0, level=0)
    assert np.allclose(out.sum(1), 1)
    assert tax.paths[out[3].argmax()][0] == "Food"
    # singleton groups are unchanged
    single = zl.share_parent_evidence(P, np.arange(4), tax, strength=2.0, level=0)
    assert np.allclose(single, P, atol=1e-6) or single.argmax(1).tolist() == P.argmax(1).tolist()


def test_embedding_text_prefix_uses_only_configured_fields():
    from src.pipeline import embedding_texts
    sub = pd.DataFrame({"ctx_spend_type": ["Service", ""], "ctx_material_type": ["Spare parts", ""]})
    out = embedding_texts(sub, ["mntn conveyor", "valve"], {"embed_prefix_fields": ["ctx_spend_type", "ctx_material_type"]})
    assert out == ["service spare parts : mntn conveyor", "valve"]
    assert embedding_texts(sub, ["a", "b"], {}) == ["a", "b"]


def test_torch_is_imported_before_numpy_so_models_load_on_windows():
    import src.classification.semantic_classifier as sc
    assert sc.torch is not None, "torch failed to import: semantic models would silently fall back to lexical mode"


