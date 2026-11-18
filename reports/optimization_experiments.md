# Optimization experiments (zero-label classification)

**Protocol.** Every variant is scored on the same rows against the hidden labels *after* prediction (`scripts/zero_label_experiments.py`).
A = 37,029 labelled ERP lines (43,282 included lines are classified); B = 2,547 labelled seed lines inside the 37,858 included lines.
"Proxy" = e5-small + MiniLM ensemble (text-variant experiments need re-embedding; the full 4-model ensemble is used for the final runs).
**All numbers are development numbers**: design choices were made on A and B, so they are not independent out-of-sample estimates and do not prove performance on Company C.
Leaf = exact leaf accuracy; spend-w = weighted by |spend|. Runtime of the experiments is not comparable to pipeline runtime (cached embeddings).

## Error analysis of the baseline (A, labels hidden)
* Of 37 leaves, 13 had F1 = 0 and the largest ones (equipment-specific services/spares: three categories of 2,300–3,000 lines each,
  and one of about 2,000 lines) had recall of 0–7 %. Their text is generic maintenance wording; the *equipment* is not in the text.
* Top confusions are between leaves of **different L1 branches** (roof bolts ↔ roof-bolter maintenance spares: 1,510 lines; road material ↔ stone dust: 1,418), i.e. wrong parent.
* Predictive value of single metadata fields (leaf purity / L1 purity of the field's majority category, labels used for this *diagnostic only*):
  supplier 78 % / **98 %**, contract number 69 % / 74 %, first description word 66 % / 84 %, material number 60 % / 73 %, UoM 22 % / 50 %, material type 19 % / 52 %,
  spend type 20 % / 54 %, cost centre 19 % / 39 %, site 16 % / 51 %, opex/capex 9 % / 38 %, PO item 9 % / 36 %. Supplier is by far the strongest structure; **account/cost-centre/site fields are not informative here**.
* Abbreviations: SAP short texts are `NOUN QUALIFIER;spec` with truncations (`DRIL`, `RCK`, `CYLDR`). Generic expansion is ambiguous without a dictionary
  (`EXPL` = explosive/explosion/explore), so no expansion table was built; E2/E3 below test cheaper alternatives.

## Results
| # | Experiment | Data | Leaf | L1 | L2 | Leaf spend-w | L1 spend-w | Macro-F1 | Decision |
|---|---|---|---|---|---|---|---|---|---|
| E0a | Embeddings only, best single model (e5-small, 1 model, no self-training) | A | 7.2 % | 32.5 % | – | 15.0 % (row-w) | – | – | baseline of the first design |
| E0b | Fine-tune encoder on B labels, test zero-label on A | A | 6.2 % (vs 6.3 % untuned) | 37.9 % | – | – | – | – | **rejected**: no transfer |
| E1 | Baseline (lexical + embedding priors, each self-trained, averaged) — full ensemble | A | 17.8 % | 43.4 % | 25.3 % | 28.1 % | 46.9 % | 0.146 | starting point |
| E1 | same | B | 62.2 % | 81.1 % | 80.0 % | 86.5 % | 96.6 % | 0.503 | starting point |
| E1p | Baseline, proxy ensemble | A | 17.8 % | 45.3 % | 24.5 % | 28.2 % | 46.7 % | 0.145 | reference for E2/E3/E5/E6 |
| E1p | Baseline, proxy ensemble | B | 53.3 % | 72.6 % | 71.1 % | 57.8 % | 72.5 % | 0.495 | reference |
| E2 | Drop alphanumeric part codes from the embedded text | A | 17.0 % | 44.6 % | 23.1 % | 28.3 % | 47.8 % | 0.136 | **rejected** (A −0.8 pt; B leaf 53.3→40.2 %) |
| E3 | Head phrase repeated + full text | A | 15.5 % | 42.4 % | 20.3 % | 25.6 % | 45.3 % | 0.122 | **rejected** (A and B worse) |
| E5 | Metadata prefix (`service spare parts : text`) for the priors | A | 18.3 % | 53.3 % | 21.4 % | 26.3 % | 62.0 % | 0.144 | **retained** for A (config `embed_prefix_fields`) |
| E6 | Share evidence over the *whole leaf distribution* by supplier, s = 0.8 | A | 16.3 % | 56.8 % | 33.1 % | 25.6 % | 62.1 % | 0.128 | rejected (leaf drops) |
| E6 | same | B | 37.1 % | 60.3 % | 57.8 % | 30.2 % | 46.4 % | 0.362 | **rejected**: −16 pt on B |
| E6d | Share evidence for the **parent (L1) branch only**, supplier, s = 1 | A | 17.8 % | 53.1 % | 27.8 % | 27.6 % | 60.4 % | – | **retained** |
| E6d | same | B | 64.4 % | 84.8 % | 83.0 % | 73.4 % | 88.4 % | – | **retained** (+11 pt leaf) |
| E6e | parent sharing at L1 **and** L2 | A / B | 16.3 % / 63.3 % | 54.1 % / 83.6 % | – | 27.2 % / 71.6 % | – | – | rejected (no gain over L1 only) |
| E5+E6 | metadata prefix + supplier sharing 0.8 (full-leaf) | A | 22.3 % | 67.3 % | 28.9 % | 27.1 % | 77.5 % | 0.167 | superseded by E5+E6d |
| E9 | Lexical prior alone / embedding prior alone / their mean prior, each self-trained (full B file) | B | 52.8 % / 41.8 % / 37.4 % | 70.6 % / 62.4 % / 59.2 % | – | 81.2 % / 33.7 % / 44.7 % | 91.5 % / 45.3 % / 60.2 % | – | **average of the two self-trained priors retained** (62.2 % leaf) |
| E-LR | Logistic regression inside self-training | B | same within 0.3 pt | | | | | | rejected: >1 h per run; SGD retained |
| **Final** | E1 + E5 (A) + E6d (A, B) — full 4-model ensemble | **A** | **24.0 %** | **65.4 %** | 30.7 % | **30.6 %** | **77.1 %** | **0.198** | **retained** |
| **Final** | same | **B** | **65.8 %** | **86.3 %** | 84.5 % | **87.8 %** | **97.9 %** | **0.525** | **retained** |

Model comparison on B seeds (unique texts, zero-label prior only, leaf): multilingual-e5-base 43.2 %, e5-small 36.9 %, MiniLM-L6 34.2 %, bge-small 31.4 %, paraphrase-multilingual-mpnet 30.1 %.
Larger models (e5-large, bge-m3) were not tried: the development machine has 7 GB RAM; the brief's time limit and the "no model without measured benefit" rule keep the 4-model ensemble.

Retained / rejected summary: **retained** metadata prefix (A), parent-only supplier evidence sharing (A and B), dual-prior self-training, SGD self-training, hierarchical top-down decoding.
**Rejected** encoder fine-tuning, code stripping, head-phrase repetition, full-leaf sharing, L2 sharing, LR inside self-training.
Hierarchical vs flat decoding is a built-in switch (`classification.hierarchical_decode`); on supervised runs the two differ by < 0.3 pt, so the hierarchical (always-valid-path) option stays on.

## Confidence
Features: top probability, margin, agreement of the two self-trained priors, text informativeness. Pooled fit on A+B (in-sample ECE 0.187/0.186);
**leave-one-company-out ECE is 0.37–0.38**: absolute calibration does not transfer between companies whose accuracy differs this much (24 % vs 66 %).
Ranking quality is usable: AUC of confidence vs correctness = 0.65 (A) / 0.78 (B); reliability tables are in `reports/confidence_fit.txt`. Interpret zero-label confidence as a **ranking signal**, not a probability.
Supervised runs use isotonic calibration on out-of-fold correctness (ECE ≈ 0.00 in-sample on the calibration data).
