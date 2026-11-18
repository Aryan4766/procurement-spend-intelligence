# APPROACH.md (one page)

**Pipeline.** One YAML per company (`configs/`, shared defaults in `common.yaml`) drives `python main.py --config … --data-dir …`:
ingest (CSV as text, column mapping, schema check) → clean (exact `Decimal` amounts; VAT removed only from gross sources at the rate shown, blank → 15 %, explicit 0 kept;
FX = local ÷ rate: line rate → monthly table for the invoice month → posting month → nearest month, flagged; every date format parsed or flagged) →
exclusions (`CANCELLED` > `NOT_APPROVED` > `NON_PROCUREMENT` > `DUPLICATE`) → cross-system matching → supplier and item de-duplication → classification → validated 11-column output.
A new company = a new YAML; no category name appears in code. Tests: 81 passed, 3 warnings in 5.02s.

**Matching and de-duplication (A).** A tool line duplicates an ERP line only if the **net amount** agrees (±0.05) **and** (same normalised PO, or supplier-similarity ≥ 0.8 with equal quantity/text) — never on text or amount alone. Pairs are
scored (amount, PO, supplier, quantity, text, date), assigned one-to-one greedily (ERP kept, tool line `DUPLICATE`); one tool line may settle several ERP lines of a PO (subset-sum); re-exports with different invoice numbers
are detected from amount+quantity+supplier+text+PO. Result on A: 20,476 tool lines matched one-to-one to ERP lines, 4 aggregates, 571 copies of an already-consumed ERP line, 713 tool re-exports, 0 ambiguous; 6,756 tool lines are tool-only (incl. cancelled/not-approved); matched-pair score median 1.000, 5th percentile 0.993. B: exact in-source re-exports (677 lines) → `DUPLICATE`; 450/450 analyst non-procurement seeds reproduced, 0 false positives.
**Suppliers**: normalised compact names (legal suffix, spacing, OCR digits) + vendor id + same-PO evidence, fuzzy ≥ 92 only for names ≥ 7 characters: purity/completeness 1.00/1.00 (pairwise P/R 1.00/1.00; with the bucket recall is only 0.10) against A's golden keys (excluding the golden "other" bucket of 122 names).
**Items**: item code authoritative; code-less lines grouped by near-identical text within a supplier **and** equal numeric specification: pairwise precision/recall 1.00/0.987 (with the catch-all bucket: 1.00/0.37) on A (excluding the golden catch-all bucket).

**Classifying into an unseen tree.** The tree is read at run time. Zero-label mode: local sentence encoders (multilingual-e5 base+small, MiniLM, bge-small; z-scored ensemble) score each description against every taxonomy *path* text → probability over leaves;
a lexical (char-n-gram) prior scores the same paths; each prior is refined by label-free self-training (confident pseudo-labels train a TF-IDF model on description + supplier/account context, 3 rounds) and the two are averaged; a supplier's lines then vote on the level-1 branch only (leaf choice stays per line); an optional metadata prefix (e.g. spend type) is added to the embedded text; top-down hierarchical decoding guarantees a valid path; confidence = logistic model on
[top probability, margin, signal agreement, text informativeness] whose coefficients were fitted on the development companies and reused. With labels (A, B, A2, B2) supervised TF-IDF/embedding models are cross-fitted (GroupKFold) and blended with the prior.

**Accuracy measured on A with labels hidden** (`--zero-label`; labels read only afterwards): exact leaf **24.0 %** (spend-weighted 30.6 %), L1 65.4 %, L2 30.7 %, macro-F1 0.198, weighted-F1 0.189.
Honest reading (second optimisation pass raised A from 17.8 % leaf / 43.4 % L1; B from 62.2 % / 81.1 %; experiments in reports/optimization_experiments.md): A's SAP short-texts (`ROD DRIL;SIZE 35 MM`) are far from the category wording, so zero-label accuracy is low; B (natural language) reaches leaf 65.8 % / L1 86.3 % (spend-weighted L1 97.9 %).
These are development numbers (design choices were made on A/B). Supervised accuracy (out-of-fold): A 92.5 % leaf / 98.4 % L1; B 88.7 % / 93.1 %.

**Runtime** (4-core CPU laptop, no GPU, cached models): A: 65,549 records in 10.1 min; B: 41,028 records in 1.2 min; A zero-label: 65,549 records in 1.6 min; B zero-label: 41,028 records in 1.2 min — well inside 60 min per 25,000 lines.
**LLM / API cost**: none used; no API key is needed; nothing leaves the machine (embedding models are downloaded once from Hugging Face). Cost per 1,000 lines: 0. Tools used to build: Claude Code (AI assistant) for code drafting; all results measured by running the code.
**Limitations**: zero-label accuracy on jargon-heavy text; non-procurement for an unseen company relies on generic rules; golden supplier/item "bucket" keys cannot be reproduced.
