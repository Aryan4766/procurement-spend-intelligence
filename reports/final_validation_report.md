# Final validation report

Generated from the final runs in `reports/run_*.json` and by `scripts/final_validation.py` (independent re-computation from the raw files).
All classification figures are **development** figures unless stated: the design was selected using A and B, so they are not independent estimates for Company C.
"Zero-label" = labels never loaded for prediction (`--zero-label`); "supervised (OOF)" = every labelled line predicted by a model that did not see it (GroupKFold by item+text).

## Baseline → final (leaf accuracy | level-1 | leaf spend-weighted | macro-F1 | runtime of the final run)
| Run | Leaf | L1 | Leaf spend-w | Macro-F1 | Runtime |
|---|---|---|---|---|---|
| A zero-label | 17.8 % → **24.0 %** | 43.4 % → **65.4 %** | 28.1 % → **30.6 %** | 0.146 → **0.198** | 1.6 min |
| B zero-label | 62.2 % → **65.8 %** | 81.1 % → **86.3 %** | 86.5 % → **87.8 %** | 0.503 → **0.525** | 1.2 min |
| A supervised (OOF) | 93.0 % → **92.5 %** | 98.7 % → **98.4 %** | 89.2 % → **89.8 %** | 0.763 → **0.770** | 10.1 min |
| B supervised (OOF) | 88.7 % → **88.7 %** | 93.1 % → **93.1 %** | 96.6 % → **96.6 %** | 0.599 → **0.599** | 1.2 min |

Runtimes include only cached embeddings for the zero-label runs; the first uncached run of A embeds ~11k unique texts with four models (+≈ 10–15 min on the 4-core dev laptop). Both companies stay far below the 60-minute limit for 25,000 lines
(A has 65,549 records, B 41,028).
**Honest reading.** Supervised A leaf accuracy moved 93.0 → 92.5 % (spend-weighted 89.2 → 89.8 %, L1 98.7 → 98.4 %): a small regression/neutral change caused by the new metadata-prefixed embeddings; kept because the zero-label gain is large and supervised accuracy is flat within noise.
Zero-label A is still weak at leaf level (24 %): the text of its largest categories carries no information about the equipment, which is only partly recoverable from the supplier.

## Matching (A, unchanged logic, re-verified)
Tool lines → one-to-one ERP matches 20,476; aggregates 4; copies of an already-consumed ERP line 571; tool re-exports 713; ambiguous 0; tool-only 6,756 (incl. cancelled / not approved).
Total included spend ZAR 2,997,176,158.14; B USD 327,728,259.50.
Supplier grouping vs golden keys (A, all ERP lines incl. the catch-all bucket): {'rows': 37029, 'pairwise_precision': 1.0, 'pairwise_recall': 0.1, 'pairwise_f1': 0.1818, 'cluster_purity': 1.0, 'cluster_completeness': 0.5448, 'predicted_groups': 191, 'true_groups': 70}.
Item grouping vs golden keys (A, all lines incl. the catch-all bucket): {'rows': 21213, 'pairwise_precision': 0.9997, 'pairwise_recall': 0.3663, 'pairwise_f1': 0.5362, 'cluster_purity': 0.9997, 'cluster_completeness': 0.8668, 'predicted_groups': 3703, 'true_groups': 2426}. Excluding the buckets (see DECISIONS.md): supplier purity/completeness 1.00/1.00, item pairwise precision/recall 1.00/0.987.
The labels' catch-all supplier bucket (122 names, 56 % of ERP lines) is a deliberate merge in the ground truth and cannot be reconstructed; the figures that include it therefore understate recall.

## Independent financial re-computation (from raw files)

### Company A (ZAR)
- records: ERP 37,029 + tool 28,520 = 65,549; output rows 65,549
- PASS: every input record appears exactly once
- PASS: excluded rows have spend == 0
- PASS: included rows have blank exclusion_reason
- PASS: excluded rows carry a valid code
- PASS: confidence within [0,1]
- PASS: included ERP spend equals the supplied net value (no VAT removed again) on 37,029 lines; 0 ERP lines excluded
- PASS: included tool spend equals gross/(1+VAT) (max abs difference 0.0043 on 6,253 lines; blank VAT -> 15 %, explicit 0 kept)
- tool lines by VAT field: 15=6920, 0.15=6826, 15%=6652, blank=6542, 0=1580
- PASS: every Cancelled / On Hold / Rejected tool line carries CANCELLED / NOT_APPROVED (0 violations)
- total included spend ZAR 2,997,176,158.14; excluded by reason: {'DUPLICATE': 21764, 'NOT_APPROVED': 315, 'CANCELLED': 188}
- ERP lines also present in the tool (duplicates removed): spend removed from the tool side = ZAR 1,506,756,106.48

### Company B (USD)
- records 41,028; output rows 41,028
- PASS: every input record appears exactly once
- PASS: excluded rows have spend == 0
- PASS: included rows have blank exclusion_reason
- PASS: excluded rows carry a valid code
- PASS: confidence within [0,1]
- PASS: lines with USD or a line rate: USD = local / rate (max abs difference 0.0050 on 35,095 lines); USD lines unchanged: PASS
- lines without a line rate: 2,760; of these 2,755 have their invoice month in the FX table and 2,755 agree with local / monthly rate; the rest use the documented fallbacks (posting month, nearest month)
- included lines whose amount converts to 0.00 USD: 753, of which 753 are legitimate sub-cent amounts (PYG/CLP/MMK); unconverted for lack of a rate: 0
- total included spend USD 327,728,259.50; excluded by reason: {'NON_PROCUREMENT': 2493, 'DUPLICATE': 677}
- PASS: every intercompany-account line is NON_PROCUREMENT (or a re-export of one)


## Tests
81 passed, 3 warnings in 5.02s (78 original tests unchanged + 3 new: parent-evidence sharing, metadata prefix, torch-before-numpy / models-used report).

## Compliance checklist against the PDF
| Requirement | Status |
|---|---|
| Config-driven ingestion (YAML/JSON, new company = new config) | done, tested with two synthetic companies |
| Combine systems, count each purchase once, ERP precedence | done (A) |
| VAT removed from gross, blank → 15 %, ERP never re-taxed | verified independently |
| FX: local / rate, line rate → monthly table (invoice month) | verified independently |
| Every date format parsed or flagged | done; no row dropped |
| Four exclusion codes with precedence | done |
| Supplier / item de-duplication | done, measured |
| Classification to the lowest level, tree read at run time, confidence 0–1 | done |
| Works with zero labels (A hidden) | done, 24.0 % leaf / 65.4 % L1 (development) |
| One command; ≤ 60 min per 25,000 lines | done (see runtimes) |
| Output schema and full record coverage | validated automatically before writing |
| No data files in the repo | `.gitignore` blocks every CSV except the two output files; verified with `git add -A --dry-run` |

## Limitations (not validated or known weak)
* Company C behaviour is untested by definition; the transferable components are the generic exclusion rules, the dual-prior self-training, the parent-only supplier evidence sharing and the ranking confidence.
* Zero-label confidence does not transfer in absolute calibration (leave-one-company-out ECE 0.37); it ranks lines (AUC 0.65 / 0.78).
* Non-procurement rules for an unseen company depend on mapped account/description fields; supplier-evidence sharing assumes supplier keys exist (rows without supplier evidence are excluded from sharing).

## Final pre-publication audit (re-executed)
* Both pipelines were re-run from scratch to temporary paths: the CSVs are **byte-identical** to `outputs/aryan_sharma__A.csv` / `__B.csv` (deterministic). Runtimes: A 65,549 records in 10.0 min, B 41,028 records in 1.3 min (CPU, cached models).
* `scripts/final_validation.py` re-run: all PASS (max VAT/FX gap 0.0043 A, 0.0050 B; 0 unconverted lines; every record exactly once).
* Contract check on the two output files: exact 11 columns, unique ids, ERP:/TOOL: prefixes on A, exclusion codes only from the allowed four, valid taxonomy paths, blank L4, no blank supplier/item key on included rows, confidence in [0,1], excluded rows have spend 0 and blank path.
* `python -m pytest tests -q`: 81 passed.
* `scripts/synthetic_company_c_check.py` (invented company, 4-level taxonomy, EUR/ZAR with monthly FX table, status column, intercompany GL line, re-export): 8/8 contract checks pass through the unchanged CLI with the template filled by text substitution only. This is a configuration test; its 18 invented lines say nothing about accuracy on the real Company C.
* Confidentiality audit: no CSV is tracked; git has no commits and no remote; the only CSVs a staging dry run lists are the two output files; no credentials found; real supplier names that had appeared in unit tests were replaced by invented names (tests still pass).
