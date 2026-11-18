# DECISIONS.md — what the data showed and what was decided

All statements below come from inspecting the two kit companies. Numbers are counts/shares only; no raw records are
reproduced. "Measured" figures come from the runs listed in `reports/` (regenerate with the commands in README.md).

## 1. Company A (two systems, ZAR)

### What the data showed
* ERP goods receipts: 37,029 lines, already **net** of tax, all in ZAR (the `document_currency` column is the PO currency;
  `value_zar` is already converted). 66 % of lines have no posting date, no vendor id and no UoM (a second feed layout) but
  every line has a supplier *name*.
* Invoice tool: 28,520 lines, **gross**, ZAR; VAT rate is typed as `15`, `0.15`, `15%`, blank or `0`. A rate of `0`
  occurs on 1,580 lines and those lines match their ERP line at gross = net, so **0 is a real zero-rated line, not a blank**.
  Blank → 15 % (brief). Dates come in three formats (`dd/mm/yyyy`, `yyyy-mm-dd`, `dd-Mon-yy`), 65 % blank. No impossible dates.
* PO references are typed `4510…`, `PO#4510…`, `PO 4510…`, `4510…/10.0`, or are missing (18 %).
* Statuses: Paid 19,837 · Approved 8,180 · Cancelled 188 · On Hold 172 · Rejected 143.
* The same purchase appears several times in the tool: 55 % of tool lines match an ERP line on PO + net amount + quantity +
  supplier; re-exports carry a **different invoice number and date format** for the same purchase (so invoice number
  equality cannot be required); some tool lines settle several ERP lines of one PO (net of the sum), some are
  partial invoices of an ERP receipt at the same unit price (quantity differs) — those are *not* duplicates.
* 434 ERP lines are negative (reversals); they are kept as negative spend so that reversal pairs net to zero.
* No tax / intercompany / revenue text exists in A (keyword scan found only false hits such as "heavy duty"), so no
  `NON_PROCUREMENT` is raised for A. Cancelled / on-hold / rejected lines are almost never ERP duplicates (5 of 188),
  i.e. they are tool-only records.

### Decisions
| Topic | Decision | Why / alternative rejected |
|---|---|---|
| VAT | `net = gross / (1+rate)`, rounded half-up to cents once; ERP values never touched | avoids double tax removal; rejected "treat 0 as blank" (it matches the ERP at gross = net) |
| Matching evidence | amount **must** agree (±0.05 abs / 0.02 %) and (same PO **or** supplier + (quantity or text)) | text-only or amount-only matches rejected: coincidence rate of two-decimal amounts is not negligible |
| One-to-one | greedy assignment by descending score; each ERP line consumed once | prevents two tool lines claiming one receipt; repeated identical receipts each keep their own invoice |
| Left-over copies | a further tool line with the same PO/amount/qty/supplier as an already-consumed ERP line → `DUPLICATE` | the purchase is already counted; spend would double otherwise |
| Re-exports | tool lines equal on amount, quantity, supplier, text and PO (or, without PO, invoice number or date) → one kept | invoice number alone is unreliable |
| Aggregates | subset-sum (≤14 lines) of free ERP lines of the same PO within 2× tolerance, quantity must add up | 4 cases found; kept conservative |
| Precedence | `CANCELLED` > `NOT_APPROVED` > `NON_PROCUREMENT` > `DUPLICATE` | a record's own state beats "already counted elsewhere" |
| Ambiguous pairs | scores between 0.50 and 0.70 are *kept and reported* (`AMBIGUOUS`), never forced | none occurred on A |
| Source precedence | ERP kept, tool excluded as `DUPLICATE` | brief |

### Supplier identity (A)
Golden `supplier_key` in `labels_A.csv` puts 122 small / unlisted names (56 % of ERP lines) into one bucket
(`vendor_id` is blank for 99 % of them), so a "one key per real supplier" grouping cannot reproduce it. We resolve each distinct
real supplier separately. Against the 69 non-bucket golden suppliers: cluster purity 1.00, completeness 1.00 (see APPROACH.md).
The nearest unmerged name pairs (edit similarity 0.85–0.90, e.g. two anonymised vendors differing in one syllable) are
genuinely different suppliers in the labels, which is why the fuzzy threshold is 0.92 and names shorter than 7
characters need an exact (compacted) match.

### Item identity (A)
Material codes are authoritative: identical wording under different codes is split in the golden keys 83 % of the time
(rows), so codes are never merged on text. Code-less lines (services, free text, OCR'd tool lines) are grouped by near-identical
text within a supplier **and identical numeric specification** (`37,5mm` ≠ `53mm`; `3,5M,8` ≠ `3,5M,6`) and join a coded
item when the same supplier has a matching description. OCR confusions in letter-rich tokens (`RE5IN`, `B0OM`) are repaired first.
A catch-all golden item bucket (2,661 lines, 192 codes) cannot be reproduced and is excluded from the quality figure.

## 2. Company B (eight ERPs, 25 currencies)

### What the data showed
* 41,028 AP lines, 8 source systems (JDE ×5, SAP ×3), **no supplier names** (3,995 vendor ids), no status column.
* Dates: all `yyyy-mm-dd`; none unparseable, but 42 invoice dates are > 2 years from the posting date (2018, 2020–2022, 2027) —
  "broken dates". They are flagged `OUT_OF_RANGE` and the FX month falls back to the posting month.
* Line `currency_rate` present on 49 % of lines: all non-USD lines of 15 currencies; CAD, HKD, MMK, THB have **no** line
  rate; a few EUR/GBP/BRL lines also miss it. These use `fx_rates_usd.csv` for the invoice month. Table months are
  not continuous (e.g. 14–20 months for some currencies); an absent month uses the **nearest** available month of the same currency and is flagged
  (`FX_NEAREST_MONTH`). No invented rates: a currency absent from the table would stay unconverted and be reported.
* 613 lines are exact re-exports of another line (identical on system, invoice, line, vendor, PO, material, quantity, currency,
  amount, invoice date; only due date / account text differ) → `DUPLICATE`, first kept.
* 833 negative and 918 zero-amount lines: kept (credit memos reduce spend; zeros add nothing).
* 25 % of lines carry the placeholder material number `MAT-1000007` for thousands of unrelated descriptions; 17 % of
  descriptions are blank or whitespace.
* The 3,000 seed labels contain `include` / `exclusion_reason` too: all 450 exclusions are `NON_PROCUREMENT`; there are no
  cancelled/duplicate seeds.

### Non-procurement (B)
Account description matching `IC…` / `I/C Payable` / `Intercompany…` and supplier types 2742 / 5406 are intercompany; description lines such as
"State & Local Taxes", "Sales Tax Expense", "GST Input Tax", "Accrued Sales & Use Tax" are tax postings. The rules are generic
regular expressions (in `configs/common.yaml`) plus one company-specific supplier-type list (in `company_b.yaml`). They reproduce the
analyst's 450 exclusions with 0 false positives on the other 2,550 seeds (450/450 recall, 2,550/2,550 specificity). Consulting
lines *about* tax ("Tax Services", "Blanket PO for Tax Work", "Research and Development Tax Credit consulting") are real supplier spend and are not excluded.

### Currency / tax (B)
`USD = local / currency_rate` (rate = units of local per 1 USD); USD lines unchanged even where a rate is present. AP line
amounts carry no VAT column, so they are used as supplied (documented assumption).

### Supplier identity (B)
Vendor ids are numbered consecutively across the whole company (V500009 … V511985) and 74 of 3,995 ids appear in
several ERPs; they are treated as **globally unique** (`vendor_id_scope: global`). This is a configuration choice; switch to
`per_system` when ids restart per ERP. No two different ids are ever merged.

## 3. Cross-cutting
* **Blank / junk descriptions** are never dropped: they take the configured fallback text (account description) and a low confidence;
  if there is nothing at all the most common predicted category is used with confidence ≤ 0.05 so that every included
  line still carries a valid category.
* **Dates** never cause records to be dropped; flags are reported in the run JSON. Day-first vs month-first is never guessed: A's three
  formats are listed explicitly, and `auto` inference only accepts formats that actually parse the column.
* **Included spend can be negative** (reversals, credit memos).
* **Excluded rows' `confidence`** is the confidence in the exclusion decision (the output template shows a non-zero value on a `DUPLICATE`).

## 4. Classification decisions
See APPROACH.md for the method and measured accuracies. Key choices and rejected alternatives are listed there and in
`PERSONAL_LEARNING_GUIDE.md`; the headline decisions are: local-first (no paid API), hierarchical decoding over leaf paths
only (impossible parent/child combinations cannot be produced), group-wise cross-fitting for every supervised prediction
(no line is predicted by a model that saw it or an identical text), and a confidence that is calibrated on out-of-fold data when labels exist and
transferred from the development companies otherwise.

## 4b. Classification experiments (development data; numbers are leaf accuracy on hidden labels)
* Embedding models alone, zero-label, B seeds: multilingual-e5-base 43 %, e5-small 37 %, MiniLM-L6 34 %, bge-small 31 %, paraphrase-multilingual-mpnet 30 %; A: 6-8 % for every model (SAP short texts).
* Fine-tuning an encoder on one company and testing on the other: no transfer (A 6.3 % -> 6.2 %); dropped.
* Self-training prior comparison on the full B file (spend-weighted L1): embeddings 46 % -> 45 %, lexical 53 % -> 92 %, average of both self-trained 95 %; on A all variants stay at 14-18 % leaf.
  Chosen: average of lexical-prior and embedding-prior self-training. Embedding prior alone was *worse* than lexical on spend-weighted metrics.
* Logistic regression inside self-training stalled for over an hour on 15k rows; replaced by SGD log-loss (same accuracy within 0.3 pt).
* Bug found and fixed during development: on Windows, importing numpy/pandas before torch silently disabled embeddings (lexical fallback); `src/__init__.py` now imports torch first and reports the models used in the run JSON.
* A local LLM was not feasible (7 GB RAM CPU machine) and sending company text to an LLM API would upload confidential data, so none is used.
* Output `supplier_key` / `item_key` are hashed so no company text is published. Output files are named `aryan_sharma__A.csv` / `aryan_sharma__B.csv`; rename to your name before submission.
* Excluded rows' confidence expresses the exclusion decision. Confidence in zero-label mode is a transferred model fitted on only two companies (ECE about 0.22 there): treat it as a ranking signal, not a probability.

## 4c. Second optimisation pass (evidence: `reports/optimization_experiments.md`, `reports/final_validation_report.md`)
| Change | Why / evidence | Alternative rejected | Trade-off / limitation |
|---|---|---|---|
| Parent-only (level-1) evidence sharing by supplier, label-free | A's supplier has 98 % level-1 purity; sharing the whole leaf distribution cost B 16 pt, parent-only gave A L1 +8-11 pt and B leaf +11 pt (dev) | full-leaf sharing; sharing at L1+L2 (no further gain) | needs supplier keys; rows without supplier evidence stay singletons |
| Metadata prefix on the embedded text (config `embed_prefix_fields`; A: spend type + material type) | A L1 45 -> 53 % (proxy ensemble) | stripping codes (-0.8 pt A, -13 pt B leaf), repeating the head phrase (worse on both) | company-specific config, no effect on B |
| Dual-prior self-training (lexical + embedding), SGD | lexical-only 52.8 %, embedding-only 41.8 %, average 62 % leaf on B | embedding prior alone; logistic regression inside self-training (>1 h) | two self-trainings; still cannot repair a wrong prior |
| Zero-label confidence refit | in-sample ECE 0.19; leave-one-company-out ECE 0.37; AUC 0.65 (A) / 0.78 (B) | per-company calibration (needs labels) | absolute values are not probabilities; use for ranking |
| Entity-resolution scores reported with and without the golden catch-all bucket | supplier: 1.00/1.00 purity/completeness without it, pairwise recall 0.10 with it (122 unrelated names in one golden key); items: P 1.00 / R 0.987 without, R 0.37 with | claiming the bucket can be reconstructed | the figures with the bucket understate recall by construction |
| Independent financial validation (`scripts/final_validation.py`) | separate code recomputes VAT, FX, exclusions from raw files: all checks PASS; 753 included B lines convert to 0.00 USD because they are sub-cent amounts (PYG/CLP/MMK) | trusting the pipeline | none |
| Supervised A leaf 93.0 -> 92.5 % (spend-weighted 89.2 -> 89.8 %) | side effect of the prefixed embeddings in the blend | reverting the prefix (loses A zero-label gain) | reported as neutral within noise |
| Not done | LLM or larger encoders (7 GB RAM, confidential data), abbreviation dictionaries (ambiguous without a corpus) | - | zero-label A leaf accuracy stays low (24 %) |

**Evaluation integrity.** Zero-label inference never loads labels (`labels_used_for_prediction: false` in every report); labels are read after prediction. Because the design was tuned on A and B, all zero-label numbers are *development* numbers; they are not independent validation and not blind-test performance. Cross-fitted supervised numbers are a separate category.

## 5. Known limitations
* The two kit companies' golden supplier / item keys contain large catch-all buckets; our quality figures exclude them and say so.
* Zero-label accuracy is limited by a small local embedding model (no LLM). See APPROACH.md for measured values.
* Non-procurement for an unseen company depends on the fields the config maps (account text, document text); a company whose
  intercompany postings carry no tell-tale text would need one extra rule in its YAML.
* Aggregate (one-to-many) invoice matching is subset-sum within one PO and capped at 14 free lines.

## 7. Publication and confidentiality decisions
* **Tests and docs use invented names only.** Supplier names that had been copied from the confidential data into unit tests were replaced by invented names with the same typing patterns (spacing, legal suffix, OCR digit, near-miss pair); category names in the error analysis were generalised. Supplier/item keys in the outputs are hashes.
* **Raw data stays outside the repository** (`../company_A`, `../company_B`); `.gitignore` is a second line of defence, ignoring every CSV except the two `__A`/`__B` output files.
* **Confidence is a ranking score, not a probability** for zero-label runs (leave-one-company-out ECE 0.37); documented in APPROACH.md and the guide. No attempt was made to tune it to improve a weighted metric.
* **Synthetic Company C test** (`scripts/synthetic_company_c_check.py`) only proves the YAML-only onboarding path on invented data.
* **No further model changes in the final pass.** A leaf accuracy (24.0 % zero-label) is a documented limitation: the reason is information missing from the text, and further variants were rejected after controlled experiments (see reports/optimization_experiments.md).
