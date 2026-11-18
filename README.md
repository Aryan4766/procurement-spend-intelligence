# Cantill.ai — Procurement Spend Intelligence Engine

A configuration-driven pipeline that turns a company's raw purchase data and its category tree into a **spend cube**:
every purchase line counted once, net of tax, in the reporting currency, with a clean supplier, a clean item and a
category at the lowest level of the company's own taxonomy.

> Status: build for the Cantill.ai Data & ML Intern Task. Companies A and B are configured; Company C (blind
> test) is supported by writing one new YAML file — no code changes.

## What it does

| Step | Module | Notes |
|---|---|---|
| Ingest by config | `src/config`, `src/ingestion` | YAML/JSON per company; column mapping, formats, tax rule, FX source; readable errors for missing columns |
| Clean amounts & dates | `src/preprocessing`, `src/spend/financials.py` | exact `Decimal` arithmetic, VAT removal for gross sources, FX to reporting currency, every date format parsed or flagged |
| Decide what is spend | `src/spend/exclusion_engine.py` | `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE` with a fixed precedence |
| Count each purchase once | `src/matching/transaction_matcher.py` | evidence-scored, one-to-one ERP↔invoice matching + subset-sum + re-export detection |
| De-duplicate suppliers / items | `src/matching/supplier_resolver.py`, `item_resolver.py` | blocked fuzzy matching, conservative merges, diagnostics |
| Classify into any tree | `src/taxonomy`, `src/classification` | tree read at run time; semantic zero-label prior + optional supervised models, hierarchical decoding, confidence score in [0, 1] (calibrated against out-of-fold correctness when labels exist; a ranking signal only in zero-label mode) |
| Output & validate | `src/output`, `src/evaluation/pipeline_validation.py` | exact 11-column contract, validated automatically before the file is written |

Design documents: [APPROACH.md](APPROACH.md) (one page), [DECISIONS.md](DECISIONS.md) (what the data showed and what was
decided). `PERSONAL_LEARNING_GUIDE.md` is private study material for the author, **not** an assessment deliverable.

## Install

Python 3.10+ (developed on 3.12). CPU only; no GPU and **no paid API** is needed.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads four small sentence-embedding models from Hugging Face (about 1.1 GB in total, public models only;
no company data is ever sent anywhere) in the standard Hugging Face cache of the user account (outside this repository); computed embeddings are cached in
`models/cache` (git-ignored). Run once with network access before any offline use. Without the models the pipeline still runs: it falls back to a lexical
(TF-IDF) similarity prior and says so (`embedding_available: false`) in its report.

## Results at a glance (development data)

| | A | B |
|---|---|---|
| Runtime (CPU, cached models) | 10.0 min / 65,549 records | 1.3 min / 41,028 records |
| Leaf accuracy, supervised (cross-fitted) | 92.5 % | 88.7 % |
| Leaf / level-1 accuracy, zero-label (labels hidden) | 24.0 % / 65.4 % | 65.8 % / 86.3 % |

Financial checks (VAT, FX, exclusions, one row per record) pass in the independent validation. Zero-label figures are
development numbers (design choices were made while looking at A and B), not blind-test results; Company C is untested.
Details: [APPROACH.md](APPROACH.md), `reports/final_validation_report.md`, `reports/optimization_experiments.md`.

## Run (one command per company)

```bash
python main.py --config configs/company_a.yaml --data-dir /path/to/company_A --out outputs/aryan_sharma__A.csv
python main.py --config configs/company_b.yaml --data-dir /path/to/company_B --out outputs/aryan_sharma__B.csv
```

* `--data-dir` – folder holding the company's input files (the data is confidential and never committed; `data_dir`
  in the config is only a default).
* `--zero-label` – never use any label for prediction (labels may still be used afterwards to *evaluate*). This is how the
  "A with labels hidden" figure in APPROACH.md is produced.
* `--labels-dir` – where the labels file lives when it is not in the data folder (e.g. blind-test data folder without
  labels, original kit with labels: the tool then trains on the kit's labels).
* `--no-evaluate` – do not even read label files for evaluation.
* Diagnostics (aggregate numbers only, never raw records) are written to `reports/run_<company>_<mode>.json`.

### New company (e.g. the blind-test Company C)

1. Copy `configs/company_c_template.yaml` to `configs/company_c.yaml`.
2. Fill in the file name, the id column, and map the columns (`description`, `amount`, `date`, `supplier_*`, `status`…).
   Pick `basis: net|gross`, the currency column or fixed currency, the FX file if any, the date formats (or `auto`).
3. Point `taxonomy.file` and `taxonomy.level_columns` at the tree (1–4 levels).
4. Run `python main.py --config configs/company_c.yaml --data-dir <folder> --out submission/aryan_sharma__C.csv`.

See the comments in the two real configs for every option.

## Output

`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence` — one row per input
record, excluded ones included (spend 0, blank categories, `confidence` then expresses confidence in the *exclusion*
decision). Included `spend` is net of tax, in the reporting currency. The file is validated before it is written
(unique ids, all input ids present, valid codes, categories exist in the tree, spend/confidence ranges).

## Blind-test usage

```bash
# A2 / B2: frozen configs, new data folder; labels (for the supervised models) still come from the kit folder
python main.py --config configs/company_a.yaml --data-dir <A2 folder> --labels-dir ../company_A --out submission/aryan_sharma__A2.csv
python main.py --config configs/company_b.yaml --data-dir <B2 folder> --labels-dir ../company_B --out submission/aryan_sharma__B2.csv
# C: new config only (copy configs/company_c_template.yaml), no labels -> zero-label mode automatically
python main.py --config configs/company_c.yaml --data-dir <C folder> --out submission/aryan_sharma__C.csv
```
The run prints and stores (`reports/run_<company>_<mode>.json`) which embedding models were really used; if it says
`embedding_available: false` the semantic models failed to load (e.g. torch import order on Windows) and only the lexical prior ran.

## Reproducing the reported numbers

```bash
python main.py --config configs/company_a.yaml --data-dir ../company_A --zero-label --out a_zl.csv   # A with labels hidden
python scripts/zero_label_experiments.py A            # experiment table (see reports/optimization_experiments.md)
python scripts/final_validation.py --a-data ../company_A --b-data ../company_B --a-out outputs/aryan_sharma__A.csv --b-out outputs/aryan_sharma__B.csv
python -m src.evaluation.fit_confidence reports/private/conf_features_A.npz reports/private/conf_features_B.npz
python scripts/synthetic_company_c_check.py           # config-only onboarding test on an invented 4-level company (not Company C validation)
```

## Tests

```bash
python -m pytest tests -q                               # 81 tests, about 6 s
python scripts/final_validation.py --a-data ../company_A --b-data ../company_B        --a-out outputs/aryan_sharma__A.csv --b-out outputs/aryan_sharma__B.csv   # independent financial re-computation
```

Unit tests (amounts, VAT, FX, dates, references), matching/exclusion/entity-resolution tests on synthetic data, and
end-to-end tests that build two throw-away companies from YAML alone (81 tests). Tests never use the confidential data
(supplier names in tests are invented). `scripts/synthetic_company_c_check.py` fills `configs/company_c_template.yaml` for an
invented company with a four-level taxonomy, runs the normal CLI in zero-label mode with the real embedding models and checks the
output contract; it proves the *onboarding path* (YAML only, no code change), not accuracy on the real Company C.

## Repository hygiene

`.gitignore` excludes every CSV/Excel/archive except `outputs/*__A.csv`, `outputs/*__B.csv` (and `submission/*__A|B*.csv`), the
company data folders and raw file names, model caches (`models/`), `reports/private/`, secrets (`.env*`, keys) and local junk.
The raw data lives outside this repository (`../company_A`, `../company_B`). Never commit task data. Check before publishing:
`git add -A --dry-run` must list no CSV other than the two outputs.

## Known limitations

See "Limitations" in [APPROACH.md](APPROACH.md) and [DECISIONS.md](DECISIONS.md). In short: zero-label accuracy
without an LLM is bounded by the quality of a small local embedding model; non-procurement detection for an unseen company
relies on generic keyword rules plus whatever account/description fields the config maps. Zero-label accuracy on A is low
(24 % leaf) because its short SAP-style texts do not name the equipment; zero-label confidence ranks lines within a company but is
not a calibrated probability across companies (leave-one-company-out ECE 0.37); Company C behaviour is unvalidated.
