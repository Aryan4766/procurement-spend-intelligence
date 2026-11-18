#!/usr/bin/env python
"""One command:  python main.py --config configs/company_a.yaml [--data-dir DIR] [--out FILE]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.config_loader import ConfigError  # noqa: E402
from src.pipeline import run_company  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cantill.ai procurement spend intelligence engine")
    ap.add_argument("--config", required=True, help="company YAML/JSON config")
    ap.add_argument("--data-dir", help="folder with the company's input files (overrides config data_dir)")
    ap.add_argument("--labels-dir", help="folder with the labels file used for supervised training (default: data dir)")
    ap.add_argument("--out", help="output CSV path (default: outputs/aryan_sharma__<company>.csv)")
    ap.add_argument("--zero-label", action="store_true", help="never use labels for prediction (labels may still be used to evaluate)")
    ap.add_argument("--no-evaluate", action="store_true", help="do not read label files for evaluation")
    ap.add_argument("--work-dir", help="folder for the embedding cache (default: models/cache)")
    ap.add_argument("--report-dir", default="reports", help="folder for diagnostics JSON")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        d = run_company(a.config, a.data_dir, a.labels_dir, a.out, a.zero_label, not a.no_evaluate, a.work_dir, a.report_dir, a.seed)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    d.pop("_txn", None)
    d.pop("_out", None)
    keys = ("company", "records", "mode", "reconciliation", "total_seconds", "stage_seconds", "output_file")
    print(json.dumps({k: d[k] for k in keys}, indent=2, default=str))
    if "classification_vs_labels" in d:
        print("classification vs labels:", json.dumps(d["classification_vs_labels"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
