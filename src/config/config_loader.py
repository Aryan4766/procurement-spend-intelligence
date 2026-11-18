"""YAML/JSON company configuration: loading, `extends` inheritance and validation.

A new company is supported by writing a new config file - never by touching processing code.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised with a human-readable message when a configuration file is invalid."""


KNOWN_ROLES = {"primary", "secondary", "standalone"}
EXCLUSION_CODES = ("CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE")


def _read(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json":
            return json.load(fh) or {}
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Dicts merge recursively; lists and scalars in `override` replace those in `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load a config, resolving an optional `extends: other.yaml` (relative to the config's folder)."""
    path = Path(path).resolve()
    cfg = _read(path)
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = deep_merge(base, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def validate_config(cfg: dict) -> dict:
    """Structural validation. Column-level validation happens in schema_validator once files are read."""
    problems: list[str] = []

    def need(d: dict, key: str, where: str):
        if key not in d or d[key] in (None, "", []):
            problems.append(f"missing required key '{key}' in {where}")

    for key in ("company_id", "reporting_currency", "taxonomy", "sources"):
        need(cfg, key, "config root")
    tax = cfg.get("taxonomy") or {}
    need(tax, "file", "taxonomy")
    need(tax, "level_columns", "taxonomy")
    if isinstance(tax.get("level_columns"), list) and not 1 <= len(tax["level_columns"]) <= 4:
        problems.append("taxonomy.level_columns must list between 1 and 4 columns")

    names, primaries = set(), 0
    for i, src in enumerate(cfg.get("sources") or []):
        where = f"sources[{i}]"
        for key in ("name", "file", "id_column", "record_id_template", "fields"):
            need(src, key, where)
        if src.get("name") in names:
            problems.append(f"duplicate source name '{src.get('name')}'")
        names.add(src.get("name"))
        role = src.get("role", "standalone")
        if role not in KNOWN_ROLES:
            problems.append(f"{where}.role must be one of {sorted(KNOWN_ROLES)}")
        primaries += role == "primary"
        fields = src.get("fields") or {}
        if "amount" not in fields:
            problems.append(f"{where}.fields must map 'amount'")
        if "{id}" not in str(src.get("record_id_template", "")):
            problems.append(f"{where}.record_id_template must contain '{{id}}'")
        amt = src.get("amount") or {}
        if amt.get("basis", "net") not in ("net", "gross"):
            problems.append(f"{where}.amount.basis must be 'net' or 'gross'")
        if not (amt.get("currency") or amt.get("currency_column")):
            problems.append(f"{where}.amount needs either 'currency' (fixed code) or 'currency_column'")
        if amt.get("currency_column") and amt.get("currency", cfg.get("reporting_currency")) != cfg.get("reporting_currency"):
            pass
    if (cfg.get("matching") or {}).get("enabled") and primaries != 1:
        problems.append("matching.enabled requires exactly one source with role: primary")
    prec = (cfg.get("exclusion") or {}).get("precedence", list(EXCLUSION_CODES))
    if sorted(prec) != sorted(EXCLUSION_CODES):
        problems.append(f"exclusion.precedence must be a permutation of {list(EXCLUSION_CODES)}")
    needs_fx = any(((s.get("amount") or {}).get("currency_column")) for s in cfg.get("sources") or [])
    if needs_fx and not cfg.get("fx"):
        problems.append("a source uses currency_column but no top-level 'fx' section is configured")
    if problems:
        raise ConfigError("Invalid configuration (" + str(cfg.get("_config_path", "?")) + "):\n  - " + "\n  - ".join(problems))
    return cfg


def load_and_validate(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    cfg = load_config(path)
    if overrides:
        cfg = deep_merge(cfg, {k: v for k, v in overrides.items() if v is not None})
    return validate_config(cfg)
