"""Dynamic taxonomy: read at run time from the company's tree file. No category name appears anywhere in code."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config.config_loader import ConfigError
from src.ingestion.data_loader import read_csv_text
from src.ingestion.schema_validator import validate_frame_columns


@dataclass
class Taxonomy:
    level_columns: list[str]
    paths: list[tuple[str, ...]]                    # one entry per leaf (full original names, no padding)
    leaf_ids: dict[tuple[str, ...], int] = field(default_factory=dict)
    children: dict[tuple[str, ...], list[tuple[str, ...]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.level_columns)

    @property
    def n_leaves(self) -> int:
        return len(self.paths)

    def padded(self, idx: int) -> tuple[str, ...]:
        p = self.paths[idx]
        return p + ("",) * (self.depth - len(p))

    def path_text(self, idx: int, sep: str = " > ") -> str:
        return sep.join(self.paths[idx])

    def level_names(self, level: int) -> list[str]:
        """Distinct category names at a given level (0-based), original spelling."""
        seen = dict.fromkeys(p[level] for p in self.paths if len(p) > level)
        return list(seen)

    def prefix_index(self, level: int) -> dict[tuple[str, ...], list[int]]:
        """Map a path prefix of length level+1 to the leaf indices below it."""
        out: dict[tuple[str, ...], list[int]] = {}
        for i, p in enumerate(self.paths):
            if len(p) > level:
                out.setdefault(p[: level + 1], []).append(i)
        return out

    def is_valid_path(self, levels: tuple[str, ...]) -> bool:
        trimmed = tuple(x for x in levels if x != "")
        return trimmed in self.leaf_ids or any(p[: len(trimmed)] == trimmed for p in self.paths)


def load_taxonomy(path: Path, level_columns: list[str]) -> Taxonomy:
    df = read_csv_text(path)
    validate_frame_columns(f"taxonomy file {path.name}", df, level_columns)
    df = df[level_columns].apply(lambda s: s.astype(str).str.strip())
    warnings: list[str] = []
    paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for row_no, row in enumerate(df.itertuples(index=False), start=2):
        vals = list(row)
        if vals[0] == "":
            raise ConfigError(f"taxonomy {path.name} line {row_no}: first level is empty")
        # the tree ends at the first blank; anything non-blank after a blank is a hierarchy error
        depth = next((i for i, v in enumerate(vals) if v == ""), len(vals))
        if any(v != "" for v in vals[depth:]):
            raise ConfigError(f"taxonomy {path.name} line {row_no}: a level is blank but a deeper level is filled: {vals}")
        p = tuple(vals[:depth])
        if p in seen:
            warnings.append(f"duplicate path ignored (line {row_no})")
            continue
        seen.add(p)
        paths.append(p)
    if not paths:
        raise ConfigError(f"taxonomy {path.name} is empty")
    # cycles are impossible in a flat path list; prefix/leaf consistency check
    leaf_ids = {p: i for i, p in enumerate(paths)}
    children: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for p in paths:
        for k in range(1, len(p)):
            children.setdefault(p[:k], [])
            if p[: k + 1] not in children[p[:k]]:
                children[p[:k]].append(p[: k + 1])
    return Taxonomy(level_columns=list(level_columns), paths=paths, leaf_ids=leaf_ids, children=children, warnings=warnings)
