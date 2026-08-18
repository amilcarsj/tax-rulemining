"""Pseudo-label rule baseline without semantic meta-pattern compression."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_io.writers import write_csv, write_json


PSEUDO_LABEL_RULE_FILES = [
    "pseudo_label_rules_without_compression.csv",
    "pseudo_label_rule_coverage_without_compression.csv",
    "pseudo_label_rules_without_compression_summary.json",
]


def _read_csv_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pseudo-label rule baseline requires '{path}'.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pseudo-label rule baseline requires '{path}'.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_pseudo_label_rule_baseline(
    step2_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Export Step 2 rules as the uncompressed pseudo-label-rule ablation."""
    rule_rows, rule_fieldnames = _read_csv_with_fieldnames(
        step2_dir / "high_level_association_rules.csv"
    )
    coverage_rows, coverage_fieldnames = _read_csv_with_fieldnames(
        step2_dir / "high_level_rule_coverage.csv"
    )
    step2_summary = _read_json(step2_dir / "step2_summary.json")

    write_csv(
        output_dir / "pseudo_label_rules_without_compression.csv",
        rule_fieldnames,
        rule_rows,
    )
    write_csv(
        output_dir / "pseudo_label_rule_coverage_without_compression.csv",
        coverage_fieldnames,
        coverage_rows,
    )

    antecedent_length_counts = Counter(
        _as_int(row.get("antecedent_length")) for row in rule_rows
    )
    semantic_meta_pattern_count = _as_int(
        step2_summary.get("results", {}).get("semantic_meta_pattern_count")
    )
    rule_count = len(rule_rows)
    summary = {
        "baseline": "pseudo_label_rules_without_compression",
        "status": "completed",
        "inputs": {
            "step2_dir": str(step2_dir),
        },
        "parameters": {
            "source": "Step 2 high_level_association_rules.csv",
            "compression": "none",
            "intended_comparison": (
                "Ablates semantic meta-pattern compression while keeping the "
                "same taxonomy-guided pseudo-label rule mining."
            ),
            "step2_parameters": step2_summary.get("parameters", {}),
        },
        "dataset": step2_summary.get("dataset", {}),
        "results": {
            "rule_count": rule_count,
            "coverage_row_count": len(coverage_rows),
            "antecedent_length_counts": dict(sorted(antecedent_length_counts.items())),
            "mean_confidence": _mean([_as_float(row.get("confidence")) for row in rule_rows]),
            "mean_lift": _mean([_as_float(row.get("lift")) for row in rule_rows]),
            "mean_rule_support_count": _mean(
                [_as_float(row.get("rule_support_count")) for row in rule_rows]
            ),
            "semantic_meta_pattern_count_available_in_main_method": semantic_meta_pattern_count,
            "rules_per_semantic_meta_pattern": (
                rule_count / semantic_meta_pattern_count
                if semantic_meta_pattern_count
                else 0.0
            ),
        },
        "generated_files": {
            filename: str(output_dir / filename) for filename in PSEUDO_LABEL_RULE_FILES
        },
    }
    write_json(output_dir / "pseudo_label_rules_without_compression_summary.json", summary)
    return summary
