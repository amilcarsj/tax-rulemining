"""Baseline experiment orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baselines.common import candidate_numeric_features
from baselines.flat_lof import run_flat_lof_baseline
from baselines.pseudo_label_rules import (
    PSEUDO_LABEL_RULE_FILES,
    run_pseudo_label_rule_baseline,
)
from baselines.raw_association import RawAssociationConfig, run_raw_association_baseline
from core.progress import ProgressReporter
from data_io.dataset import load_dataset_schema, load_dataset_table
from data_io.writers import write_json


BASELINE_OUTPUT_FILES = [
    "flat_lof_scores.csv",
    "flat_lof_feature_metadata.csv",
    "flat_lof_summary.json",
    "raw_feature_discretization_ranges.csv",
    "raw_feature_bins.csv",
    "raw_feature_transactions.csv",
    "raw_frequent_itemsets.csv",
    "raw_association_rules.csv",
    "raw_association_summary.json",
    *PSEUDO_LABEL_RULE_FILES,
    "baseline_summary.json",
]


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for all comparison baselines."""

    exclude_columns: list[str] = field(default_factory=lambda: ["object_id"])
    raw_association: RawAssociationConfig = field(default_factory=RawAssociationConfig)


def run_baselines(
    data_path: Path,
    output_dir: Path,
    trajectory_id: str,
    config: BaselineConfig,
    step2_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all baseline experiments for one vectorized trajectory dataset."""
    progress = ProgressReporter()
    progress.stage("Baselines: loading dataset schema")
    schema = load_dataset_schema(data_path, trajectory_id)
    progress.stage("Baselines: loading dataset table")
    table = load_dataset_table(data_path, trajectory_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = candidate_numeric_features(schema, trajectory_id, config.exclude_columns)
    progress.stage(f"Baselines: selected {len(features)} numeric raw features")

    progress.stage("Baseline 1: flat LOF over all selected numeric features")
    flat_summary = run_flat_lof_baseline(
        table=table,
        schema=schema,
        data_path=data_path,
        output_dir=output_dir,
        trajectory_id_column=trajectory_id,
        features=features,
        exclude_columns=config.exclude_columns,
    )

    progress.stage("Baseline 2: raw-feature association rules")
    raw_summary = run_raw_association_baseline(
        table=table,
        schema=schema,
        data_path=data_path,
        output_dir=output_dir,
        trajectory_id_column=trajectory_id,
        features=features,
        exclude_columns=config.exclude_columns,
        config=config.raw_association,
    )

    pseudo_summary: dict[str, Any] | None = None
    if step2_dir is not None:
        progress.stage("Baseline 3: pseudo-label rules without semantic compression")
        pseudo_summary = run_pseudo_label_rule_baseline(
            step2_dir=step2_dir,
            output_dir=output_dir,
        )

    payload = {
        "baseline_suite": (
            "flat_lof_raw_association_and_uncompressed_pseudo_label_rules"
            if pseudo_summary is not None
            else "flat_lof_and_raw_association_rules"
        ),
        "status": "completed",
        "inputs": {
            "data": str(data_path),
            "trajectory_id": trajectory_id,
            "excluded_columns": config.exclude_columns,
            "step2_dir": str(step2_dir) if step2_dir is not None else "",
        },
        "dataset": {
            "row_count": schema.row_count,
            "column_count": len(schema.headers),
            "selected_numeric_feature_count": len(features),
        },
        "results": {
            "flat_lof": flat_summary["results"],
            "raw_association_rules": raw_summary["results"],
            "pseudo_label_rules_without_compression": (
                pseudo_summary["results"] if pseudo_summary is not None else {}
            ),
        },
        "generated_files": {
            filename: str(output_dir / filename)
            for filename in BASELINE_OUTPUT_FILES
            if (output_dir / filename).exists() or filename == "baseline_summary.json"
        },
    }
    write_json(output_dir / "baseline_summary.json", payload)
    progress.stage("Baselines: completed")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated comparison baselines for the taxonomy-guided pipeline."
    )
    parser.add_argument("--data", required=True, help="Path to the input feature CSV.")
    parser.add_argument(
        "--trajectory-id",
        default="trajectory_id",
        help="Identifier column in the feature CSV. Default: trajectory_id",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where baseline outputs will be written.",
    )
    parser.add_argument(
        "--step2-dir",
        default=None,
        help=(
            "Optional Step 2 output directory. When provided, the baseline suite "
            "also exports uncompressed pseudo-label rules."
        ),
    )
    parser.add_argument(
        "--exclude-column",
        action="append",
        default=["object_id"],
        help=(
            "Numeric column to exclude from baseline feature sets. Can be repeated. "
            "Default excludes object_id."
        ),
    )
    parser.add_argument(
        "--raw-min-support-count",
        type=int,
        default=20,
        help="Minimum raw association itemset support count. Default: 20",
    )
    parser.add_argument(
        "--raw-min-support-ratio",
        type=float,
        default=0.10,
        help="Minimum raw association itemset support ratio. Default: 0.10",
    )
    parser.add_argument(
        "--raw-min-confidence",
        type=float,
        default=0.60,
        help="Minimum raw association rule confidence. Default: 0.60",
    )
    parser.add_argument(
        "--raw-min-lift",
        type=float,
        default=1.20,
        help="Minimum raw association rule lift. Default: 1.20",
    )
    parser.add_argument(
        "--max-raw-itemset-length",
        type=int,
        default=2,
        help="Maximum raw frequent itemset length for FP-growth. Default: 2",
    )
    parser.add_argument(
        "--raw-max-rules",
        type=int,
        default=50000,
        help="Maximum number of raw association rules to write after sorting. Default: 50000",
    )
    parser.add_argument(
        "--raw-min-antecedent-length",
        type=int,
        default=1,
        help="Minimum antecedent length for raw association rules. Default: 1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_baselines(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        trajectory_id=args.trajectory_id,
        step2_dir=Path(args.step2_dir) if args.step2_dir else None,
        config=BaselineConfig(
            exclude_columns=args.exclude_column,
            raw_association=RawAssociationConfig(
                min_support_count=args.raw_min_support_count,
                min_support_ratio=args.raw_min_support_ratio,
                min_confidence=args.raw_min_confidence,
                min_lift=args.raw_min_lift,
                max_itemset_length=args.max_raw_itemset_length,
                max_rules=args.raw_max_rules,
                min_antecedent_length=args.raw_min_antecedent_length,
            ),
        ),
    )
    return 0
