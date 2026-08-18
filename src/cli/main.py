"""Main CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.step1 import run_step1
from pipeline.step2 import Step2Config, run_step2
from pipeline.step3 import Step3Config, run_step3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taxonomy-guided movement pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command")

    step1_parser = subparsers.add_parser(
        "step1",
        help="Run Step 1: taxonomy parsing, LOF scoring, and pseudo-label generation.",
        description=(
            "Run Step 1 of the taxonomy-guided movement pipeline: dataset validation, "
            "taxonomy parsing, valid pair generation, LOF scoring, and pseudo-label generation."
        ),
    )
    step1_parser.add_argument("--data", required=True, help="Path to the input feature CSV.")
    step1_parser.add_argument("--taxonomy", required=True, help="Path to the taxonomy JSON definition.")
    step1_parser.add_argument(
        "--trajectory-id",
        default="trajectory_id",
        help="Identifier column in the feature CSV. Default: trajectory_id",
    )
    step1_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Step 1 outputs will be written.",
    )

    step2_parser = subparsers.add_parser(
        "step2",
        help="Run Step 2: high-level analysis of pseudo-label fingerprints.",
        description=(
            "Run Step 2 of the taxonomy-guided movement pipeline: fingerprint summaries, "
            "high-level association-rule mining, behavioural pattern cataloguing, and quantitative "
            "rule evaluation."
        ),
    )
    step2_parser.add_argument(
        "--fingerprint",
        required=True,
        help="Path to the trajectory fingerprint CSV, typically trajectory_pseudo_labels.csv.",
    )
    step2_parser.add_argument(
        "--trajectory-id",
        default="trajectory_id",
        help="Identifier column in the fingerprint CSV. Default: trajectory_id",
    )
    step2_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Step 2 outputs will be written.",
    )
    step2_parser.add_argument(
        "--include-normal-targets",
        action="store_true",
        help="Also allow Normal labels to be considered as rule targets.",
    )
    step2_parser.add_argument(
        "--include-hybrid-antecedents",
        action="store_false",
        dest="exclude_hybrid_antecedents",
        help=(
            "Retain Hybrid labels as antecedent context. By default, Hybrid labels "
            "are excluded from antecedents and never used as rule targets."
        ),
    )
    step2_parser.add_argument(
        "--include-target-label-echoes",
        action="store_false",
        dest="exclude_target_label_echoes",
        help=(
            "Allow an antecedent to repeat the target label, such as 'Uncommon "
            "Duration' implying another 'Uncommon Duration' pair. By default, "
            "these self-echo rules are excluded."
        ),
    )
    step2_parser.add_argument(
        "--max-high-level-rule-length",
        type=int,
        default=2,
        help="Maximum antecedent length for association rules. Default: 2",
    )
    step2_parser.add_argument(
        "--min-support-count",
        type=int,
        default=20,
        help="Minimum joint rule-support count. Default: 20",
    )
    step2_parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.0,
        help="Minimum dataset-level joint rule-support ratio. Default: 0.0",
    )
    step2_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum rule confidence after contrast evaluation. Default: 0.8",
    )
    step2_parser.add_argument(
        "--min-lift",
        type=float,
        default=2.0,
        help="Minimum rule lift over the target base rate. Default: 2.0",
    )
    step2_parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=0.0,
        help="Minimum fraction of target trajectories covered by a rule. Default: 0.0",
    )
    step2_parser.add_argument(
        "--min-target-prevalence-ratio",
        type=float,
        default=0.02,
        help=(
            "Minimum dataset-level prevalence ratio required for an eligible target "
            "item to be considered for a rule. Default: 0.02"
        ),
    )
    step2_parser.add_argument(
        "--min-node-depth",
        type=int,
        default=None,
        help="Minimum taxonomy node depth allowed in Step 2 pair columns. Default: no minimum",
    )
    step2_parser.add_argument(
        "--max-node-depth",
        type=int,
        default=None,
        help="Maximum taxonomy node depth allowed in Step 2 pair columns. Default: no maximum",
    )

    step3_parser = subparsers.add_parser(
        "step3",
        help="Run Step 3: semantic meta-pattern explanation.",
        description=(
            "Explain Step 2 semantic meta-patterns with comparison-level evidence, "
            "feature-level contrasts, and optional rule-level contrast outputs."
        ),
    )
    step3_parser.add_argument(
        "--data",
        required=True,
        help="Path to the original feature CSV used in Step 1.",
    )
    step3_parser.add_argument(
        "--step1-dir",
        required=True,
        help="Directory containing Step 1 outputs, including node_feature_sets.csv.",
    )
    step3_parser.add_argument(
        "--step2-dir",
        required=True,
        help="Directory containing Step 2 association-rule outputs.",
    )
    step3_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Step 3 outputs will be written.",
    )
    step3_parser.add_argument(
        "--trajectory-id",
        default="trajectory_id",
        help="Identifier column in the feature CSV. Default: trajectory_id",
    )
    step3_parser.add_argument(
        "--all-counterexamples",
        action="store_true",
        default=True,
        help=(
            "When rule-level outputs are enabled, retain all real contrast trajectories "
            "for each rule-positive trajectory. Default: enabled"
        ),
    )
    step3_parser.add_argument(
        "--include-rule-level-outputs",
        action="store_true",
        help=(
            "Also generate the older exhaustive rule-level contrast and feature files. "
            "By default, Step 3 writes only semantic meta-pattern explanations."
        ),
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] not in {"step1", "step2", "step3", "-h", "--help"}:
        raw_args = ["step1", *raw_args]
    args = parser.parse_args(raw_args)
    if not getattr(args, "command", None):
        parser.error("Please specify a pipeline step, for example: step1")
    return args


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        if args.command == "step1":
            run_step1(
                data_path=Path(args.data),
                taxonomy_path=Path(args.taxonomy),
                trajectory_id=args.trajectory_id,
                output_dir=Path(args.output_dir),
            )
        elif args.command == "step2":
            run_step2(
                fingerprint_path=Path(args.fingerprint),
                output_dir=Path(args.output_dir),
                trajectory_id=args.trajectory_id,
                config=Step2Config(
                    include_normal_targets=args.include_normal_targets,
                    exclude_hybrid_antecedents=args.exclude_hybrid_antecedents,
                    exclude_target_label_echoes=args.exclude_target_label_echoes,
                    max_high_level_rule_length=args.max_high_level_rule_length,
                    min_support_count=args.min_support_count,
                    min_support_ratio=args.min_support_ratio,
                    min_confidence=args.min_confidence,
                    min_lift=args.min_lift,
                    min_target_coverage=args.min_target_coverage,
                    min_target_prevalence_ratio=args.min_target_prevalence_ratio,
                    min_node_depth=args.min_node_depth,
                    max_node_depth=args.max_node_depth,
                ),
            )
        elif args.command == "step3":
            run_step3(
                data_path=Path(args.data),
                step1_dir=Path(args.step1_dir),
                step2_dir=Path(args.step2_dir),
                output_dir=Path(args.output_dir),
                trajectory_id=args.trajectory_id,
                config=Step3Config(
                    include_all_counterexamples=args.all_counterexamples,
                    include_rule_level_outputs=args.include_rule_level_outputs,
                ),
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    except ValueError as error:
        raise SystemExit(str(error)) from error
    except ImportError as error:
        raise SystemExit(str(error)) from error

    return 0
