"""Step 1 pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.models import DatasetSchema, NodeLofSummary, TaxonomyNode
from core.progress import ProgressReporter
from data_io.dataset import load_dataset_schema, load_dataset_table
from data_io.writers import write_csv, write_json
from scoring.lof import compute_node_lof_scores, lof_strategy_summary
from scoring.pseudo_labels import (
    build_trajectory_pseudo_label_table,
    generate_pair_pseudo_labels,
)
from taxonomy.pairs import generate_valid_pairs
from taxonomy.parser import parse_taxonomy
from taxonomy.validation import validate_taxonomy_nodes

LOGGER = logging.getLogger("tax_rulemining")

STEP1_OUTPUT_FILES = [
    "node_feature_sets.csv",
    "taxonomy_pairs.csv",
    "pipeline_metadata.csv",
    "node_lof_scores.csv",
    "pair_pseudo_labels.csv",
    "trajectory_pseudo_labels.csv",
    "step1_summary.json",
]


def write_step1_outputs(
    output_dir: Path,
    nodes: list[TaxonomyNode],
    pairs: list[dict[str, str | int]],
    metadata_rows: list[dict[str, str]],
    node_lof_rows: list[dict[str, str | int | float]],
    pseudo_label_rows: list[dict[str, str | int | float]],
    trajectory_pseudo_label_rows: list[dict[str, str]],
    trajectory_pseudo_label_fieldnames: list[str],
) -> None:
    node_rows = [
        {
            "node_id": node.node_id,
            "node_name": node.node_name,
            "parent_id": node.parent_id or "",
            "depth": node.depth,
            "is_leaf": str(node.is_leaf).lower(),
            "n_features": len(node.valid_features),
            "features": ";".join(node.valid_features),
        }
        for node in nodes
        if node.is_usable
    ]
    write_csv(
        output_dir / "node_feature_sets.csv",
        ["node_id", "node_name", "parent_id", "depth", "is_leaf", "n_features", "features"],
        node_rows,
    )
    write_csv(
        output_dir / "taxonomy_pairs.csv",
        ["pair_id", "node_a", "node_b", "n_features_a", "n_features_b", "features_a", "features_b"],
        pairs,
    )
    write_csv(output_dir / "pipeline_metadata.csv", ["category", "item", "details"], metadata_rows)
    write_csv(
        output_dir / "node_lof_scores.csv",
        [
            "trajectory_id",
            "node_id",
            "raw_lof",
            "lof_percentile_score",
            "n_features_used",
            "n_neighbors",
        ],
        node_lof_rows,
    )
    write_csv(
        output_dir / "pair_pseudo_labels.csv",
        [
            "trajectory_id",
            "pair_id",
            "node_a",
            "node_b",
            "score_a",
            "score_b",
            "pseudo_label_code",
            "pseudo_label",
        ],
        pseudo_label_rows,
    )
    write_csv(
        output_dir / "trajectory_pseudo_labels.csv",
        trajectory_pseudo_label_fieldnames,
        trajectory_pseudo_label_rows,
    )


def write_step1_summary(
    output_dir: Path,
    data_path: Path,
    taxonomy_path: Path,
    trajectory_id: str,
    schema: DatasetSchema,
    nodes: list[TaxonomyNode],
    pairs: list[dict[str, str | int]],
    node_summaries: list[NodeLofSummary],
    pseudo_label_rows: list[dict[str, str | int | float]],
    metadata_rows: list[dict[str, str]],
) -> None:
    usable_nodes = [node for node in nodes if node.is_usable]
    lof_scored_node_ids = [summary.node_id for summary in node_summaries]
    payload = {
        "step": "step1",
        "status": "completed",
        "inputs": {
            "data": str(data_path),
            "taxonomy": str(taxonomy_path),
            "trajectory_id": trajectory_id,
        },
        "dataset": {
            "row_count": schema.row_count,
            "column_count": len(schema.headers),
            "numeric_feature_count": len(schema.numeric_columns),
        },
        "lof_strategy": lof_strategy_summary(),
        "results": {
            "usable_node_count": len(usable_nodes),
            "valid_pair_count": len(pairs),
            "lof_scored_node_count": len(node_summaries),
            "pseudo_label_row_count": len(pseudo_label_rows),
            "metadata_event_count": len(metadata_rows),
            "lof_scored_nodes": lof_scored_node_ids,
        },
        "generated_files": {
            filename: str(output_dir / filename) for filename in STEP1_OUTPUT_FILES
        },
    }
    write_json(output_dir / "step1_summary.json", payload)


def run_step1(data_path: Path, taxonomy_path: Path, trajectory_id: str, output_dir: Path) -> None:
    progress = ProgressReporter()
    progress.stage("Step 1: loading dataset schema")
    schema = load_dataset_schema(data_path, trajectory_id)
    progress.stage("Step 1: loading dataset table")
    table = load_dataset_table(data_path, trajectory_id)
    progress.stage("Step 1: parsing taxonomy")
    nodes = parse_taxonomy(taxonomy_path)
    progress.stage("Step 1: validating taxonomy nodes")
    nodes, metadata_rows = validate_taxonomy_nodes(nodes, schema)
    progress.stage("Step 1: generating valid taxonomy pairs")
    pairs = generate_valid_pairs(nodes)
    progress.stage("Step 1: LOF scoring usable taxonomy nodes")
    lof_task = progress.task(
        "LOF node scoring",
        sum(1 for node in nodes if node.is_usable),
    )
    node_summaries, score_lookup = compute_node_lof_scores(
        table=table,
        nodes=nodes,
        trajectory_id_column=trajectory_id,
        metadata_rows=metadata_rows,
        progress_task=lof_task,
    )
    lof_task.complete("completed")
    node_lof_rows = [
        row
        for summary in sorted(node_summaries, key=lambda summary: summary.node_id)
        for row in summary.rows
    ]
    node_name_lookup = {node.node_id: node.node_name for node in nodes}
    progress.stage("Step 1: generating pairwise pseudo-labels")
    pseudo_task = progress.task("Pair pseudo-label generation", len(pairs))
    pseudo_label_rows = generate_pair_pseudo_labels(
        table.trajectory_ids,
        pairs,
        score_lookup,
        node_name_lookup,
        progress_task=pseudo_task,
    )
    pseudo_task.complete("completed")
    progress.stage("Step 1: building trajectory pseudo-label table")
    trajectory_pseudo_label_rows = build_trajectory_pseudo_label_table(
        table.trajectory_ids,
        pairs,
        pseudo_label_rows,
    )
    trajectory_pseudo_label_fieldnames = [
        "trajectory_id",
        *[str(pair["pair_id"]) for pair in pairs],
    ]
    write_step1_outputs(
        output_dir,
        nodes,
        pairs,
        metadata_rows,
        node_lof_rows,
        pseudo_label_rows,
        trajectory_pseudo_label_rows,
        trajectory_pseudo_label_fieldnames,
    )
    progress.stage("Step 1: writing summary")
    write_step1_summary(
        output_dir=output_dir,
        data_path=data_path,
        taxonomy_path=taxonomy_path,
        trajectory_id=trajectory_id,
        schema=schema,
        nodes=nodes,
        pairs=pairs,
        node_summaries=node_summaries,
        pseudo_label_rows=pseudo_label_rows,
        metadata_rows=metadata_rows,
    )

    LOGGER.info(
        "Prepared %d usable taxonomy nodes, %d valid node pairs, %d LOF-scored nodes, "
        "and %d pseudo-label rows from %d dataset rows.",
        sum(1 for node in nodes if node.is_usable),
        len(pairs),
        len(node_summaries),
        len(pseudo_label_rows),
        schema.row_count,
    )
