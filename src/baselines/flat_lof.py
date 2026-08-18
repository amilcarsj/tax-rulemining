"""Flat LOF baseline over all numeric movement features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.models import DatasetSchema, DatasetTable
from data_io.writers import write_csv, write_json
from scoring.lof import (
    build_numeric_matrix,
    choose_n_neighbors,
    lof_strategy_summary,
    require_scoring_dependencies,
)


FLAT_LOF_FILES = [
    "flat_lof_scores.csv",
    "flat_lof_feature_metadata.csv",
    "flat_lof_summary.json",
]


def _prepare_matrix(
    table: DatasetTable,
    features: list[str],
    np: Any,
) -> tuple[Any, list[str], list[dict[str, str | int | float]]]:
    matrix = build_numeric_matrix(table, features, np)
    metadata_rows: list[dict[str, str | int | float]] = []

    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    nan_rows, nan_columns = np.where(np.isnan(matrix))
    if nan_rows.size:
        matrix[nan_rows, nan_columns] = medians[nan_columns]

    feature_ranges = np.ptp(matrix, axis=0)
    kept_indices = [index for index, width in enumerate(feature_ranges) if width > 0.0]
    used_features = [features[index] for index in kept_indices]

    dropped = {features[index] for index in range(len(features)) if index not in kept_indices}
    for index, feature in enumerate(features):
        status = "used" if feature not in dropped else "dropped_constant_after_imputation"
        metadata_rows.append(
            {
                "feature": feature,
                "status": status,
                "median_imputation_value": float(medians[index]),
                "feature_range_after_imputation": float(feature_ranges[index]),
            }
        )

    return matrix[:, kept_indices], used_features, metadata_rows


def run_flat_lof_baseline(
    table: DatasetTable,
    schema: DatasetSchema,
    data_path: Path,
    output_dir: Path,
    trajectory_id_column: str,
    features: list[str],
    exclude_columns: list[str],
) -> dict[str, Any]:
    """Compute a single global LOF score using all selected numeric features."""
    np, rankdata, LocalOutlierFactor, RobustScaler = require_scoring_dependencies()

    if len(features) == 0:
        raise ValueError("Flat LOF baseline requires at least one numeric feature column.")

    matrix, used_features, metadata_rows = _prepare_matrix(table, features, np)
    if len(used_features) == 0:
        raise ValueError("Flat LOF baseline has no non-constant numeric features to score.")

    n_samples = table.row_count
    n_neighbors = choose_n_neighbors(n_samples)
    if n_neighbors < 1:
        raise ValueError("Flat LOF baseline requires at least two dataset rows.")

    scaled_matrix = RobustScaler().fit_transform(matrix)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric="euclidean",
        contamination="auto",
        novelty=False,
    )
    model.fit_predict(scaled_matrix)
    raw_lof = -model.negative_outlier_factor_

    if n_samples == 1:
        percentile_scores = np.zeros(1, dtype=float)
    else:
        ranks = rankdata(raw_lof, method="average")
        percentile_scores = (ranks - 1.0) / (n_samples - 1.0)

    score_rows = [
        {
            "trajectory_id": trajectory_id,
            "raw_lof": float(raw_score),
            "lof_percentile_score": float(percentile_score),
            "n_features_used": len(used_features),
            "n_neighbors": n_neighbors,
        }
        for trajectory_id, raw_score, percentile_score in zip(
            table.trajectory_ids,
            raw_lof,
            percentile_scores,
            strict=False,
        )
    ]

    write_csv(
        output_dir / "flat_lof_scores.csv",
        ["trajectory_id", "raw_lof", "lof_percentile_score", "n_features_used", "n_neighbors"],
        score_rows,
    )
    write_csv(
        output_dir / "flat_lof_feature_metadata.csv",
        ["feature", "status", "median_imputation_value", "feature_range_after_imputation"],
        metadata_rows,
    )

    summary = {
        "baseline": "flat_lof",
        "status": "completed",
        "inputs": {
            "data": str(data_path),
            "trajectory_id": trajectory_id_column,
            "excluded_columns": exclude_columns,
        },
        "dataset": {
            "row_count": schema.row_count,
            "column_count": len(schema.headers),
            "candidate_numeric_feature_count": len(features),
        },
        "lof_strategy": lof_strategy_summary(),
        "results": {
            "n_features_used": len(used_features),
            "n_neighbors": n_neighbors,
            "score_row_count": len(score_rows),
        },
        "generated_files": {filename: str(output_dir / filename) for filename in FLAT_LOF_FILES},
    }
    write_json(output_dir / "flat_lof_summary.json", summary)
    return summary
