"""LOF scoring logic."""

from __future__ import annotations

import logging
from math import floor, sqrt
from typing import Any

from core.models import DatasetTable, NodeLofSummary, TaxonomyNode
from core.progress import ProgressTask

LOGGER = logging.getLogger("tax_rulemining")


def lof_strategy_summary() -> dict[str, object]:
    return {
        "estimator": "sklearn.neighbors.LocalOutlierFactor",
        "preprocessing": [
            "select node feature subset",
            "convert values to numeric",
            "median imputation for missing values",
            "drop constant features within node",
            "scale features with RobustScaler",
        ],
        "parameters": {
            "metric": "euclidean",
            "contamination": "auto",
            "novelty": False,
            "n_neighbors_rule": "if n_samples <= 6: n_samples - 1; else min(20, max(5, floor(sqrt(n_samples))))",
        },
        "score_outputs": {
            "raw_lof": "-negative_outlier_factor_",
            "percentile_normalization": "scipy.stats.rankdata(method='average') scaled to [0, 1]",
        },
    }


def require_scoring_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from scipy.stats import rankdata
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import RobustScaler
    except ImportError as error:
        raise ImportError(
            "LOF scoring requires numpy, scipy, and scikit-learn. "
            "Install the project dependencies first."
        ) from error
    return np, rankdata, LocalOutlierFactor, RobustScaler


def build_numeric_matrix(table: DatasetTable, features: list[str], np: Any) -> Any:
    matrix = np.empty((table.row_count, len(features)), dtype=float)
    matrix.fill(np.nan)
    for column_index, feature in enumerate(features):
        raw_values = table.columns[feature]
        converted = []
        for value in raw_values:
            stripped = value.strip()
            if stripped == "":
                converted.append(np.nan)
            else:
                converted.append(float(stripped))
        matrix[:, column_index] = np.asarray(converted, dtype=float)
    return matrix


def choose_n_neighbors(n_samples: int) -> int:
    if n_samples <= 1:
        return 0
    if n_samples <= 6:
        return n_samples - 1
    return min(20, max(5, floor(sqrt(n_samples))))


def compute_node_lof_scores(
    table: DatasetTable,
    nodes: list[TaxonomyNode],
    trajectory_id_column: str,
    metadata_rows: list[dict[str, str]],
    progress_task: ProgressTask | None = None,
) -> tuple[list[NodeLofSummary], dict[str, dict[str, float]]]:
    np, rankdata, LocalOutlierFactor, RobustScaler = require_scoring_dependencies()

    node_summaries: list[NodeLofSummary] = []
    score_lookup: dict[str, dict[str, float]] = {}
    trajectory_ids = table.columns[trajectory_id_column]

    for node in nodes:
        if not node.is_usable:
            continue

        candidate_features = list(node.valid_features)
        matrix = build_numeric_matrix(table, candidate_features, np)

        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        nan_rows, nan_columns = np.where(np.isnan(matrix))
        if nan_rows.size:
            matrix[nan_rows, nan_columns] = medians[nan_columns]

        feature_ranges = np.ptp(matrix, axis=0)
        kept_indices = [index for index, width in enumerate(feature_ranges) if width > 0.0]
        dropped_features = [
            feature for index, feature in enumerate(candidate_features) if index not in kept_indices
        ]
        for feature in dropped_features:
            LOGGER.warning(
                "Feature '%s' is constant within taxonomy node '%s' after imputation and "
                "will be dropped before LOF scoring.",
                feature,
                node.node_id,
            )
            metadata_rows.append(
                {"category": "constant_feature", "item": feature, "details": f"node={node.node_id}"}
            )

        if not kept_indices:
            LOGGER.warning(
                "Taxonomy node '%s' has no non-constant features after preprocessing and "
                "will be skipped for LOF scoring.",
                node.node_id,
            )
            metadata_rows.append(
                {
                    "category": "skipped_lof_node",
                    "item": node.node_id,
                    "details": "all candidate features were constant after preprocessing",
                }
            )
            continue

        matrix = matrix[:, kept_indices]
        n_samples = matrix.shape[0]
        n_neighbors = choose_n_neighbors(n_samples)
        if n_neighbors < 1:
            LOGGER.warning(
                "Taxonomy node '%s' cannot be scored with LOF because the dataset has fewer "
                "than 2 rows.",
                node.node_id,
            )
            metadata_rows.append(
                {
                    "category": "skipped_lof_node",
                    "item": node.node_id,
                    "details": "LOF requires at least 2 samples",
                }
            )
            continue

        scaler = RobustScaler()
        scaled_matrix = scaler.fit_transform(matrix)

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

        score_lookup[node.node_id] = {
            trajectory_id: float(score)
            for trajectory_id, score in zip(trajectory_ids, percentile_scores, strict=False)
        }

        rows = [
            {
                "trajectory_id": trajectory_id,
                "node_id": node.node_id,
                "raw_lof": float(raw_score),
                "lof_percentile_score": float(percentile_score),
                "n_features_used": matrix.shape[1],
                "n_neighbors": n_neighbors,
            }
            for trajectory_id, raw_score, percentile_score in zip(
                trajectory_ids,
                raw_lof,
                percentile_scores,
                strict=False,
            )
        ]
        node_summaries.append(
            NodeLofSummary(
                node_id=node.node_id,
                n_features_used=matrix.shape[1],
                n_neighbors=n_neighbors,
                rows=rows,
            )
        )
        if progress_task is not None:
            # Each usable taxonomy node advances the LOF stage by one unit.
            progress_task.advance(1, note=node.node_id)

    return node_summaries, score_lookup
