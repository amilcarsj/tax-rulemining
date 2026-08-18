"""Shared helpers for baseline experiments."""

from __future__ import annotations

from collections.abc import Iterable

from core.models import DatasetSchema


def candidate_numeric_features(
    schema: DatasetSchema,
    trajectory_id_column: str,
    exclude_columns: Iterable[str],
) -> list[str]:
    """Return numeric feature columns after removing identifier-like columns."""
    excluded = {trajectory_id_column, *exclude_columns}
    return [
        column
        for column in schema.headers
        if column in schema.numeric_columns and column not in excluded
    ]


def finite_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a finite ratio, using 0 when the denominator is absent."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def item_feature(item: str) -> str:
    """Extract the raw feature name from a symbolic item such as feature=high."""
    return item.rsplit("=", 1)[0]


def itemset_has_unique_features(itemset: tuple[str, ...]) -> bool:
    """Association itemsets cannot contain two bins from the same raw feature."""
    features = [item_feature(item) for item in itemset]
    return len(features) == len(set(features))
