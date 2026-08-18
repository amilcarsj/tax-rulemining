"""Shared data models for the taxonomy-guided pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSchema:
    """Summary of the dataset columns relevant to taxonomy validation."""

    trajectory_id_column: str
    headers: list[str]
    numeric_columns: set[str]
    row_count: int


@dataclass
class DatasetTable:
    """In-memory dataset representation for scoring."""

    trajectory_ids: list[str]
    columns: dict[str, list[str]]
    row_count: int


@dataclass
class TaxonomyNode:
    """Parsed taxonomy node with computed feature metadata."""

    node_id: str
    node_name: str
    parent_id: str | None
    depth: int
    is_leaf: bool
    raw_features: list[str]
    descendant_features: list[str]
    ancestor_ids: tuple[str, ...]
    valid_features: list[str] = field(default_factory=list)
    skipped_features: list[str] = field(default_factory=list)
    invalid_numeric_features: list[str] = field(default_factory=list)
    is_usable: bool = False


@dataclass
class NodeLofSummary:
    """LOF scoring results for a single taxonomy node."""

    node_id: str
    n_features_used: int
    n_neighbors: int
    rows: list[dict[str, str | int | float]]
