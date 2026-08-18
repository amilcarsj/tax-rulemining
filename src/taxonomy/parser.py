"""Taxonomy parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.models import TaxonomyNode


def normalize_node_id(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", name.strip())
    normalized = normalized.strip("_")
    return normalized or "node"


def ensure_unique_node_id(candidate: str, counts: dict[str, int]) -> str:
    counter = counts.get(candidate, 0)
    counts[candidate] = counter + 1
    if counter == 0:
        return candidate
    return f"{candidate}_{counter + 1}"


def parse_taxonomy(taxonomy_path: Path) -> list[TaxonomyNode]:
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Taxonomy JSON must contain a root object.")

    id_counts: dict[str, int] = {}
    nodes: list[TaxonomyNode] = []

    def visit(
        raw_node: dict[str, Any],
        parent_id: str | None,
        depth: int,
        ancestor_ids: tuple[str, ...],
    ) -> tuple[str, list[str]]:
        node_name = raw_node.get("name")
        if not isinstance(node_name, str) or not node_name.strip():
            raise ValueError("Every taxonomy node must have a non-empty string 'name'.")

        children = raw_node.get("children", [])
        if children is None:
            children = []
        if not isinstance(children, list):
            raise ValueError(f"Node '{node_name}' has a non-list 'children' field.")

        declared_features = raw_node.get("features", [])
        if declared_features is None:
            declared_features = []
        if not isinstance(declared_features, list) or any(
            not isinstance(feature, str) for feature in declared_features
        ):
            raise ValueError(f"Node '{node_name}' has an invalid 'features' list.")

        base_id = normalize_node_id(node_name)
        node_id = ensure_unique_node_id(base_id, id_counts)
        child_feature_union: set[str] = set()

        for child in children:
            if not isinstance(child, dict):
                raise ValueError(f"Node '{node_name}' has a child that is not an object.")
            _, child_features = visit(
                child,
                parent_id=node_id,
                depth=depth + 1,
                ancestor_ids=ancestor_ids + (node_id,),
            )
            child_feature_union.update(child_features)

        effective_features = sorted(set(declared_features) | child_feature_union)
        node = TaxonomyNode(
            node_id=node_id,
            node_name=node_name,
            parent_id=parent_id,
            depth=depth,
            is_leaf=len(children) == 0,
            raw_features=sorted(set(declared_features)),
            descendant_features=effective_features,
            ancestor_ids=ancestor_ids,
        )
        nodes.append(node)
        return node_id, effective_features

    visit(payload, parent_id=None, depth=0, ancestor_ids=())
    nodes.sort(key=lambda node: (node.depth, node.node_id))
    return nodes
