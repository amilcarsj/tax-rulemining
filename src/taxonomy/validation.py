"""Taxonomy validation against dataset schema."""

from __future__ import annotations

import logging

from core.models import DatasetSchema, TaxonomyNode

LOGGER = logging.getLogger("tax_rulemining")


def validate_taxonomy_nodes(
    nodes: list[TaxonomyNode],
    schema: DatasetSchema,
) -> tuple[list[TaxonomyNode], list[dict[str, str]]]:
    metadata_rows: list[dict[str, str]] = []
    dataset_columns = set(schema.headers)

    for node in nodes:
        missing_features = sorted(
            feature for feature in node.descendant_features if feature not in dataset_columns
        )
        non_numeric_features = sorted(
            feature
            for feature in node.descendant_features
            if feature in dataset_columns and feature not in schema.numeric_columns
        )
        valid_features = sorted(
            feature for feature in node.descendant_features if feature in schema.numeric_columns
        )

        node.skipped_features = missing_features
        node.invalid_numeric_features = non_numeric_features
        node.valid_features = valid_features
        node.is_usable = bool(valid_features)

        for feature in missing_features:
            LOGGER.warning(
                "Feature '%s' listed for taxonomy node '%s' is missing from the dataset and "
                "will be skipped.",
                feature,
                node.node_id,
            )
            metadata_rows.append(
                {"category": "missing_feature", "item": feature, "details": f"node={node.node_id}"}
            )

        for feature in non_numeric_features:
            LOGGER.warning(
                "Feature '%s' listed for taxonomy node '%s' is not numeric and will be skipped.",
                feature,
                node.node_id,
            )
            metadata_rows.append(
                {
                    "category": "non_numeric_feature",
                    "item": feature,
                    "details": f"node={node.node_id}",
                }
            )

        if not node.is_usable:
            LOGGER.warning(
                "Taxonomy node '%s' has no valid numeric features after validation and will "
                "be skipped.",
                node.node_id,
            )
            metadata_rows.append(
                {
                    "category": "skipped_node",
                    "item": node.node_id,
                    "details": "no valid numeric features available",
                }
            )

    return nodes, metadata_rows
