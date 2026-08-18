"""Valid taxonomy-node pair generation."""

from __future__ import annotations

from core.models import TaxonomyNode


def generate_valid_pairs(nodes: list[TaxonomyNode]) -> list[dict[str, str | int]]:
    usable_nodes = [node for node in nodes if node.is_usable]
    pairs: list[dict[str, str | int]] = []

    for index, node_a in enumerate(usable_nodes):
        features_a = set(node_a.valid_features)
        for node_b in usable_nodes[index + 1 :]:
            if node_a.node_id == node_b.node_id:
                continue
            if node_a.node_id in node_b.ancestor_ids or node_b.node_id in node_a.ancestor_ids:
                continue
            features_b = set(node_b.valid_features)
            if features_a & features_b:
                continue

            ordered_a, ordered_b = sorted([node_a, node_b], key=lambda node: node.node_id)
            pairs.append(
                {
                    "pair_id": f"{ordered_a.node_id}__x__{ordered_b.node_id}",
                    "node_a": ordered_a.node_id,
                    "node_b": ordered_b.node_id,
                    "n_features_a": len(ordered_a.valid_features),
                    "n_features_b": len(ordered_b.valid_features),
                    "features_a": ";".join(ordered_a.valid_features),
                    "features_b": ";".join(ordered_b.valid_features),
                }
            )

    pairs.sort(key=lambda row: str(row["pair_id"]))
    return pairs
