"""Semantic meta-pattern helpers shared by Step 2 and Step 3."""

from __future__ import annotations

from collections import Counter


def parse_item(item: str) -> tuple[str, str]:
    return item.split("=", 1)


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    return pair_id.split("__x__", 1)


def label_type(label: str) -> str:
    if label == "Normal":
        return "normal"
    if label == "Hybrid":
        return "hybrid"
    if label.startswith("Uncommon "):
        return "uncommon"
    return "other"


def uncommon_node_from_label(label: str) -> str:
    if label.startswith("Uncommon "):
        return label[len("Uncommon ") :]
    return ""


def humanize_node_name(node_name: str) -> str:
    return node_name.replace("_", " ")


def semantic_counts_from_items(items: list[str] | tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        if not item:
            continue
        _, label = parse_item(item)
        node = uncommon_node_from_label(label)
        if node:
            counts[node] += 1
    return counts


def format_required_counts(required_counts: Counter[str] | dict[str, int]) -> str:
    return ";".join(
        f"{node}>={count}"
        for node, count in sorted(required_counts.items())
        if count > 0
    )


def parse_required_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value).split(";"):
        stripped = part.strip()
        if not stripped:
            continue
        if ">=" not in stripped:
            raise ValueError(f"Invalid meta-pattern requirement: {stripped}")
        node, count_text = stripped.split(">=", 1)
        counts[node] = int(count_text)
    return counts


def format_observed_counts(counts: Counter[str] | dict[str, int], nodes: list[str]) -> str:
    return ";".join(f"{node}={counts.get(node, 0)}" for node in nodes)


def row_uncommon_node_counts(row: dict[str, str], pair_ids: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for pair_id in pair_ids:
        label = str(row.get(pair_id, "")).strip()
        node = uncommon_node_from_label(label)
        if node:
            counts[node] += 1
    return counts


def pattern_satisfied(
    observed_counts: Counter[str] | dict[str, int],
    required_counts: Counter[str] | dict[str, int],
) -> bool:
    return all(observed_counts.get(node, 0) >= count for node, count in required_counts.items())


def readable_required_counts(required_counts: Counter[str] | dict[str, int]) -> str:
    parts: list[str] = []
    for node, count in sorted(required_counts.items()):
        node_name = humanize_node_name(node)
        if count == 1:
            parts.append(f"{node_name} is uncommon in at least 1 retained comparison")
        else:
            parts.append(f"{node_name} is uncommon in at least {count} retained comparisons")
    return " and ".join(parts)


def pair_contains_node(pair_id: str, node_id: str) -> bool:
    left_node, right_node = parse_pair_id(pair_id)
    return node_id in {left_node, right_node}
