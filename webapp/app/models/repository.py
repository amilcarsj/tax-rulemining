"""Read-only model layer over the generated taxonomy V2 pipeline outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


STEP1_DIRECTORY = ("taxonomy_v2", "step1")
STEP2_DIRECTORY = ("taxonomy_v2", "step2_top_nodes")
STEP3_DIRECTORY = ("taxonomy_v2", "step3_meta_patterns")

META_CONTRAST_GROUPS = [
    {
        "name": "near_miss",
        "label": "Near miss",
        "description": "Almost satisfies the meta-pattern, usually one missing uncommonness count away.",
    },
    {
        "name": "matched_non_pattern",
        "label": "Matched non-pattern",
        "description": "Does not satisfy the pattern but is matched by overall uncommonness complexity.",
    },
    {
        "name": "typical_normal",
        "label": "Typical normal",
        "description": "Low-uncommonness reference trajectories, preferably no uncommon and no hybrid labels.",
    },
]


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def percentage(value: float) -> float:
    return value * 100.0


def humanize(value: str) -> str:
    return value.replace("_", " ")


def pair_display(pair_id: str) -> str:
    return humanize(pair_id.replace("__x__", " x "))


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    if "__x__" not in pair_id:
        return pair_id, ""
    return pair_id.split("__x__", 1)


def parse_item(item: str) -> tuple[str, str]:
    if "=" not in item:
        return item, ""
    return item.split("=", 1)


def parse_required_counts(value: str) -> dict[str, int]:
    requirements: dict[str, int] = {}
    for part in value.split(";"):
        if ">=" not in part:
            continue
        node_id, count = part.split(">=", 1)
        node_id = node_id.strip()
        if node_id:
            requirements[node_id] = as_int(count)
    return requirements


def readable_item(item: str) -> str:
    pair_id, label = parse_item(item)
    comparison = pair_display(pair_id)
    if label.startswith("Uncommon "):
        node = humanize(label.removeprefix("Uncommon "))
        return f"{node} is uncommon in {comparison}"
    if label == "Hybrid":
        return f"Hybrid behaviour in {comparison}"
    if label.startswith("Normal"):
        return f"Normal behaviour in {comparison}"
    return f"{humanize(label)} in {comparison}" if label else pair_display(pair_id)


def readable_rule(target_item: str, antecedent_items: str) -> str:
    antecedents = [
        readable_item(item)
        for item in antecedent_items.split(";")
        if item
    ]
    if not antecedents:
        return f"THEN {readable_item(target_item)}."
    return f"IF {' AND '.join(antecedents)} THEN {readable_item(target_item)}."


def format_number(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 100:
        return f"{value:.1f}"
    if absolute_value >= 10:
        return f"{value:.2f}"
    if absolute_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_signed_number(value: float) -> str:
    if value > 0:
        return f"+{format_number(value)}"
    return format_number(value)


def retained_counterpart_nodes(node_id: str, pair_ids: set[str]) -> list[str]:
    counterparts: set[str] = set()
    for pair_id in pair_ids:
        left_node, right_node = parse_pair_id(pair_id)
        if left_node == node_id and right_node:
            counterparts.add(right_node)
        elif right_node == node_id and left_node:
            counterparts.add(left_node)
    return sorted(counterparts, key=humanize)


def retained_comparison_summary(
    required_counts: str,
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requirements = parse_required_counts(required_counts)
    pair_ids_by_node: dict[str, set[str]] = defaultdict(set)
    for row in comparison_rows:
        node_id = row.get("node_id", "")
        pair_id = row.get("pair_id", "")
        if node_id and pair_id:
            pair_ids_by_node[node_id].add(pair_id)

    summaries: list[dict[str, Any]] = []
    for node_id, required_count in requirements.items():
        comparisons = sorted(pair_ids_by_node.get(node_id, set()), key=pair_display)
        comparison_names = [pair_display(pair_id) for pair_id in comparisons]
        counterpart_nodes = retained_counterpart_nodes(node_id, set(comparisons))
        counterpart_names = [humanize(node) for node in counterpart_nodes]
        counterpart_text = ", ".join(counterpart_names)
        comparison_text = "; ".join(comparison_names)
        plural = "comparison" if required_count == 1 else "comparisons"
        summaries.append(
            {
                "node_id": node_id,
                "node_display": humanize(node_id),
                "required_count": required_count,
                "retained_node_names": counterpart_names,
                "retained_node_text": counterpart_text,
                "comparison_names": comparison_names,
                "comparison_text": comparison_text,
                "description": (
                    f"{humanize(node_id)} must be uncommon in at least "
                    f"{required_count} retained {plural} against "
                    f"{counterpart_text}: {comparison_text}"
                ),
            }
        )
    return summaries


def readable_pattern_with_retained_nodes(
    required_counts: str,
    comparison_rows: list[dict[str, Any]],
) -> str:
    summaries = retained_comparison_summary(required_counts, comparison_rows)
    pieces: list[str] = []
    for summary in summaries:
        required_count = summary["required_count"]
        plural = "comparison" if required_count == 1 else "comparisons"
        retained_node_text = summary["retained_node_text"] or "the retained nodes"
        pieces.append(
            f"{summary['node_display']} is uncommon in at least "
            f"{required_count} retained {plural} against {retained_node_text}"
        )
    return "; and ".join(pieces)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_taxonomy_tree(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Turn Step 1 node metadata into a tree that includes feature names."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        node_id = row.get("node_id", "")
        if not node_id:
            continue
        features = [feature for feature in row.get("features", "").split(";") if feature]
        nodes_by_id[node_id] = {
            "node_id": node_id,
            "node_name": row.get("node_name", node_id),
            "display_name": humanize(row.get("node_name", node_id)),
            "parent_id": row.get("parent_id", ""),
            "depth": as_int(row.get("depth")),
            "n_features": as_int(row.get("n_features")),
            "features": features,
            "children": [],
        }

    roots: list[dict[str, Any]] = []
    for node in nodes_by_id.values():
        parent = nodes_by_id.get(node["parent_id"])
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)

    def sort_nodes(nodes: list[dict[str, Any]]) -> None:
        nodes.sort(key=lambda node: (node["depth"], node["display_name"].lower()))
        for child_node in nodes:
            sort_nodes(child_node["children"])

    sort_nodes(roots)
    return roots


class ResultRepository:
    """Exposes semantic meta-pattern outputs as view-ready objects."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def dataset_names(self) -> list[str]:
        return sorted(
            directory.name
            for directory in self.output_root.iterdir()
            if directory.is_dir()
            and (directory / Path(*STEP1_DIRECTORY) / "step1_summary.json").exists()
            and (directory / Path(*STEP2_DIRECTORY) / "semantic_meta_patterns.csv").exists()
            and (directory / Path(*STEP3_DIRECTORY) / "meta_pattern_explanation_summary.csv").exists()
        )

    def assert_dataset(self, dataset: str) -> None:
        if dataset not in self.dataset_names():
            raise KeyError(dataset)

    def _path(self, dataset: str, *parts: str) -> Path:
        self.assert_dataset(dataset)
        return self.output_root / dataset / Path(*parts)

    @lru_cache(maxsize=64)
    def _csv(self, path: str) -> tuple[dict[str, str], ...]:
        return tuple(read_csv(Path(path)))

    @lru_cache(maxsize=64)
    def _json(self, path: str) -> dict[str, Any]:
        return read_json(Path(path))

    def csv(self, dataset: str, *parts: str) -> list[dict[str, str]]:
        return [dict(row) for row in self._csv(str(self._path(dataset, *parts)))]

    def summary(self, dataset: str, *parts: str) -> dict[str, Any]:
        return self._json(str(self._path(dataset, *parts)))

    def step1_csv(self, dataset: str, filename: str) -> list[dict[str, str]]:
        return self.csv(dataset, *STEP1_DIRECTORY, filename)

    def step2_csv(self, dataset: str, filename: str) -> list[dict[str, str]]:
        return self.csv(dataset, *STEP2_DIRECTORY, filename)

    def step3_csv(self, dataset: str, filename: str) -> list[dict[str, str]]:
        return self.csv(dataset, *STEP3_DIRECTORY, filename)

    def step1_summary(self, dataset: str) -> dict[str, Any]:
        return self.summary(dataset, *STEP1_DIRECTORY, "step1_summary.json")

    def step2_summary(self, dataset: str) -> dict[str, Any]:
        return self.summary(dataset, *STEP2_DIRECTORY, "step2_summary.json")

    def step3_summary(self, dataset: str) -> dict[str, Any]:
        return self.summary(dataset, *STEP3_DIRECTORY, "step3_summary.json")

    def taxonomy_rows(self, dataset: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.step1_csv(dataset, "node_feature_sets.csv"):
            node_id = row.get("node_id", "")
            features = [feature for feature in row.get("features", "").split(";") if feature]
            rows.append(
                {
                    **row,
                    "node_id": node_id,
                    "display_name": humanize(row.get("node_name", node_id)),
                    "depth_value": as_int(row.get("depth")),
                    "n_features_value": as_int(row.get("n_features")),
                    "features_list": features,
                }
            )
        return rows

    def node_lookup(self, dataset: str) -> dict[str, dict[str, Any]]:
        return {row["node_id"]: row for row in self.taxonomy_rows(dataset)}

    def pair_label_composition(self, dataset: str) -> list[dict[str, Any]]:
        counts = Counter()
        for row in self.step2_csv(dataset, "pair_label_summary.csv"):
            counts["Normal"] += as_int(row.get("n_normal"))
            counts["Hybrid"] += as_int(row.get("n_hybrid"))
            counts["Uncommon"] += as_int(row.get("n_uncommon_total"))
        total = sum(counts.values()) or 1
        return [
            {
                "label": label,
                "count": counts[label],
                "share": counts[label] / total,
                "share_pct": percentage(counts[label] / total),
            }
            for label in ("Normal", "Hybrid", "Uncommon")
        ]

    def patterns(self, dataset: str) -> list[dict[str, Any]]:
        explanations = {
            row["meta_pattern_id"]: row
            for row in self.step3_csv(dataset, "meta_pattern_explanation_summary.csv")
        }
        comparison_rows_by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.step3_csv(dataset, "meta_pattern_comparison_evidence.csv"):
            comparison_rows_by_pattern[row["meta_pattern_id"]].append(row)
        rows: list[dict[str, Any]] = []
        for row in self.step2_csv(dataset, "semantic_meta_patterns.csv"):
            explanation = explanations.get(row["meta_pattern_id"], {})
            coverage = as_float(row.get("coverage_ratio"))
            retained_summaries = retained_comparison_summary(
                row.get("required_uncommon_counts", ""),
                comparison_rows_by_pattern.get(row["meta_pattern_id"], []),
            )
            rows.append(
                {
                    **row,
                    "display_nodes": [
                        humanize(node)
                        for node in row.get("nodes", "").split(";")
                        if node
                    ],
                    "coverage_pct": percentage(coverage),
                    "n_covered_value": as_int(row.get("n_covered")),
                    "n_source_rules_value": as_int(row.get("n_source_rules")),
                    "importance_value": as_float(row.get("pattern_importance_score")),
                    "mean_confidence_pct": percentage(as_float(row.get("mean_source_confidence"))),
                    "mean_lift_value": as_float(row.get("mean_source_lift")),
                    "top_comparison_evidence": explanation.get("top_comparison_evidence", ""),
                    "top_feature_evidence": explanation.get("top_feature_evidence", ""),
                    "readable_pattern": row.get("readable_pattern", ""),
                    "retained_comparison_summaries": retained_summaries,
                    "retained_comparison_text": " ".join(
                        summary["description"] for summary in retained_summaries
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                -row["importance_value"],
                -row["n_source_rules_value"],
                -row["n_covered_value"],
                row["meta_pattern_id"],
            )
        )
        return rows

    def dashboard(self, dataset: str) -> dict[str, Any]:
        step1 = self.step1_summary(dataset)
        step2 = self.step2_summary(dataset)
        step3 = self.step3_summary(dataset)
        patterns = self.patterns(dataset)
        taxonomy_rows = self.taxonomy_rows(dataset)
        top_node_ids = [
            row["node_id"]
            for row in taxonomy_rows
            if row["depth_value"] == 1
        ]
        node_pattern_counts = Counter()
        for pattern in patterns:
            for node in pattern.get("nodes", "").split(";"):
                if node:
                    node_pattern_counts[node] += 1
        top_node_cards = [
            {
                "node_id": node_id,
                "display_name": humanize(node_id),
                "n_features": self.node_lookup(dataset).get(node_id, {}).get("n_features_value", 0),
                "n_patterns": node_pattern_counts[node_id],
            }
            for node_id in top_node_ids
        ]
        return {
            "dataset": dataset,
            "step1": step1,
            "step2": step2,
            "step3": step3,
            "labels": self.pair_label_composition(dataset),
            "taxonomy_tree": build_taxonomy_tree(self.step1_csv(dataset, "node_feature_sets.csv")),
            "top_node_cards": top_node_cards,
            "patterns": patterns[:8],
        }

    def pattern_detail(self, dataset: str, meta_pattern_id: str) -> dict[str, Any]:
        pattern = next(
            (
                row
                for row in self.patterns(dataset)
                if row["meta_pattern_id"] == meta_pattern_id
            ),
            None,
        )
        if pattern is None:
            raise KeyError(meta_pattern_id)

        node_lookup = self.node_lookup(dataset)
        contrast_rows = [
            row
            for row in self.step3_csv(dataset, "meta_pattern_contrast_groups.csv")
            if row["meta_pattern_id"] == meta_pattern_id
        ]
        group_counts = Counter(row["contrast_group"] for row in contrast_rows)
        group_reasons: dict[str, str] = {}
        for row in contrast_rows:
            group_name = row.get("contrast_group", "")
            if group_name and group_name not in group_reasons:
                group_reasons[group_name] = row.get("selection_reason", "")
        total_trajectories = as_int(self.step1_summary(dataset)["dataset"].get("row_count"), 1) or 1
        group_cards = [
            {
                "name": "covered",
                "label": "Target group",
                "description": "Trajectories satisfying the semantic meta-pattern.",
                "selection_reason": "satisfies_required_uncommon_counts",
                "n_trajectories": group_counts["covered"],
                "ratio_pct": percentage(group_counts["covered"] / total_trajectories),
            }
        ]
        for group in META_CONTRAST_GROUPS:
            group_cards.append(
                {
                    **group,
                    "selection_reason": group_reasons.get(group["name"], ""),
                    "n_trajectories": group_counts[group["name"]],
                    "ratio_pct": percentage(group_counts[group["name"]] / total_trajectories),
                }
            )

        comparison_rows = [
            {
                **row,
                "node_display": humanize(row.get("node_id", "")),
                "pair_display": pair_display(row.get("pair_id", "")),
                "contrast_group_label": row.get("contrast_group_label", humanize(row.get("contrast_group", ""))),
                "covered_ratio_pct": percentage(as_float(row.get("covered_uncommon_ratio"))),
                "contrast_ratio_pct": percentage(as_float(row.get("contrast_uncommon_ratio"))),
                "enrichment_value": as_float(row.get("covered_to_contrast_ratio")),
            }
            for row in self.step3_csv(dataset, "meta_pattern_comparison_evidence.csv")
            if row["meta_pattern_id"] == meta_pattern_id
        ]
        comparison_rows.sort(
            key=lambda row: (
                row["node_display"],
                -row["covered_ratio_pct"],
                row["pair_id"],
            )
        )
        retained_summaries = retained_comparison_summary(
            pattern.get("required_uncommon_counts", ""),
            comparison_rows,
        )

        feature_difference_rows = [
            {
                **row,
                "node_display": humanize(row.get("node_id", "")),
                "contrast_group_label": row.get("contrast_group_label", humanize(row.get("contrast_group", ""))),
                "difference_value": abs(as_float(row.get("covered_minus_contrast_median_value"))),
                "signed_difference_value": as_float(row.get("covered_minus_contrast_median_value")),
                "covered_median": as_float(row.get("covered_median_value")),
                "contrast_median": as_float(row.get("contrast_median_value")),
                "covered_median_label": format_number(as_float(row.get("covered_median_value"))),
                "contrast_median_label": format_number(as_float(row.get("contrast_median_value"))),
                "raw_difference_label": format_signed_number(
                    as_float(row.get("covered_minus_contrast_median_value"))
                ),
            }
            for row in self.step3_csv(dataset, "meta_pattern_feature_differences.csv")
            if row["meta_pattern_id"] == meta_pattern_id
        ]
        feature_difference_rows.sort(
            key=lambda row: (
                row["contrast_group"],
                -row["difference_value"],
                row["node_display"],
                row["feature"],
            )
        )
        maximum_difference_by_group: dict[str, float] = {}
        for row in feature_difference_rows:
            group_name = row.get("contrast_group", "")
            maximum_difference_by_group[group_name] = max(
                maximum_difference_by_group.get(group_name, 0.0),
                row["difference_value"],
            )
        for row in feature_difference_rows:
            maximum_difference = maximum_difference_by_group.get(row.get("contrast_group", ""), 1.0) or 1.0
            row["bar_pct"] = percentage(row["difference_value"] / maximum_difference)
            row_maximum_median = max(
                abs(row["covered_median"]),
                abs(row["contrast_median"]),
                1e-12,
            )
            row["covered_median_pct"] = percentage(abs(row["covered_median"]) / row_maximum_median)
            row["contrast_median_pct"] = percentage(abs(row["contrast_median"]) / row_maximum_median)

        feature_rows_by_node_and_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in feature_difference_rows:
            feature_rows_by_node_and_group[(row["node_id"], row["contrast_group"])].append(row)

        comparison_rows_by_node_and_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in comparison_rows:
            comparison_rows_by_node_and_group[(row["node_id"], row["contrast_group"])].append(row)

        taxonomy_groups: list[dict[str, Any]] = []
        for node_id in pattern.get("nodes", "").split(";"):
            if not node_id:
                continue
            node = node_lookup.get(node_id, {})
            contrast_groups = []
            for group in META_CONTRAST_GROUPS:
                features = feature_rows_by_node_and_group.get((node_id, group["name"]), [])
                comparisons = comparison_rows_by_node_and_group.get((node_id, group["name"]), [])
                contrast_groups.append(
                    {
                        **group,
                        "features": features,
                        "top_features": features[:8],
                        "comparisons": comparisons,
                    }
                )
            taxonomy_groups.append(
                {
                    "node_id": node_id,
                    "display_name": humanize(node_id),
                    "n_features": node.get("n_features_value", 0),
                    "contrast_groups": contrast_groups,
                }
            )

        top_features_by_contrast = []
        for group in META_CONTRAST_GROUPS:
            features = [
                row
                for row in feature_difference_rows
                if row.get("contrast_group") == group["name"]
            ]
            features.sort(key=lambda row: (-row["difference_value"], row["node_display"], row["feature"]))
            top_features_by_contrast.append({**group, "features": features[:12]})

        source_rules = [
            {
                **row,
                "confidence_pct": percentage(as_float(row.get("confidence"))),
                "lift_value": as_float(row.get("lift")),
                "support_count": as_int(row.get("rule_support_count")),
                "target_comparison_display": pair_display(row.get("target_pair", "")),
                "readable_rule": readable_rule(
                    row.get("target_item", ""),
                    row.get("antecedent_items", ""),
                ),
                "readable_antecedents": [
                    readable_item(item)
                    for item in row.get("antecedent_items", "").split(";")
                    if item
                ],
                "readable_target": readable_item(row.get("target_item", "")),
            }
            for row in self.step2_csv(dataset, "semantic_meta_pattern_source_rules.csv")
            if row["meta_pattern_id"] == meta_pattern_id
        ]
        source_rules.sort(
            key=lambda row: (
                -row["confidence_pct"],
                -row["lift_value"],
                -row["support_count"],
                row["rule_id"],
            )
        )

        explanation = next(
            (
                row
                for row in self.step3_csv(dataset, "meta_pattern_explanation_summary.csv")
                if row["meta_pattern_id"] == meta_pattern_id
            ),
            {},
        )
        return {
            "pattern": pattern,
            "group_cards": group_cards,
            "retained_comparison_summaries": retained_summaries,
            "taxonomy_groups": taxonomy_groups,
            "top_features_by_contrast": top_features_by_contrast,
            "source_rules": source_rules,
            "comparison_rows": comparison_rows,
            "explanation": explanation,
        }
