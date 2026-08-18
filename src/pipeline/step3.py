"""Step 3 taxonomy-level counterfactual matching and feature comparison."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.progress import ProgressReporter
from data_io.dataset import load_dataset_table
from data_io.writers import write_csv, write_json
from pipeline import meta_patterns as meta

LOGGER = logging.getLogger("tax_rulemining")

STEP3_RULE_LEVEL_OUTPUT_FILES = [
    "rule_counterexample_summary.csv",
    "all_real_counterexamples.csv",
    "all_counterexample_feature_comparisons.csv",
    "counterexample_feature_summary.csv",
    "rule_contrast_groups.csv",
    "rule_contrast_group_summary.csv",
    "rule_contrast_feature_summary.csv",
    "rule_contrast_feature_differences.csv",
]

STEP3_META_PATTERN_OUTPUT_FILES = [
    "meta_pattern_contrast_groups.csv",
    "meta_pattern_comparison_evidence.csv",
    "meta_pattern_feature_summary.csv",
    "meta_pattern_feature_differences.csv",
    "meta_pattern_explanation_summary.csv",
    "step3_summary.json",
]


def step3_output_files(config: Step3Config) -> list[str]:
    if config.include_rule_level_outputs:
        return [*STEP3_RULE_LEVEL_OUTPUT_FILES, *STEP3_META_PATTERN_OUTPUT_FILES]
    return list(STEP3_META_PATTERN_OUTPUT_FILES)

CONTRAST_GROUPS = [
    "rule_positive",
    "antecedent_only",
    "target_only",
    "neither",
]

META_CONTRAST_GROUPS = [
    {
        "name": "near_miss",
        "label": "Near miss",
        "description": (
            "Non-pattern trajectories that are closest to satisfying the meta-pattern, "
            "preferably exactly one missing uncommonness count away."
        ),
    },
    {
        "name": "matched_non_pattern",
        "label": "Matched non-pattern",
        "description": (
            "Non-pattern trajectories matched to the target group by overall "
            "uncommon-label complexity."
        ),
    },
    {
        "name": "typical_normal",
        "label": "Typical normal",
        "description": (
            "Non-pattern trajectories with no uncommon and no hybrid labels across "
            "the retained top-node comparisons, with a low-uncommonness fallback."
        ),
    },
]


@dataclass
class Step3Config:
    include_all_counterexamples: bool = True
    include_rule_level_outputs: bool = False


def parse_item(item: str) -> tuple[str, str]:
    return item.split("=", 1)


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    return pair_id.split("__x__", 1)


def truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() == "true"


def load_node_features(path: Path) -> dict[str, list[str]]:
    node_features: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node_id = str(row.get("node_id", "")).strip()
            features = [
                feature for feature in str(row.get("features", "")).split(";") if feature
            ]
            if node_id:
                node_features[node_id] = features
    return node_features


def load_node_depths(path: Path) -> dict[str, int]:
    node_depths: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node_id = str(row.get("node_id", "")).strip()
            depth = str(row.get("depth", "")).strip()
            if node_id and depth:
                node_depths[node_id] = int(depth)
    return node_depths


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return load_csv_rows(path)


def load_step2_retained_pair_ids(step2_dir: Path, fallback_pair_ids: list[str]) -> list[str]:
    summary_path = step2_dir / "step2_summary.json"
    if not summary_path.exists():
        return fallback_pair_ids
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    retained_pair_ids = summary.get("depth_filter", {}).get("retained_pair_ids", [])
    if not retained_pair_ids:
        return fallback_pair_ids
    available_pair_ids = set(fallback_pair_ids)
    return [pair_id for pair_id in retained_pair_ids if pair_id in available_pair_ids]


def nodes_and_feature_map(
    items: list[str],
    node_features: dict[str, list[str]],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    nodes: list[str] = []
    for item in items:
        if not item:
            continue
        pair_id, _ = parse_item(item)
        nodes.extend(parse_pair_id(pair_id))
    unique_nodes = list(dict.fromkeys(nodes))
    missing_nodes = [node for node in unique_nodes if node not in node_features]
    if missing_nodes:
        raise ValueError(
            "Step 3 could not resolve feature sets for antecedent nodes: "
            + ", ".join(missing_nodes)
        )

    feature_nodes: dict[str, list[str]] = defaultdict(list)
    for node in unique_nodes:
        for feature in node_features[node]:
            feature_nodes[feature].append(node)
    return unique_nodes, list(feature_nodes), dict(feature_nodes)


def rule_feature_sets(
    rule: dict[str, str],
    node_features: dict[str, list[str]],
) -> tuple[list[str], list[str], list[str], list[str], dict[str, list[str]], dict[str, str]]:
    antecedent_items = [item for item in str(rule["antecedent_items"]).split(";") if item]
    antecedent_nodes, antecedent_features, antecedent_feature_nodes = nodes_and_feature_map(
        antecedent_items, node_features
    )
    target_nodes, target_features, target_feature_nodes = nodes_and_feature_map(
        [str(rule["target_item"])], node_features
    )

    comparison_features = list(dict.fromkeys([*antecedent_features, *target_features]))
    comparison_feature_nodes: dict[str, list[str]] = {}
    feature_roles: dict[str, str] = {}
    antecedent_feature_set = set(antecedent_features)
    target_feature_set = set(target_features)
    for feature in comparison_features:
        comparison_feature_nodes[feature] = list(
            dict.fromkeys(
                [
                    *antecedent_feature_nodes.get(feature, []),
                    *target_feature_nodes.get(feature, []),
                ]
            )
        )
        if feature in antecedent_feature_set and feature in target_feature_set:
            feature_roles[feature] = "antecedent_and_target"
        elif feature in antecedent_feature_set:
            feature_roles[feature] = "antecedent"
        else:
            feature_roles[feature] = "target"
    return (
        antecedent_nodes,
        target_nodes,
        antecedent_features,
        comparison_features,
        comparison_feature_nodes,
        feature_roles,
    )


def fingerprint_match_statistics(
    positive_fingerprint: dict[str, str],
    counterexample_fingerprint: dict[str, str],
    comparison_pair_ids: list[str],
) -> tuple[int, int, float]:
    matching_pair_count = sum(
        positive_fingerprint.get(pair_id, "") == counterexample_fingerprint.get(pair_id, "")
        for pair_id in comparison_pair_ids
    )
    comparison_pair_count = len(comparison_pair_ids)
    similarity = (
        matching_pair_count / comparison_pair_count if comparison_pair_count else 1.0
    )
    return matching_pair_count, comparison_pair_count, similarity


def robust_standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(values, axis=0)
    filled = np.where(np.isnan(values), medians, values)
    q1 = np.quantile(filled, 0.25, axis=0)
    q3 = np.quantile(filled, 0.75, axis=0)
    scales = q3 - q1
    fallback_scales = np.std(filled, axis=0)
    scales = np.where(scales > 0, scales, fallback_scales)
    scales = np.where(scales > 0, scales, 1.0)
    return (filled - medians) / scales, medians, scales


def feature_matrix(
    table_columns: dict[str, list[str]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    raw_values = np.empty((len(table_columns[feature_names[0]]), len(feature_names)), dtype=float)
    for column_index, feature in enumerate(feature_names):
        try:
            parsed_values: list[float] = []
            for value in table_columns[feature]:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = float("nan")
                parsed_values.append(numeric_value if np.isfinite(numeric_value) else float("nan"))
            raw_values[:, column_index] = parsed_values
        except KeyError as error:
            raise ValueError(f"Step 3 feature '{feature}' is missing from the input dataset.") from error

    empty_features = [
        feature_names[column_index]
        for column_index in range(len(feature_names))
        if not np.isfinite(raw_values[:, column_index]).any()
    ]
    if empty_features:
        raise ValueError(
            "Step 3 cannot compare features with no numeric values: "
            + ", ".join(empty_features)
        )

    standardized_values, _, _ = robust_standardize(raw_values)
    filled_values = np.where(np.isnan(raw_values), np.nanmedian(raw_values, axis=0), raw_values)
    return filled_values, standardized_values


def fingerprint_items_by_trajectory(
    fingerprint_rows: list[dict[str, str]],
    trajectory_id: str,
) -> dict[str, set[str]]:
    items_by_trajectory_id: dict[str, set[str]] = {}
    for row in fingerprint_rows:
        trajectory_id_value = str(row[trajectory_id])
        items: set[str] = set()
        for pair_id, label in row.items():
            if pair_id == trajectory_id:
                continue
            cleaned_label = str(label).strip()
            if not cleaned_label or cleaned_label.lower() in {"nan", "none", "null"}:
                continue
            items.add(f"{pair_id}={cleaned_label}")
        items_by_trajectory_id[trajectory_id_value] = items
    return items_by_trajectory_id


def classify_rule_contrast_group(
    antecedent_satisfied: bool,
    target_satisfied: bool,
) -> str:
    if antecedent_satisfied and target_satisfied:
        return "rule_positive"
    if antecedent_satisfied and not target_satisfied:
        return "antecedent_only"
    if not antecedent_satisfied and target_satisfied:
        return "target_only"
    return "neither"


def finite_summary(values: np.ndarray) -> dict[str, object]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "mean": "",
            "median": "",
            "std": "",
            "q25": "",
            "q75": "",
        }
    return {
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "std": float(np.std(finite_values)),
        "q25": float(np.quantile(finite_values, 0.25)),
        "q75": float(np.quantile(finite_values, 0.75)),
    }


def finite_difference(left: object, right: object) -> float | str:
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def finite_abs(value: object) -> float | str:
    if value == "":
        return ""
    return abs(float(value))


def difference_direction(value: object) -> str:
    if value == "":
        return "undefined"
    numeric_value = float(value)
    if numeric_value > 0:
        return "higher_in_covered"
    if numeric_value < 0:
        return "lower_in_covered"
    return "same"


def ratio_with_status(numerator: float, denominator: float) -> tuple[float | str, str]:
    if denominator == 0 and numerator > 0:
        return "", "zero_contrast"
    if denominator == 0:
        return "", "zero_both"
    return numerator / denominator, "finite"


def profile_deficit(observed_counts: dict[str, int], required_counts: dict[str, int]) -> int:
    return sum(
        max(0, required_count - observed_counts.get(node, 0))
        for node, required_count in required_counts.items()
    )


def profile_pattern_evidence(observed_counts: dict[str, int], required_counts: dict[str, int]) -> int:
    return sum(
        min(observed_counts.get(node, 0), required_count)
        for node, required_count in required_counts.items()
    )


def select_meta_contrast_sets(
    profiles: list[dict[str, object]],
    covered_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    noncovered_profiles = [
        profile for profile in profiles if not bool(profile["is_covered"])
    ]
    contrast_sets = {
        group["name"]: [] for group in META_CONTRAST_GROUPS
    }
    selection_reasons = {
        group["name"]: "no_noncovered_candidates" for group in META_CONTRAST_GROUPS
    }
    if not noncovered_profiles:
        return contrast_sets, selection_reasons

    positive_deficits = sorted(
        {
            int(profile["pattern_deficit"])
            for profile in noncovered_profiles
            if int(profile["pattern_deficit"]) > 0
        }
    )
    if positive_deficits:
        near_miss_deficit = 1 if 1 in positive_deficits else positive_deficits[0]
        contrast_sets["near_miss"] = [
            str(profile["trajectory_id"])
            for profile in noncovered_profiles
            if int(profile["pattern_deficit"]) == near_miss_deficit
        ]
        selection_reasons["near_miss"] = (
            "one_missing_uncommonness_count"
            if near_miss_deficit == 1
            else f"closest_available_deficit_{near_miss_deficit}"
        )

    covered_profiles = [
        profile for profile in profiles if bool(profile["is_covered"])
    ]
    if covered_profiles:
        target_size = min(len(covered_profiles), len(noncovered_profiles))
        target_uncommon_median = float(
            np.median([float(profile["n_uncommon_total"]) for profile in covered_profiles])
        )
        target_hybrid_median = float(
            np.median([float(profile["n_hybrid_total"]) for profile in covered_profiles])
        )
        target_pattern_evidence_median = float(
            np.median([float(profile["pattern_evidence_count"]) for profile in covered_profiles])
        )
        matched_profiles = sorted(
            noncovered_profiles,
            key=lambda profile: (
                abs(float(profile["n_uncommon_total"]) - target_uncommon_median),
                abs(float(profile["n_hybrid_total"]) - target_hybrid_median),
                abs(float(profile["pattern_evidence_count"]) - target_pattern_evidence_median),
                str(profile["trajectory_id"]),
            ),
        )[:target_size]
        contrast_sets["matched_non_pattern"] = [
            str(profile["trajectory_id"]) for profile in matched_profiles
        ]
        selection_reasons["matched_non_pattern"] = (
            "nearest_nonpattern_rows_by_total_uncommon_hybrid_and_pattern_evidence"
        )

    strict_typical_profiles = [
        profile
        for profile in noncovered_profiles
        if int(profile["n_uncommon_total"]) == 0 and int(profile["n_hybrid_total"]) == 0
    ]
    if strict_typical_profiles:
        contrast_sets["typical_normal"] = [
            str(profile["trajectory_id"]) for profile in strict_typical_profiles
        ]
        selection_reasons["typical_normal"] = "zero_uncommon_and_zero_hybrid_labels"
    else:
        zero_uncommon_profiles = [
            profile for profile in noncovered_profiles if int(profile["n_uncommon_total"]) == 0
        ]
        if zero_uncommon_profiles:
            contrast_sets["typical_normal"] = [
                str(profile["trajectory_id"]) for profile in zero_uncommon_profiles
            ]
            selection_reasons["typical_normal"] = "zero_uncommon_labels"
        else:
            minimum_uncommon = min(int(profile["n_uncommon_total"]) for profile in noncovered_profiles)
            contrast_sets["typical_normal"] = [
                str(profile["trajectory_id"])
                for profile in noncovered_profiles
                if int(profile["n_uncommon_total"]) == minimum_uncommon
            ]
            selection_reasons["typical_normal"] = f"minimum_available_uncommon_count_{minimum_uncommon}"

    return contrast_sets, selection_reasons


def build_meta_pattern_explanations(
    meta_pattern_rows: list[dict[str, str]],
    table_columns: dict[str, list[str]],
    index_by_trajectory_id: dict[str, int],
    fingerprint_rows: list[dict[str, str]],
    trajectory_id_column: str,
    retained_pair_ids: list[str],
    node_features: dict[str, list[str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    contrast_group_rows: list[dict[str, object]] = []
    comparison_evidence_rows: list[dict[str, object]] = []
    feature_summary_rows: list[dict[str, object]] = []
    feature_difference_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []

    fingerprint_ids = [str(row[trajectory_id_column]) for row in fingerprint_rows]
    for pattern_row in meta_pattern_rows:
        meta_pattern_id = str(pattern_row["meta_pattern_id"])
        required_counts = meta.parse_required_counts(
            str(pattern_row["required_uncommon_counts"])
        )
        nodes = sorted(required_counts)
        profiles: list[dict[str, object]] = []
        profile_by_id: dict[str, dict[str, object]] = {}
        covered_ids: list[str] = []

        for row in fingerprint_rows:
            trajectory_id = str(row[trajectory_id_column])
            all_observed_counts = meta.row_uncommon_node_counts(row, retained_pair_ids)
            observed_counts = {
                node: all_observed_counts.get(node, 0) for node in nodes
            }
            observed_counts_text = meta.format_observed_counts(all_observed_counts, nodes)
            pattern_deficit = profile_deficit(observed_counts, required_counts)
            pattern_evidence_count = profile_pattern_evidence(
                observed_counts,
                required_counts,
            )
            n_hybrid_total = sum(
                1
                for pair_id in retained_pair_ids
                if str(row.get(pair_id, "")).strip() == "Hybrid"
            )
            n_normal_total = sum(
                1
                for pair_id in retained_pair_ids
                if str(row.get(pair_id, "")).strip() == "Normal"
            )
            n_pair_labels = sum(
                1 for pair_id in retained_pair_ids if str(row.get(pair_id, "")).strip()
            )
            is_covered = pattern_deficit == 0
            if is_covered:
                covered_ids.append(trajectory_id)
            profile = {
                "trajectory_id": trajectory_id,
                "is_covered": is_covered,
                "observed_uncommon_counts": observed_counts_text,
                "pattern_deficit": pattern_deficit,
                "pattern_evidence_count": pattern_evidence_count,
                "n_pair_labels": n_pair_labels,
                "n_uncommon_total": sum(all_observed_counts.values()),
                "n_hybrid_total": n_hybrid_total,
                "n_normal_total": n_normal_total,
            }
            profiles.append(profile)
            profile_by_id[trajectory_id] = profile

        contrast_sets, selection_reasons = select_meta_contrast_sets(profiles, covered_ids)
        for profile in profiles:
            if not bool(profile["is_covered"]):
                continue
            contrast_group_rows.append(
                {
                    "meta_pattern_id": meta_pattern_id,
                    "trajectory_id": profile["trajectory_id"],
                    "contrast_group": "covered",
                    "contrast_group_label": "Target group",
                    "is_covered": True,
                    "is_contrast_member": False,
                    "selection_reason": "satisfies_required_uncommon_counts",
                    "required_uncommon_counts": pattern_row["required_uncommon_counts"],
                    "observed_uncommon_counts": profile["observed_uncommon_counts"],
                    "pattern_deficit": profile["pattern_deficit"],
                    "pattern_evidence_count": profile["pattern_evidence_count"],
                    "n_pair_labels": profile["n_pair_labels"],
                    "n_uncommon_total": profile["n_uncommon_total"],
                    "n_hybrid_total": profile["n_hybrid_total"],
                    "n_normal_total": profile["n_normal_total"],
                }
            )
        for group in META_CONTRAST_GROUPS:
            group_name = group["name"]
            for trajectory_id in contrast_sets[group_name]:
                profile = profile_by_id[trajectory_id]
                contrast_group_rows.append(
                    {
                        "meta_pattern_id": meta_pattern_id,
                        "trajectory_id": trajectory_id,
                        "contrast_group": group_name,
                        "contrast_group_label": group["label"],
                        "is_covered": False,
                        "is_contrast_member": True,
                        "selection_reason": selection_reasons[group_name],
                        "required_uncommon_counts": pattern_row["required_uncommon_counts"],
                        "observed_uncommon_counts": profile["observed_uncommon_counts"],
                        "pattern_deficit": profile["pattern_deficit"],
                        "pattern_evidence_count": profile["pattern_evidence_count"],
                        "n_pair_labels": profile["n_pair_labels"],
                        "n_uncommon_total": profile["n_uncommon_total"],
                        "n_hybrid_total": profile["n_hybrid_total"],
                        "n_normal_total": profile["n_normal_total"],
                    }
                )

        default_contrast_group = next(
            (
                group["name"]
                for group in META_CONTRAST_GROUPS
                if contrast_sets[group["name"]]
            ),
            "",
        )
        covered_set = set(covered_ids)
        contrast_sets_by_name = {
            group_name: set(trajectory_ids)
            for group_name, trajectory_ids in contrast_sets.items()
        }
        pattern_comparison_rows_by_group: dict[str, list[dict[str, object]]] = {
            group["name"]: [] for group in META_CONTRAST_GROUPS
        }
        for node in nodes:
            expected_label = f"Uncommon {node}"
            for pair_id in retained_pair_ids:
                if not meta.pair_contains_node(pair_id, node):
                    continue
                covered_count = 0
                contrast_counts = {group["name"]: 0 for group in META_CONTRAST_GROUPS}
                for row in fingerprint_rows:
                    trajectory_id = str(row[trajectory_id_column])
                    has_node_label = str(row.get(pair_id, "")).strip() == expected_label
                    if not has_node_label:
                        continue
                    if trajectory_id in covered_set:
                        covered_count += 1
                    for group_name, contrast_set in contrast_sets_by_name.items():
                        if trajectory_id in contrast_set:
                            contrast_counts[group_name] += 1

                covered_ratio = covered_count / len(covered_ids) if covered_ids else 0.0
                for group in META_CONTRAST_GROUPS:
                    group_name = group["name"]
                    contrast_ids = contrast_sets[group_name]
                    contrast_count = contrast_counts[group_name]
                    contrast_ratio = contrast_count / len(contrast_ids) if contrast_ids else 0.0
                    enrichment_ratio, enrichment_status = ratio_with_status(
                        covered_ratio,
                        contrast_ratio,
                    )
                    evidence_row = {
                        "meta_pattern_id": meta_pattern_id,
                        "contrast_group": group_name,
                        "contrast_group_label": group["label"],
                        "selection_reason": selection_reasons[group_name],
                        "node_id": node,
                        "pair_id": pair_id,
                        "required_uncommon_count": required_counts[node],
                        "n_covered": len(covered_ids),
                        "n_contrast": len(contrast_ids),
                        "covered_uncommon_count": covered_count,
                        "contrast_uncommon_count": contrast_count,
                        "covered_uncommon_ratio": covered_ratio,
                        "contrast_uncommon_ratio": contrast_ratio,
                        "covered_to_contrast_ratio": enrichment_ratio,
                        "covered_to_contrast_ratio_status": enrichment_status,
                    }
                    comparison_evidence_rows.append(evidence_row)
                    pattern_comparison_rows_by_group[group_name].append(evidence_row)

        pattern_features = sorted(
            {
                feature
                for node in nodes
                for feature in node_features.get(node, [])
            }
        )
        pattern_feature_difference_rows_by_group: dict[str, list[dict[str, object]]] = {
            group["name"]: [] for group in META_CONTRAST_GROUPS
        }
        if pattern_features and covered_ids:
            raw_values, standardized_values = feature_matrix(table_columns, pattern_features)
            covered_indices = [index_by_trajectory_id[item] for item in covered_ids]
            contrast_indices_by_group = {
                group["name"]: [
                    index_by_trajectory_id[item] for item in contrast_sets[group["name"]]
                ]
                for group in META_CONTRAST_GROUPS
            }
            feature_index_by_name = {
                feature: index for index, feature in enumerate(pattern_features)
            }
            for node in nodes:
                for feature in node_features.get(node, []):
                    feature_index = feature_index_by_name[feature]
                    group_stats: dict[str, dict[str, object]] = {}
                    group_definitions = [
                        (
                            "covered",
                            "Target group",
                            covered_indices,
                            "satisfies_required_uncommon_counts",
                        )
                    ]
                    group_definitions.extend(
                        (
                            group["name"],
                            group["label"],
                            contrast_indices_by_group[group["name"]],
                            selection_reasons[group["name"]],
                        )
                        for group in META_CONTRAST_GROUPS
                    )
                    for group_name, group_label, group_indices, selection_reason in group_definitions:
                        if not group_indices:
                            continue
                        raw_stats = finite_summary(raw_values[group_indices, feature_index])
                        standardized_stats = finite_summary(
                            standardized_values[group_indices, feature_index]
                        )
                        group_stats[group_name] = {
                            "raw": raw_stats,
                            "standardized": standardized_stats,
                            "n_trajectories": len(group_indices),
                        }
                        feature_summary_rows.append(
                            {
                                "meta_pattern_id": meta_pattern_id,
                                "contrast_group": group_name,
                                "contrast_group_label": group_label,
                                "selection_reason": selection_reason,
                                "node_id": node,
                                "feature": feature,
                                "n_trajectories": len(group_indices),
                                "mean_value": raw_stats["mean"],
                                "median_value": raw_stats["median"],
                                "std_value": raw_stats["std"],
                                "q25_value": raw_stats["q25"],
                                "q75_value": raw_stats["q75"],
                                "mean_robust_standardized_value": standardized_stats["mean"],
                                "median_robust_standardized_value": standardized_stats["median"],
                            }
                        )

                    covered_stats = group_stats.get("covered")
                    if not covered_stats:
                        continue
                    covered_raw = covered_stats["raw"]
                    covered_standardized = covered_stats["standardized"]
                    for group in META_CONTRAST_GROUPS:
                        group_name = group["name"]
                        contrast_stats = group_stats.get(group_name)
                        if not contrast_stats:
                            continue
                        contrast_raw = contrast_stats["raw"]
                        contrast_standardized = contrast_stats["standardized"]
                        mean_difference = finite_difference(
                            covered_raw["mean"],
                            contrast_raw["mean"],
                        )
                        median_difference = finite_difference(
                            covered_raw["median"],
                            contrast_raw["median"],
                        )
                        standardized_mean_difference = finite_difference(
                            covered_standardized["mean"],
                            contrast_standardized["mean"],
                        )
                        standardized_median_difference = finite_difference(
                            covered_standardized["median"],
                            contrast_standardized["median"],
                        )
                        difference_row = {
                            "meta_pattern_id": meta_pattern_id,
                            "contrast_group": group_name,
                            "contrast_group_label": group["label"],
                            "selection_reason": selection_reasons[group_name],
                            "node_id": node,
                            "feature": feature,
                            "n_covered": len(covered_indices),
                            "n_contrast": contrast_stats["n_trajectories"],
                            "covered_mean_value": covered_raw["mean"],
                            "contrast_mean_value": contrast_raw["mean"],
                            "covered_median_value": covered_raw["median"],
                            "contrast_median_value": contrast_raw["median"],
                            "covered_minus_contrast_mean_value": mean_difference,
                            "covered_minus_contrast_median_value": median_difference,
                            "covered_mean_robust_standardized_value": covered_standardized["mean"],
                            "contrast_mean_robust_standardized_value": contrast_standardized["mean"],
                            "covered_median_robust_standardized_value": covered_standardized["median"],
                            "contrast_median_robust_standardized_value": contrast_standardized["median"],
                            "covered_minus_contrast_mean_robust_standardized_value": (
                                standardized_mean_difference
                            ),
                            "absolute_covered_minus_contrast_mean_robust_standardized_value": finite_abs(
                                standardized_mean_difference
                            ),
                            "covered_minus_contrast_median_robust_standardized_value": (
                                standardized_median_difference
                            ),
                            "absolute_covered_minus_contrast_median_robust_standardized_value": finite_abs(
                                standardized_median_difference
                            ),
                            "median_difference_direction": difference_direction(
                                standardized_median_difference
                            ),
                        }
                        feature_difference_rows.append(difference_row)
                        pattern_feature_difference_rows_by_group[group_name].append(difference_row)

        top_comparison_evidence_by_group: dict[str, str] = {}
        top_feature_evidence_by_group: dict[str, str] = {}
        for group in META_CONTRAST_GROUPS:
            group_name = group["name"]
            top_comparisons = sorted(
                pattern_comparison_rows_by_group[group_name],
                key=lambda row: (
                    -float(row["covered_uncommon_ratio"]),
                    -int(row["covered_uncommon_count"]),
                    str(row["node_id"]),
                    str(row["pair_id"]),
                ),
            )[:5]
            top_features = sorted(
                [
                    row
                    for row in pattern_feature_difference_rows_by_group[group_name]
                    if row["absolute_covered_minus_contrast_median_robust_standardized_value"] != ""
                ],
                key=lambda row: (
                    -float(row["absolute_covered_minus_contrast_median_robust_standardized_value"]),
                    str(row["node_id"]),
                    str(row["feature"]),
                ),
            )[:8]
            top_comparison_evidence_by_group[group_name] = ";".join(
                (
                    f"{row['node_id']} via {row['pair_id']} "
                    f"({float(row['covered_uncommon_ratio']):.1%} target, "
                    f"{float(row['contrast_uncommon_ratio']):.1%} {group['label'].lower()})"
                )
                for row in top_comparisons
            )
            top_feature_evidence_by_group[group_name] = ";".join(
                (
                    f"{row['node_id']}:{row['feature']} "
                    f"{row['median_difference_direction']} "
                    f"({float(row['absolute_covered_minus_contrast_median_robust_standardized_value']):.3f} robust median units)"
                )
                for row in top_features
            )

        explanation_rows.append(
            {
                "meta_pattern_id": meta_pattern_id,
                "readable_pattern": pattern_row.get("readable_pattern", ""),
                "required_uncommon_counts": pattern_row["required_uncommon_counts"],
                "nodes": ";".join(nodes),
                "n_covered": len(covered_ids),
                "n_near_miss": len(contrast_sets["near_miss"]),
                "n_matched_non_pattern": len(contrast_sets["matched_non_pattern"]),
                "n_typical_normal": len(contrast_sets["typical_normal"]),
                "coverage_ratio": len(covered_ids) / len(fingerprint_ids) if fingerprint_ids else 0.0,
                "n_source_rules": pattern_row.get("n_source_rules", ""),
                "default_contrast_group": default_contrast_group,
                "contrast_selection_reasons": ";".join(
                    f"{group['name']}={selection_reasons[group['name']]}"
                    for group in META_CONTRAST_GROUPS
                ),
                "top_comparison_evidence": top_comparison_evidence_by_group.get(
                    default_contrast_group,
                    "",
                ),
                "top_feature_evidence": top_feature_evidence_by_group.get(
                    default_contrast_group,
                    "",
                ),
                "top_comparison_evidence_near_miss": top_comparison_evidence_by_group["near_miss"],
                "top_feature_evidence_near_miss": top_feature_evidence_by_group["near_miss"],
                "top_comparison_evidence_matched_non_pattern": top_comparison_evidence_by_group[
                    "matched_non_pattern"
                ],
                "top_feature_evidence_matched_non_pattern": top_feature_evidence_by_group[
                    "matched_non_pattern"
                ],
                "top_comparison_evidence_typical_normal": top_comparison_evidence_by_group[
                    "typical_normal"
                ],
                "top_feature_evidence_typical_normal": top_feature_evidence_by_group[
                    "typical_normal"
                ],
                "covered_trajectory_ids": ";".join(covered_ids),
                "near_miss_trajectory_ids": ";".join(contrast_sets["near_miss"]),
                "matched_non_pattern_trajectory_ids": ";".join(
                    contrast_sets["matched_non_pattern"]
                ),
                "typical_normal_trajectory_ids": ";".join(contrast_sets["typical_normal"]),
            }
        )

    comparison_evidence_rows.sort(
        key=lambda row: (
            str(row["meta_pattern_id"]),
            str(row["contrast_group"]),
            str(row["node_id"]),
            -float(row["covered_uncommon_ratio"]),
            str(row["pair_id"]),
        )
    )
    feature_summary_rows.sort(
        key=lambda row: (
            str(row["meta_pattern_id"]),
            str(row["contrast_group"]),
            str(row["node_id"]),
            str(row["feature"]),
        )
    )
    feature_difference_rows.sort(
        key=lambda row: (
            str(row["meta_pattern_id"]),
            str(row["contrast_group"]),
            -float(row["absolute_covered_minus_contrast_median_robust_standardized_value"])
            if row["absolute_covered_minus_contrast_median_robust_standardized_value"] != ""
            else 0.0,
            str(row["node_id"]),
            str(row["feature"]),
        )
    )
    return (
        contrast_group_rows,
        comparison_evidence_rows,
        feature_summary_rows,
        feature_difference_rows,
        explanation_rows,
    )


def run_step3(
    data_path: Path,
    step1_dir: Path,
    step2_dir: Path,
    output_dir: Path,
    trajectory_id: str,
    config: Step3Config,
) -> None:
    progress = ProgressReporter()
    progress.stage("Step 3: loading original feature table")
    table = load_dataset_table(data_path, trajectory_id)
    index_by_trajectory_id = {
        trajectory_id_value: index for index, trajectory_id_value in enumerate(table.trajectory_ids)
    }

    progress.stage("Step 3: loading taxonomy fingerprints, feature groups, and association rules")
    node_feature_sets_path = step1_dir / "node_feature_sets.csv"
    node_features = load_node_features(node_feature_sets_path)
    node_depths = load_node_depths(node_feature_sets_path)
    fingerprint_rows = load_csv_rows(step1_dir / "trajectory_pseudo_labels.csv")
    if not fingerprint_rows:
        raise ValueError("Step 3 requires at least one Step 1 trajectory fingerprint.")
    fingerprint_pair_ids = [pair_id for pair_id in fingerprint_rows[0] if pair_id != trajectory_id]
    same_depth_fingerprint_pair_ids: list[str] = []
    for pair_id in fingerprint_pair_ids:
        left_node, right_node = parse_pair_id(pair_id)
        left_depth = node_depths.get(left_node)
        right_depth = node_depths.get(right_node)
        if left_depth is None or right_depth is None:
            raise ValueError(
                f"Step 3 could not resolve taxonomy depths for fingerprint pair '{pair_id}'."
            )
        if left_depth == right_depth:
            same_depth_fingerprint_pair_ids.append(pair_id)
    fingerprint_by_trajectory_id = {
        str(row[trajectory_id]): row for row in fingerprint_rows
    }
    fingerprint_trajectory_ids = [str(row[trajectory_id]) for row in fingerprint_rows]
    fingerprint_items_by_id = fingerprint_items_by_trajectory(fingerprint_rows, trajectory_id)
    retained_fingerprint_pair_ids = load_step2_retained_pair_ids(
        step2_dir,
        same_depth_fingerprint_pair_ids,
    )
    missing_fingerprint_ids = [
        trajectory_id_value
        for trajectory_id_value in fingerprint_trajectory_ids
        if trajectory_id_value not in index_by_trajectory_id
    ]
    if missing_fingerprint_ids:
        raise ValueError(
            "Step 3 could not find fingerprint trajectories in the original data: "
            + ", ".join(sorted(set(missing_fingerprint_ids))[:10])
        )
    rules = load_csv_rows(step2_dir / "high_level_association_rules.csv")
    rule_level_rules = rules if config.include_rule_level_outputs else []
    coverage_rows = load_csv_rows(step2_dir / "high_level_rule_coverage.csv")
    meta_pattern_rows = load_optional_csv_rows(step2_dir / "semantic_meta_patterns.csv")

    coverage_by_rule: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in coverage_rows:
        coverage_by_rule[str(row["rule_id"])].append(row)

    summary_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    contrast_group_rows: list[dict[str, object]] = []
    contrast_group_summary_rows: list[dict[str, object]] = []
    contrast_feature_summary_rows: list[dict[str, object]] = []
    contrast_feature_difference_rows: list[dict[str, object]] = []
    contrast_group_ids_by_rule: dict[tuple[str, str], list[str]] = defaultdict(list)
    contrast_feature_stats: dict[tuple[str, str, str], dict[str, object]] = {}
    feature_summary_accumulators: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    feature_nodes_by_rule: dict[tuple[str, str], list[str]] = {}
    feature_roles_by_rule: dict[tuple[str, str], str] = {}

    if config.include_rule_level_outputs:
        progress.stage("Step 3: matching rule-positive trajectories to real counterexamples")
    else:
        progress.stage("Step 3: skipping rule-level counterexample matching")
    matching_task = progress.task("Counterexample matching", len(rule_level_rules))
    for rule in rule_level_rules:
        rule_id = str(rule["rule_id"])
        (
            antecedent_nodes,
            target_nodes,
            antecedent_features,
            comparison_features,
            feature_nodes,
            feature_roles,
        ) = rule_feature_sets(
            rule, node_features
        )
        antecedent_items = tuple(
            item for item in str(rule["antecedent_items"]).split(";") if item
        )
        antecedent_pair_ids = {parse_item(item)[0] for item in antecedent_items}
        target_item = str(rule["target_item"])
        target_pair_id, target_label = parse_item(target_item)
        comparison_pair_ids = [
            pair_id
            for pair_id in retained_fingerprint_pair_ids
            if pair_id not in antecedent_pair_ids and pair_id != target_pair_id
        ]
        for trajectory_id_value in fingerprint_trajectory_ids:
            trajectory_items = fingerprint_items_by_id[trajectory_id_value]
            antecedent_satisfied = all(
                item in trajectory_items for item in antecedent_items
            )
            target_satisfied = target_item in trajectory_items
            contrast_group = classify_rule_contrast_group(
                antecedent_satisfied, target_satisfied
            )
            contrast_group_ids_by_rule[(rule_id, contrast_group)].append(trajectory_id_value)
            contrast_group_rows.append(
                {
                    "rule_id": rule_id,
                    "trajectory_id": trajectory_id_value,
                    "contrast_group": contrast_group,
                    "antecedent_satisfied": antecedent_satisfied,
                    "target_satisfied": target_satisfied,
                    "target_item": target_item,
                    "target_pair": target_pair_id,
                    "target_label": target_label,
                    "antecedent_items": ";".join(antecedent_items),
                }
            )
        coverage = coverage_by_rule.get(rule_id, [])
        positive_ids = [
            str(row["trajectory_id"])
            for row in coverage
            if truthy(row.get("antecedent_satisfied")) and truthy(row.get("target_satisfied"))
        ]
        counterexample_ids = [
            str(row["trajectory_id"])
            for row in coverage
            if truthy(row.get("antecedent_satisfied")) and not truthy(row.get("target_satisfied"))
        ]
        missing_ids = [
            item
            for item in [*positive_ids, *counterexample_ids]
            if item not in index_by_trajectory_id or item not in fingerprint_by_trajectory_id
        ]
        if missing_ids:
            raise ValueError(
                "Step 3 could not find rule-covered trajectories in both the original data and "
                "the Step 1 fingerprint: " + ", ".join(sorted(set(missing_ids))[:10])
            )
        positive_pairs = [(item, index_by_trajectory_id[item]) for item in positive_ids]
        counterexample_pairs = [(item, index_by_trajectory_id[item]) for item in counterexample_ids]

        status = "matched"
        if not positive_pairs:
            status = "no_rule_positive_trajectories"
        elif not counterexample_pairs:
            status = "no_real_counterexamples"
        elif not antecedent_features:
            status = "no_antecedent_features"

        comparison_raw_values, comparison_standardized_values = feature_matrix(
            table.columns, comparison_features
        )
        antecedent_standardized_values: np.ndarray | None = None
        if antecedent_features:
            _, antecedent_standardized_values = feature_matrix(
                table.columns, antecedent_features
            )

        for contrast_group in CONTRAST_GROUPS:
            group_ids = contrast_group_ids_by_rule[(rule_id, contrast_group)]
            group_indices = [index_by_trajectory_id[item] for item in group_ids]
            contrast_group_summary_rows.append(
                {
                    "rule_id": rule_id,
                    "target_item": target_item,
                    "antecedent_items": ";".join(antecedent_items),
                    "contrast_group": contrast_group,
                    "n_trajectories": len(group_ids),
                    "trajectory_ratio": (
                        len(group_ids) / len(fingerprint_trajectory_ids)
                        if fingerprint_trajectory_ids
                        else 0.0
                    ),
                    "trajectory_ids": ";".join(group_ids),
                }
            )
            if not group_indices:
                continue
            for feature_index, feature in enumerate(comparison_features):
                feature_nodes_by_rule[(rule_id, feature)] = feature_nodes[feature]
                feature_roles_by_rule[(rule_id, feature)] = feature_roles[feature]
                raw_stats = finite_summary(comparison_raw_values[group_indices, feature_index])
                standardized_stats = finite_summary(
                    comparison_standardized_values[group_indices, feature_index]
                )
                contrast_feature_stats[(rule_id, feature, contrast_group)] = {
                    "n_trajectories": len(group_indices),
                    "mean_value": raw_stats["mean"],
                    "mean_robust_standardized_value": standardized_stats["mean"],
                }
                contrast_feature_summary_rows.append(
                    {
                        "rule_id": rule_id,
                        "target_item": target_item,
                        "antecedent_items": ";".join(antecedent_items),
                        "contrast_group": contrast_group,
                        "feature": feature,
                        "feature_role": feature_roles[feature],
                        "feature_nodes": ";".join(feature_nodes[feature]),
                        "n_trajectories": len(group_indices),
                        "mean_value": raw_stats["mean"],
                        "median_value": raw_stats["median"],
                        "std_value": raw_stats["std"],
                        "q25_value": raw_stats["q25"],
                        "q75_value": raw_stats["q75"],
                        "mean_robust_standardized_value": standardized_stats["mean"],
                        "median_robust_standardized_value": standardized_stats["median"],
                    }
                )

        for feature in comparison_features:
            positive_stats = contrast_feature_stats.get((rule_id, feature, "rule_positive"))
            if not positive_stats:
                continue
            for contrast_group in CONTRAST_GROUPS[1:]:
                group_stats = contrast_feature_stats.get((rule_id, feature, contrast_group))
                if not group_stats:
                    continue
                positive_mean = float(positive_stats["mean_value"])
                group_mean = float(group_stats["mean_value"])
                positive_standardized_mean = float(
                    positive_stats["mean_robust_standardized_value"]
                )
                group_standardized_mean = float(
                    group_stats["mean_robust_standardized_value"]
                )
                raw_mean_difference = positive_mean - group_mean
                standardized_mean_difference = (
                    positive_standardized_mean - group_standardized_mean
                )
                contrast_feature_difference_rows.append(
                    {
                        "rule_id": rule_id,
                        "target_item": target_item,
                        "antecedent_items": ";".join(antecedent_items),
                        "contrast_group": contrast_group,
                        "feature": feature,
                        "feature_role": feature_roles[feature],
                        "feature_nodes": ";".join(feature_nodes[feature]),
                        "n_rule_positive": positive_stats["n_trajectories"],
                        "n_contrast_group": group_stats["n_trajectories"],
                        "rule_positive_mean_value": positive_mean,
                        "contrast_group_mean_value": group_mean,
                        "rule_positive_minus_contrast_mean_value": raw_mean_difference,
                        "absolute_rule_positive_minus_contrast_mean_value": abs(
                            raw_mean_difference
                        ),
                        "rule_positive_mean_robust_standardized_value": (
                            positive_standardized_mean
                        ),
                        "contrast_group_mean_robust_standardized_value": (
                            group_standardized_mean
                        ),
                        "rule_positive_minus_contrast_mean_robust_standardized_value": (
                            standardized_mean_difference
                        ),
                        "absolute_rule_positive_minus_contrast_mean_robust_standardized_value": abs(
                            standardized_mean_difference
                        ),
                    }
                )

        pair_count = 0
        selected_similarities: list[float] = []
        selected_mismatch_counts: list[int] = []
        if status == "matched":
            if antecedent_standardized_values is None:
                raise RuntimeError(
                    "Step 3 expected antecedent features for matched counterexample search."
                )
            for positive_id, positive_index in positive_pairs:
                candidate_scores: list[tuple[int, float, str, int, int, float]] = []
                for counterexample_id, counterexample_index in counterexample_pairs:
                    matching_pair_count, total_pair_count, fingerprint_similarity = (
                        fingerprint_match_statistics(
                            fingerprint_by_trajectory_id[positive_id],
                            fingerprint_by_trajectory_id[counterexample_id],
                            comparison_pair_ids,
                        )
                    )
                    label_mismatch_count = total_pair_count - matching_pair_count
                    antecedent_feature_distance = float(
                        np.linalg.norm(
                            antecedent_standardized_values[positive_index]
                            - antecedent_standardized_values[counterexample_index]
                        )
                    )
                    candidate_scores.append(
                        (
                            label_mismatch_count,
                            antecedent_feature_distance,
                            counterexample_id,
                            counterexample_index,
                            matching_pair_count,
                            fingerprint_similarity,
                        )
                    )
                candidate_scores.sort(key=lambda score: (score[0], score[1], score[2]))
                for rank, candidate_score in enumerate(candidate_scores, start=1):
                    (
                        label_mismatch_count,
                        antecedent_feature_distance,
                        counterexample_id,
                        counterexample_index,
                        matching_pair_count,
                        fingerprint_similarity,
                    ) = candidate_score
                    pair_count += 1
                    selected_similarities.append(fingerprint_similarity)
                    selected_mismatch_counts.append(label_mismatch_count)
                    neighbor_rows.append(
                        {
                            "rule_id": rule_id,
                            "target_item": rule["target_item"],
                            "positive_trajectory_id": positive_id,
                            "counterexample_trajectory_id": counterexample_id,
                            "counterexample_rank": rank,
                            "fingerprint_matching_pair_count": matching_pair_count,
                            "fingerprint_comparison_pair_count": len(comparison_pair_ids),
                            "fingerprint_similarity": fingerprint_similarity,
                            "fingerprint_label_mismatch_count": label_mismatch_count,
                            "antecedent_feature_distance": antecedent_feature_distance,
                            "antecedent_nodes": ";".join(antecedent_nodes),
                            "target_nodes": ";".join(target_nodes),
                            "antecedent_feature_count": len(antecedent_features),
                            "comparison_feature_count": len(comparison_features),
                        }
                    )
                    for feature_index, feature in enumerate(comparison_features):
                        feature_nodes_by_rule[(rule_id, feature)] = feature_nodes[feature]
                        feature_roles_by_rule[(rule_id, feature)] = feature_roles[feature]
                        raw_difference = float(
                            comparison_raw_values[positive_index, feature_index]
                            - comparison_raw_values[counterexample_index, feature_index]
                        )
                        standardized_difference = float(
                            comparison_standardized_values[positive_index, feature_index]
                            - comparison_standardized_values[counterexample_index, feature_index]
                        )
                        comparison_rows.append(
                            {
                                "rule_id": rule_id,
                                "positive_trajectory_id": positive_id,
                                "counterexample_trajectory_id": counterexample_id,
                                "counterexample_rank": rank,
                                "feature": feature,
                                "feature_role": feature_roles[feature],
                                "feature_nodes": ";".join(feature_nodes[feature]),
                                "positive_value": comparison_raw_values[positive_index, feature_index],
                                "counterexample_value": comparison_raw_values[counterexample_index, feature_index],
                                "raw_difference": raw_difference,
                                "robust_standardized_difference": standardized_difference,
                                "absolute_robust_standardized_difference": abs(standardized_difference),
                            }
                        )
                        feature_summary_accumulators[(rule_id, feature)].append(
                            {
                                "positive_value": float(comparison_raw_values[positive_index, feature_index]),
                                "counterexample_value": float(comparison_raw_values[counterexample_index, feature_index]),
                                "raw_difference": raw_difference,
                                "absolute_robust_standardized_difference": abs(standardized_difference),
                            }
                        )

        summary_rows.append(
            {
                "rule_id": rule_id,
                "target_item": rule["target_item"],
                "antecedent_items": rule["antecedent_items"],
                "antecedent_nodes": ";".join(antecedent_nodes),
                "target_nodes": ";".join(target_nodes),
                "antecedent_feature_count": len(antecedent_features),
                "comparison_feature_count": len(comparison_features),
                "n_remaining_fingerprint_pairs": len(comparison_pair_ids),
                "n_rule_positive": len(positive_pairs),
                "n_counterexample_candidates": len(counterexample_pairs),
                "n_counterexample_pairs": pair_count,
                "mean_counterexample_fingerprint_similarity": (
                    float(np.mean(selected_similarities)) if selected_similarities else ""
                ),
                "mean_counterexample_label_mismatch_count": (
                    float(np.mean(selected_mismatch_counts)) if selected_mismatch_counts else ""
                ),
                "status": status,
            }
        )
        matching_task.advance(1, note=rule_id)
    matching_task.complete("completed")

    feature_summary_rows: list[dict[str, object]] = []
    for (rule_id, feature), values in feature_summary_accumulators.items():
        feature_summary_rows.append(
            {
                "rule_id": rule_id,
                "feature": feature,
                "feature_role": feature_roles_by_rule[(rule_id, feature)],
                "feature_nodes": ";".join(feature_nodes_by_rule[(rule_id, feature)]),
                "n_counterexample_pairs": len(values),
                "mean_positive_value": float(np.mean([value["positive_value"] for value in values])),
                "mean_counterexample_value": float(np.mean([value["counterexample_value"] for value in values])),
                "mean_raw_difference": float(np.mean([value["raw_difference"] for value in values])),
                "mean_absolute_robust_standardized_difference": float(
                    np.mean([value["absolute_robust_standardized_difference"] for value in values])
                ),
            }
        )
    feature_summary_rows.sort(
        key=lambda row: (
            str(row["rule_id"]),
            -float(row["mean_absolute_robust_standardized_difference"]),
            str(row["feature"]),
        )
    )
    contrast_feature_summary_rows.sort(
        key=lambda row: (
            str(row["rule_id"]),
            str(row["feature"]),
            CONTRAST_GROUPS.index(str(row["contrast_group"])),
        )
    )
    contrast_feature_difference_rows.sort(
        key=lambda row: (
            str(row["rule_id"]),
            -float(
                row[
                    "absolute_rule_positive_minus_contrast_mean_robust_standardized_value"
                ]
            ),
            str(row["feature"]),
            str(row["contrast_group"]),
        )
    )

    progress.stage("Step 3: explaining semantic meta-patterns")
    (
        meta_pattern_contrast_group_rows,
        meta_pattern_comparison_evidence_rows,
        meta_pattern_feature_summary_rows,
        meta_pattern_feature_difference_rows,
        meta_pattern_explanation_rows,
    ) = build_meta_pattern_explanations(
        meta_pattern_rows,
        table.columns,
        index_by_trajectory_id,
        fingerprint_rows,
        trajectory_id,
        retained_fingerprint_pair_ids,
        node_features,
    )

    progress.stage("Step 3: writing all-counterexample and contrast-group outputs")
    write_csv(
        output_dir / "rule_counterexample_summary.csv",
        [
            "rule_id", "target_item", "antecedent_items", "antecedent_nodes",
            "target_nodes", "antecedent_feature_count", "comparison_feature_count",
            "n_remaining_fingerprint_pairs", "n_rule_positive", "n_counterexample_candidates",
            "n_counterexample_pairs", "mean_counterexample_fingerprint_similarity",
            "mean_counterexample_label_mismatch_count", "status",
        ],
        summary_rows,
    )
    write_csv(
        output_dir / "all_real_counterexamples.csv",
        [
            "rule_id", "target_item", "positive_trajectory_id", "counterexample_trajectory_id",
            "counterexample_rank", "fingerprint_matching_pair_count", "fingerprint_comparison_pair_count",
            "fingerprint_similarity", "fingerprint_label_mismatch_count",
            "antecedent_feature_distance", "antecedent_nodes", "target_nodes",
            "antecedent_feature_count", "comparison_feature_count",
        ],
        neighbor_rows,
    )
    write_csv(
        output_dir / "all_counterexample_feature_comparisons.csv",
        [
            "rule_id", "positive_trajectory_id", "counterexample_trajectory_id", "counterexample_rank",
            "feature", "feature_role", "feature_nodes", "positive_value", "counterexample_value", "raw_difference",
            "robust_standardized_difference", "absolute_robust_standardized_difference",
        ],
        comparison_rows,
    )
    write_csv(
        output_dir / "counterexample_feature_summary.csv",
        [
            "rule_id", "feature", "feature_role", "feature_nodes", "n_counterexample_pairs", "mean_positive_value",
            "mean_counterexample_value", "mean_raw_difference",
            "mean_absolute_robust_standardized_difference",
        ],
        feature_summary_rows,
    )
    write_csv(
        output_dir / "rule_contrast_groups.csv",
        [
            "rule_id", "trajectory_id", "contrast_group", "antecedent_satisfied",
            "target_satisfied", "target_item", "target_pair", "target_label",
            "antecedent_items",
        ],
        contrast_group_rows,
    )
    write_csv(
        output_dir / "rule_contrast_group_summary.csv",
        [
            "rule_id", "target_item", "antecedent_items", "contrast_group",
            "n_trajectories", "trajectory_ratio", "trajectory_ids",
        ],
        contrast_group_summary_rows,
    )
    write_csv(
        output_dir / "rule_contrast_feature_summary.csv",
        [
            "rule_id", "target_item", "antecedent_items", "contrast_group",
            "feature", "feature_role", "feature_nodes", "n_trajectories",
            "mean_value", "median_value", "std_value", "q25_value", "q75_value",
            "mean_robust_standardized_value", "median_robust_standardized_value",
        ],
        contrast_feature_summary_rows,
    )
    write_csv(
        output_dir / "rule_contrast_feature_differences.csv",
        [
            "rule_id", "target_item", "antecedent_items", "contrast_group",
            "feature", "feature_role", "feature_nodes", "n_rule_positive",
            "n_contrast_group", "rule_positive_mean_value",
            "contrast_group_mean_value",
            "rule_positive_minus_contrast_mean_value",
            "absolute_rule_positive_minus_contrast_mean_value",
            "rule_positive_mean_robust_standardized_value",
            "contrast_group_mean_robust_standardized_value",
            "rule_positive_minus_contrast_mean_robust_standardized_value",
            "absolute_rule_positive_minus_contrast_mean_robust_standardized_value",
        ],
        contrast_feature_difference_rows,
    )
    if not config.include_rule_level_outputs:
        for filename in STEP3_RULE_LEVEL_OUTPUT_FILES:
            path = output_dir / filename
            if path.exists():
                path.unlink()
    write_csv(
        output_dir / "meta_pattern_contrast_groups.csv",
        [
            "meta_pattern_id",
            "trajectory_id",
            "contrast_group",
            "contrast_group_label",
            "is_covered",
            "is_contrast_member",
            "selection_reason",
            "required_uncommon_counts",
            "observed_uncommon_counts",
            "pattern_deficit",
            "pattern_evidence_count",
            "n_pair_labels",
            "n_uncommon_total",
            "n_hybrid_total",
            "n_normal_total",
        ],
        meta_pattern_contrast_group_rows,
    )
    write_csv(
        output_dir / "meta_pattern_comparison_evidence.csv",
        [
            "meta_pattern_id",
            "contrast_group",
            "contrast_group_label",
            "selection_reason",
            "node_id",
            "pair_id",
            "required_uncommon_count",
            "n_covered",
            "n_contrast",
            "covered_uncommon_count",
            "contrast_uncommon_count",
            "covered_uncommon_ratio",
            "contrast_uncommon_ratio",
            "covered_to_contrast_ratio",
            "covered_to_contrast_ratio_status",
        ],
        meta_pattern_comparison_evidence_rows,
    )
    write_csv(
        output_dir / "meta_pattern_feature_summary.csv",
        [
            "meta_pattern_id",
            "contrast_group",
            "contrast_group_label",
            "selection_reason",
            "node_id",
            "feature",
            "n_trajectories",
            "mean_value",
            "median_value",
            "std_value",
            "q25_value",
            "q75_value",
            "mean_robust_standardized_value",
            "median_robust_standardized_value",
        ],
        meta_pattern_feature_summary_rows,
    )
    write_csv(
        output_dir / "meta_pattern_feature_differences.csv",
        [
            "meta_pattern_id",
            "contrast_group",
            "contrast_group_label",
            "selection_reason",
            "node_id",
            "feature",
            "n_covered",
            "n_contrast",
            "covered_mean_value",
            "contrast_mean_value",
            "covered_median_value",
            "contrast_median_value",
            "covered_minus_contrast_mean_value",
            "covered_minus_contrast_median_value",
            "covered_mean_robust_standardized_value",
            "contrast_mean_robust_standardized_value",
            "covered_median_robust_standardized_value",
            "contrast_median_robust_standardized_value",
            "covered_minus_contrast_mean_robust_standardized_value",
            "absolute_covered_minus_contrast_mean_robust_standardized_value",
            "covered_minus_contrast_median_robust_standardized_value",
            "absolute_covered_minus_contrast_median_robust_standardized_value",
            "median_difference_direction",
        ],
        meta_pattern_feature_difference_rows,
    )
    write_csv(
        output_dir / "meta_pattern_explanation_summary.csv",
        [
            "meta_pattern_id",
            "readable_pattern",
            "required_uncommon_counts",
            "nodes",
            "n_covered",
            "n_near_miss",
            "n_matched_non_pattern",
            "n_typical_normal",
            "coverage_ratio",
            "n_source_rules",
            "default_contrast_group",
            "contrast_selection_reasons",
            "top_comparison_evidence",
            "top_feature_evidence",
            "top_comparison_evidence_near_miss",
            "top_feature_evidence_near_miss",
            "top_comparison_evidence_matched_non_pattern",
            "top_feature_evidence_matched_non_pattern",
            "top_comparison_evidence_typical_normal",
            "top_feature_evidence_typical_normal",
            "covered_trajectory_ids",
            "near_miss_trajectory_ids",
            "matched_non_pattern_trajectory_ids",
            "typical_normal_trajectory_ids",
        ],
        meta_pattern_explanation_rows,
    )
    write_json(
        output_dir / "step3_summary.json",
        {
            "step": "step3",
            "status": "completed",
            "inputs": {
                "data": str(data_path),
                "step1_dir": str(step1_dir),
                "step2_dir": str(step2_dir),
                "trajectory_id": trajectory_id,
            },
            "parameters": {
                "include_all_counterexamples": config.include_all_counterexamples,
                "include_rule_level_outputs": config.include_rule_level_outputs,
                "matching_strategy": (
                    "all_real_counterexamples_sorted_by_minimum_remaining_fingerprint_"
                    "label_mismatches_then_antecedent_feature_distance"
                ),
                "fingerprint_pair_scope": "step2_retained_pairs",
                "feature_distance": "robust_standardized_euclidean",
                "feature_comparison_groups": "antecedent_and_target",
                "rule_contrast_groups": CONTRAST_GROUPS,
                "meta_pattern_definition": (
                    "node-level uncommonness counts derived from Step 2 source rules "
                    "and evaluated on Step 2 retained fingerprint pairs"
                ),
                "meta_pattern_contrast_groups": META_CONTRAST_GROUPS,
                "rule_contrast_definition": {
                    "rule_positive": "antecedent_satisfied_and_target_satisfied",
                    "antecedent_only": "antecedent_satisfied_and_target_not_satisfied",
                    "target_only": "antecedent_not_satisfied_and_target_satisfied",
                    "neither": "antecedent_not_satisfied_and_target_not_satisfied",
                },
            },
            "results": {
                "available_rule_count": len(rules),
                "rule_level_output_rule_count": len(rule_level_rules),
                "rules_with_counterexamples": sum(row["status"] == "matched" for row in summary_rows),
                "counterexample_pair_count": len(neighbor_rows),
                "feature_comparison_count": len(comparison_rows),
                "contrast_group_row_count": len(contrast_group_rows),
                "contrast_group_summary_row_count": len(contrast_group_summary_rows),
                "contrast_feature_summary_row_count": len(contrast_feature_summary_rows),
                "contrast_feature_difference_row_count": len(contrast_feature_difference_rows),
                "meta_pattern_count": len(meta_pattern_rows),
                "meta_pattern_contrast_group_row_count": len(meta_pattern_contrast_group_rows),
                "meta_pattern_comparison_evidence_row_count": len(meta_pattern_comparison_evidence_rows),
                "meta_pattern_feature_summary_row_count": len(meta_pattern_feature_summary_rows),
                "meta_pattern_feature_difference_row_count": len(meta_pattern_feature_difference_rows),
                "meta_pattern_explanation_row_count": len(meta_pattern_explanation_rows),
            },
            "generated_files": {
                filename: str(output_dir / filename)
                for filename in step3_output_files(config)
            },
        },
    )
    LOGGER.info(
        "Prepared Step 3 explanations for %d semantic meta-patterns and %d rule-level outputs.",
        len(meta_pattern_rows),
        len(neighbor_rows),
    )
