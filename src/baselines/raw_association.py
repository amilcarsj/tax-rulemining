"""Association-rule baseline mined directly from discretized raw features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, count
from math import ceil
from pathlib import Path
from typing import Any

from baselines.common import finite_ratio, itemset_has_unique_features
from core.models import DatasetSchema, DatasetTable
from data_io.writers import write_csv, write_json
from scoring.lof import require_scoring_dependencies


RAW_ASSOCIATION_FILES = [
    "raw_feature_discretization_ranges.csv",
    "raw_feature_bins.csv",
    "raw_feature_transactions.csv",
    "raw_frequent_itemsets.csv",
    "raw_association_rules.csv",
    "raw_association_summary.json",
]


@dataclass(frozen=True)
class RawAssociationConfig:
    """Configuration for direct raw-feature association-rule mining."""

    min_support_count: int = 20
    min_support_ratio: float = 0.10
    min_confidence: float = 0.60
    min_lift: float = 1.20
    max_itemset_length: int = 2
    max_rules: int = 50000
    min_antecedent_length: int = 1


@dataclass
class DiscretizedRawFeatures:
    """Discretized raw feature table plus transaction representation."""

    used_features: list[str]
    range_rows: list[dict[str, str | int | float]]
    bin_rows: list[dict[str, str]]
    transaction_rows: list[dict[str, str]]
    transactions: list[tuple[str, ...]]


@dataclass
class FPNode:
    """Node in the FP-tree used by the raw association baseline."""

    item: str | None
    count: int
    parent: FPNode | None
    children: dict[str, FPNode] = field(default_factory=dict)
    link: FPNode | None = None


@dataclass
class HeaderEntry:
    """Header-table entry linking all FP-tree nodes for one item."""

    support_count: int
    first_node: FPNode | None = None


@dataclass(frozen=True)
class RuleMiningStats:
    """Summary counts from bounded association-rule generation."""

    candidate_rule_count: int
    passing_rule_count: int
    written_rule_count: int
    rule_output_limit_reached: bool


def _clean_number(value: str, np: Any) -> float:
    stripped = value.strip()
    if stripped == "":
        return float(np.nan)
    return float(stripped)


def _format_bound(value: float) -> str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return str(float(value))


def _bin_label(value: float, lower_threshold: float, upper_threshold: float, np: Any) -> str:
    if np.isnan(value):
        return ""
    if value <= lower_threshold:
        return "low"
    if value <= upper_threshold:
        return "medium"
    return "high"


def discretize_raw_features(
    table: DatasetTable,
    features: list[str],
) -> DiscretizedRawFeatures:
    """Convert each usable numeric feature into low/medium/high tertile bins."""
    np, _, _, _ = require_scoring_dependencies()

    bin_rows = [{"trajectory_id": trajectory_id} for trajectory_id in table.trajectory_ids]
    transactions: list[list[str]] = [[] for _ in table.trajectory_ids]
    range_rows: list[dict[str, str | int | float]] = []
    used_features: list[str] = []

    for feature in features:
        values = np.asarray(
            [_clean_number(value, np) for value in table.columns[feature]],
            dtype=float,
        )
        valid_values = values[~np.isnan(values)]
        n_missing = int(np.isnan(values).sum())

        if valid_values.size == 0:
            range_rows.append(
                {
                    "feature": feature,
                    "bin_label": "",
                    "lower_bound": "",
                    "upper_bound": "",
                    "lower_inclusive": "",
                    "upper_inclusive": "",
                    "n_non_missing": 0,
                    "n_missing": n_missing,
                    "status": "dropped_all_missing",
                }
            )
            continue

        unique_values = np.unique(valid_values)
        if unique_values.size < 2:
            range_rows.append(
                {
                    "feature": feature,
                    "bin_label": "",
                    "lower_bound": str(float(unique_values[0])),
                    "upper_bound": str(float(unique_values[0])),
                    "lower_inclusive": "true",
                    "upper_inclusive": "true",
                    "n_non_missing": int(valid_values.size),
                    "n_missing": n_missing,
                    "status": "dropped_constant",
                }
            )
            continue

        lower_threshold, upper_threshold = np.nanpercentile(
            valid_values,
            [100.0 / 3.0, 200.0 / 3.0],
        )
        if not float(lower_threshold) < float(upper_threshold):
            range_rows.append(
                {
                    "feature": feature,
                    "bin_label": "",
                    "lower_bound": str(float(lower_threshold)),
                    "upper_bound": str(float(upper_threshold)),
                    "lower_inclusive": "",
                    "upper_inclusive": "",
                    "n_non_missing": int(valid_values.size),
                    "n_missing": n_missing,
                    "status": "dropped_insufficient_distinct_quantiles",
                }
            )
            continue

        used_features.append(feature)
        range_specs = [
            ("low", float("-inf"), float(lower_threshold), "true", "true"),
            ("medium", float(lower_threshold), float(upper_threshold), "false", "true"),
            ("high", float(upper_threshold), float("inf"), "false", "true"),
        ]
        for label, lower, upper, lower_inclusive, upper_inclusive in range_specs:
            range_rows.append(
                {
                    "feature": feature,
                    "bin_label": label,
                    "lower_bound": _format_bound(lower),
                    "upper_bound": _format_bound(upper),
                    "lower_inclusive": lower_inclusive,
                    "upper_inclusive": upper_inclusive,
                    "n_non_missing": int(valid_values.size),
                    "n_missing": n_missing,
                    "status": "used",
                }
            )

        for row_index, value in enumerate(values):
            label = _bin_label(value, float(lower_threshold), float(upper_threshold), np)
            bin_rows[row_index][feature] = label
            if label:
                transactions[row_index].append(f"{feature}={label}")

    transaction_tuples = [tuple(items) for items in transactions]
    transaction_rows = [
        {"trajectory_id": trajectory_id, "items": ";".join(items)}
        for trajectory_id, items in zip(table.trajectory_ids, transaction_tuples, strict=False)
    ]
    return DiscretizedRawFeatures(
        used_features=used_features,
        range_rows=range_rows,
        bin_rows=bin_rows,
        transaction_rows=transaction_rows,
        transactions=transaction_tuples,
    )


def _support_threshold(config: RawAssociationConfig, n_transactions: int) -> int:
    ratio_count = ceil(config.min_support_ratio * n_transactions)
    return max(config.min_support_count, ratio_count)


def _append_header_node(header_entry: HeaderEntry, node: FPNode) -> None:
    if header_entry.first_node is None:
        header_entry.first_node = node
        return
    current = header_entry.first_node
    while current.link is not None:
        current = current.link
    current.link = node


def _insert_tree_path(
    root: FPNode,
    items: list[str],
    count_value: int,
    header_table: dict[str, HeaderEntry],
) -> None:
    current = root
    for item in items:
        child = current.children.get(item)
        if child is None:
            child = FPNode(item=item, count=count_value, parent=current)
            current.children[item] = child
            _append_header_node(header_table[item], child)
        else:
            child.count += count_value
        current = child


def _build_fp_tree(
    weighted_transactions: list[tuple[tuple[str, ...], int]],
    min_support_count: int,
) -> dict[str, HeaderEntry]:
    support_counts: Counter[str] = Counter()
    for transaction, transaction_count in weighted_transactions:
        for item in transaction:
            support_counts[item] += transaction_count

    frequent_supports = {
        item: support_count
        for item, support_count in support_counts.items()
        if support_count >= min_support_count
    }
    if not frequent_supports:
        return {}

    item_order = {
        item: index
        for index, item in enumerate(
            sorted(frequent_supports, key=lambda candidate: (-frequent_supports[candidate], candidate))
        )
    }
    header_table = {
        item: HeaderEntry(support_count=support_count)
        for item, support_count in frequent_supports.items()
    }
    root = FPNode(item=None, count=0, parent=None)

    for transaction, transaction_count in weighted_transactions:
        ordered_items = sorted(
            (item for item in transaction if item in frequent_supports),
            key=lambda item: item_order[item],
        )
        if ordered_items:
            _insert_tree_path(root, ordered_items, transaction_count, header_table)

    return header_table


def _conditional_pattern_base(header_entry: HeaderEntry) -> list[tuple[tuple[str, ...], int]]:
    patterns: list[tuple[tuple[str, ...], int]] = []
    node = header_entry.first_node
    while node is not None:
        path: list[str] = []
        parent = node.parent
        while parent is not None and parent.item is not None:
            path.append(parent.item)
            parent = parent.parent
        if path:
            patterns.append((tuple(path), node.count))
        node = node.link
    return patterns


def _mine_fp_tree(
    header_table: dict[str, HeaderEntry],
    min_support_count: int,
    max_itemset_length: int,
    prefix: tuple[str, ...],
    itemset_counts: dict[tuple[str, ...], int],
) -> None:
    mining_order = sorted(
        header_table,
        key=lambda item: (header_table[item].support_count, item),
    )
    for item in mining_order:
        new_prefix = (*prefix, item)
        itemset = tuple(sorted(new_prefix))
        if itemset_has_unique_features(itemset):
            itemset_counts[itemset] = header_table[item].support_count
        if len(new_prefix) >= max_itemset_length:
            continue

        conditional_patterns = _conditional_pattern_base(header_table[item])
        conditional_header = _build_fp_tree(conditional_patterns, min_support_count)
        if conditional_header:
            _mine_fp_tree(
                conditional_header,
                min_support_count,
                max_itemset_length,
                new_prefix,
                itemset_counts,
            )


def mine_frequent_itemsets(
    transactions: list[tuple[str, ...]],
    config: RawAssociationConfig,
) -> tuple[dict[tuple[str, ...], int], int]:
    """Mine frequent raw-feature itemsets with FP-growth."""
    n_transactions = len(transactions)
    min_support_count = _support_threshold(config, n_transactions)
    weighted_transactions = [(transaction, 1) for transaction in transactions]
    header_table = _build_fp_tree(weighted_transactions, min_support_count)
    itemset_counts: dict[tuple[str, ...], int] = {}
    if header_table:
        _mine_fp_tree(
            header_table=header_table,
            min_support_count=min_support_count,
            max_itemset_length=config.max_itemset_length,
            prefix=(),
            itemset_counts=itemset_counts,
        )
    return itemset_counts, min_support_count


def _rule_priority(row: dict[str, str | int | float]) -> tuple[float, float, int, int]:
    return (
        float(row["lift"]),
        float(row["confidence"]),
        int(row["support_count"]),
        int(row["antecedent_length"]),
    )


def build_association_rules(
    itemset_counts: dict[tuple[str, ...], int],
    n_transactions: int,
    config: RawAssociationConfig,
) -> tuple[list[dict[str, str | int | float]], RuleMiningStats]:
    """Generate a bounded top-rule table from FP-growth frequent itemsets."""
    import heapq

    if config.max_rules < 1:
        raise ValueError("Raw association baseline requires max_rules >= 1.")
    heap: list[tuple[tuple[float, float, int, int], int, dict[str, str | int | float]]] = []
    sequence = count()
    candidate_rule_count = 0
    passing_rule_count = 0

    for itemset, support_count in itemset_counts.items():
        if len(itemset) < 2:
            continue
        itemset_items = set(itemset)
        for antecedent_length in range(config.min_antecedent_length, len(itemset)):
            for antecedent in combinations(itemset, antecedent_length):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(itemset_items.difference(antecedent)))
                antecedent_count = itemset_counts.get(antecedent, 0)
                consequent_count = itemset_counts.get(consequent, 0)
                if antecedent_count == 0 or consequent_count == 0:
                    continue

                candidate_rule_count += 1
                support_ratio = finite_ratio(support_count, n_transactions)
                antecedent_support_ratio = finite_ratio(antecedent_count, n_transactions)
                consequent_support_ratio = finite_ratio(consequent_count, n_transactions)
                confidence = finite_ratio(support_count, antecedent_count)
                lift = finite_ratio(confidence, consequent_support_ratio)
                leverage = support_ratio - antecedent_support_ratio * consequent_support_ratio

                if confidence < config.min_confidence or lift < config.min_lift:
                    continue
                passing_rule_count += 1

                rule = {
                    "rule_id": "",
                    "antecedent_items": ";".join(antecedent),
                    "consequent_items": ";".join(consequent),
                    "antecedent_length": len(antecedent),
                    "consequent_length": len(consequent),
                    "itemset_length": len(itemset),
                    "support_count": support_count,
                    "support_ratio": support_ratio,
                    "antecedent_support_count": antecedent_count,
                    "antecedent_support_ratio": antecedent_support_ratio,
                    "consequent_support_count": consequent_count,
                    "consequent_support_ratio": consequent_support_ratio,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                }
                entry = (_rule_priority(rule), next(sequence), rule)
                if len(heap) < config.max_rules:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    rules = [entry[2] for entry in heap]
    rules.sort(
        key=lambda row: (
            -float(row["lift"]),
            -float(row["confidence"]),
            -int(row["support_count"]),
            -int(row["antecedent_length"]),
            str(row["antecedent_items"]),
        )
    )
    for index, row in enumerate(rules, start=1):
        row["rule_id"] = f"RAW-FP-R{index:05d}"

    stats = RuleMiningStats(
        candidate_rule_count=candidate_rule_count,
        passing_rule_count=passing_rule_count,
        written_rule_count=len(rules),
        rule_output_limit_reached=passing_rule_count > len(rules),
    )
    return rules, stats


def frequent_itemset_rows(
    itemset_counts: dict[tuple[str, ...], int],
    n_transactions: int,
) -> tuple[list[dict[str, str | int | float]], dict[int, int]]:
    itemset_length_counts: dict[int, int] = Counter(len(itemset) for itemset in itemset_counts)
    rows = [
        {
            "itemset_id": "",
            "items": ";".join(itemset),
            "itemset_length": len(itemset),
            "support_count": count_value,
            "support_ratio": finite_ratio(count_value, n_transactions),
        }
        for itemset, count_value in itemset_counts.items()
    ]
    rows.sort(
        key=lambda row: (
            int(row["itemset_length"]),
            -int(row["support_count"]),
            str(row["items"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["itemset_id"] = f"RAW-FP-I{index:05d}"
    return rows, dict(itemset_length_counts)


def run_raw_association_baseline(
    table: DatasetTable,
    schema: DatasetSchema,
    data_path: Path,
    output_dir: Path,
    trajectory_id_column: str,
    features: list[str],
    exclude_columns: list[str],
    config: RawAssociationConfig,
) -> dict[str, Any]:
    """Run FP-growth association-rule mining over discretized raw features."""
    if config.max_itemset_length < 2:
        raise ValueError("Raw association baseline requires max_itemset_length >= 2.")
    if config.min_antecedent_length < 1:
        raise ValueError("Raw association baseline requires min_antecedent_length >= 1.")
    if config.min_antecedent_length >= config.max_itemset_length:
        raise ValueError("min_antecedent_length must be smaller than max_itemset_length.")
    if len(features) == 0:
        raise ValueError("Raw association baseline requires at least one numeric feature column.")

    discretized = discretize_raw_features(table, features)
    if len(discretized.used_features) < 2:
        raise ValueError("Raw association baseline requires at least two discretized features.")

    itemset_counts, min_support_count_used = mine_frequent_itemsets(
        discretized.transactions,
        config,
    )
    itemset_rows, itemset_length_counts = frequent_itemset_rows(itemset_counts, table.row_count)
    rule_rows, rule_stats = build_association_rules(itemset_counts, table.row_count, config)

    write_csv(
        output_dir / "raw_feature_discretization_ranges.csv",
        [
            "feature",
            "bin_label",
            "lower_bound",
            "upper_bound",
            "lower_inclusive",
            "upper_inclusive",
            "n_non_missing",
            "n_missing",
            "status",
        ],
        discretized.range_rows,
    )
    write_csv(
        output_dir / "raw_feature_bins.csv",
        ["trajectory_id", *discretized.used_features],
        discretized.bin_rows,
    )
    write_csv(
        output_dir / "raw_feature_transactions.csv",
        ["trajectory_id", "items"],
        discretized.transaction_rows,
    )
    write_csv(
        output_dir / "raw_frequent_itemsets.csv",
        ["itemset_id", "items", "itemset_length", "support_count", "support_ratio"],
        itemset_rows,
    )
    write_csv(
        output_dir / "raw_association_rules.csv",
        [
            "rule_id",
            "antecedent_items",
            "consequent_items",
            "antecedent_length",
            "consequent_length",
            "itemset_length",
            "support_count",
            "support_ratio",
            "antecedent_support_count",
            "antecedent_support_ratio",
            "consequent_support_count",
            "consequent_support_ratio",
            "confidence",
            "lift",
            "leverage",
        ],
        rule_rows,
    )

    summary = {
        "baseline": "raw_association_rules",
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
        "parameters": {
            "discretization": "tertile bins over non-missing numeric values",
            "frequent_itemset_algorithm": "fp_growth",
            "min_support_count": config.min_support_count,
            "min_support_ratio": config.min_support_ratio,
            "min_support_count_used": min_support_count_used,
            "min_confidence": config.min_confidence,
            "min_lift": config.min_lift,
            "max_itemset_length": config.max_itemset_length,
            "max_rules": config.max_rules,
            "min_antecedent_length": config.min_antecedent_length,
            "same_feature_item_conflicts": "disallowed by construction",
        },
        "results": {
            "discretized_feature_count": len(discretized.used_features),
            "transaction_count": len(discretized.transactions),
            "frequent_itemset_count": len(itemset_rows),
            "itemset_length_counts": itemset_length_counts,
            "candidate_rule_count": rule_stats.candidate_rule_count,
            "passing_rule_count": rule_stats.passing_rule_count,
            "association_rule_count": rule_stats.written_rule_count,
            "rule_output_limit_reached": rule_stats.rule_output_limit_reached,
        },
        "generated_files": {filename: str(output_dir / filename) for filename in RAW_ASSOCIATION_FILES},
    }
    write_json(output_dir / "raw_association_summary.json", summary)
    return summary
