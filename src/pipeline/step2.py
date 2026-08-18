"""Step 2 high-level pseudo-label analysis."""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from core.progress import ProgressReporter, ProgressTask
from data_io.writers import write_csv, write_json
from pipeline import meta_patterns as meta

LOGGER = logging.getLogger("tax_rulemining")

STEP2_OUTPUT_FILES = [
    "pair_label_summary.csv",
    "trajectory_fingerprint_summary.csv",
    "high_level_item_cooccurrence.csv",
    "high_level_association_rules.csv",
    "high_level_rule_coverage.csv",
    "behavioral_pattern_catalogue.csv",
    "semantic_meta_patterns.csv",
    "semantic_meta_pattern_coverage.csv",
    "semantic_meta_pattern_source_rules.csv",
    "semantic_behavioral_pattern_catalogue.csv",
    "step2_summary.json",
]


@dataclass
class Step2Config:
    include_normal_targets: bool = False
    exclude_hybrid_antecedents: bool = True
    exclude_target_label_echoes: bool = True
    max_high_level_rule_length: int = 2
    min_support_count: int = 20
    min_support_ratio: float = 0.0
    min_confidence: float = 0.8
    min_lift: float = 2.0
    min_target_coverage: float = 0.0
    min_target_prevalence_ratio: float = 0.02
    min_node_depth: int | None = None
    max_node_depth: int | None = None


@dataclass
class FingerprintDataset:
    trajectory_id_column: str
    trajectory_ids: list[str]
    pair_ids: list[str]
    rows: list[dict[str, str]]


@dataclass
class DepthFilterSummary:
    original_pair_ids: list[str]
    retained_pair_ids: list[str]
    skipped_pair_ids: list[str]
    skipped_depth_range_pair_ids: list[str]


def canonicalize_label(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped == "Normal" or stripped.startswith("Normal "):
        return "Normal"
    if stripped == "Hybrid" or stripped.startswith("Hybrid "):
        return "Hybrid"
    if stripped.startswith("Uncommon "):
        return stripped
    return stripped


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


def sanitize_label_for_column(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").replace("×", "x")


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    left_node, right_node = pair_id.split("__x__", 1)
    return left_node, right_node


def load_node_depths(node_feature_sets_path: Path) -> dict[str, int]:
    if not node_feature_sets_path.exists():
        raise FileNotFoundError(
            "Step 2 now requires the Step 1 node feature metadata so it can restrict "
            f"mining to same-depth taxonomy pairs. Missing file: {node_feature_sets_path}"
        )

    node_depths: dict[str, int] = {}
    with node_feature_sets_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = row.get("node_id", "")
            depth_text = row.get("depth", "")
            if not node_id or depth_text is None or str(depth_text).strip() == "":
                continue
            node_depths[node_id] = int(depth_text)
    return node_depths


def load_fingerprint_dataset(path: Path, trajectory_id_column: str = "trajectory_id") -> FingerprintDataset:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"Fingerprint CSV is empty: {path}")
        if trajectory_id_column not in headers:
            raise ValueError(
                "Fingerprint CSV is missing the required trajectory identifier column "
                f"'{trajectory_id_column}'."
            )
        pair_ids = [header for header in headers if header != trajectory_id_column]
        trajectory_ids: list[str] = []
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            trajectory_id = raw_row.get(trajectory_id_column, "")
            if trajectory_id is None or str(trajectory_id).strip() == "":
                raise ValueError("Fingerprint CSV contains an empty trajectory_id value.")
            trajectory_id_str = str(trajectory_id)
            trajectory_ids.append(trajectory_id_str)
            canonical_row = {pair_id: canonicalize_label(str(raw_row.get(pair_id, "") or "")) for pair_id in pair_ids}
            rows.append(canonical_row)
    return FingerprintDataset(
        trajectory_id_column=trajectory_id_column,
        trajectory_ids=trajectory_ids,
        pair_ids=pair_ids,
        rows=rows,
    )


def filter_dataset_to_same_depth_pairs(
    dataset: FingerprintDataset,
    node_depths: dict[str, int],
    config: Step2Config,
) -> tuple[FingerprintDataset, DepthFilterSummary]:
    retained_pair_ids: list[str] = []
    skipped_pair_ids: list[str] = []
    skipped_depth_range_pair_ids: list[str] = []

    for pair_id in dataset.pair_ids:
        left_node, right_node = parse_pair_id(pair_id)
        left_depth = node_depths.get(left_node)
        right_depth = node_depths.get(right_node)
        if left_depth is None or right_depth is None:
            raise ValueError(
                f"Step 2 could not resolve taxonomy depths for pair '{pair_id}'."
            )
        # We keep only pair columns whose two taxonomy nodes live at the same depth,
        # which prevents Step 2 from mixing leaf-level and internal-level comparisons.
        if left_depth != right_depth:
            skipped_pair_ids.append(pair_id)
            continue
        if config.min_node_depth is not None and left_depth < config.min_node_depth:
            skipped_depth_range_pair_ids.append(pair_id)
            continue
        if config.max_node_depth is not None and left_depth > config.max_node_depth:
            skipped_depth_range_pair_ids.append(pair_id)
            continue
        retained_pair_ids.append(pair_id)

    filtered_rows = [
        {pair_id: row.get(pair_id, "") for pair_id in retained_pair_ids}
        for row in dataset.rows
    ]
    filtered_dataset = FingerprintDataset(
        trajectory_id_column=dataset.trajectory_id_column,
        trajectory_ids=dataset.trajectory_ids,
        pair_ids=retained_pair_ids,
        rows=filtered_rows,
    )
    return filtered_dataset, DepthFilterSummary(
        original_pair_ids=list(dataset.pair_ids),
        retained_pair_ids=retained_pair_ids,
        skipped_pair_ids=skipped_pair_ids,
        skipped_depth_range_pair_ids=skipped_depth_range_pair_ids,
    )


def build_transactions(
    dataset: FingerprintDataset,
    *,
    exclude_hybrid_items: bool = False,
) -> list[dict[str, str]]:
    """Build labelled transactions, optionally omitting Hybrid items for rule mining."""
    transactions: list[dict[str, str]] = []
    for row in dataset.rows:
        transactions.append(
            {
                pair_id: f"{pair_id}={label}"
                for pair_id, label in row.items()
                if label and not (exclude_hybrid_items and label_type(label) == "hybrid")
            }
        )
    return transactions


def parse_item(item: str) -> tuple[str, str]:
    pair_id, label = item.split("=", 1)
    return pair_id, label


def is_target_label(label: str, include_normal_targets: bool) -> bool:
    """Return whether a label is an eligible rule consequent.

    Step 2 is intentionally focused on uncommon behaviour. Hybrid labels are
    never consequents; they may be restored as antecedent context only through
    the explicit configuration switch.
    """
    label_kind = label_type(label)
    return label_kind == "uncommon" or (
        include_normal_targets and label_kind == "normal"
    )


def filter_target_items_by_prevalence(
    observed_items: set[str],
    item_to_indices: dict[str, list[int]],
    n_total: int,
    config: Step2Config,
) -> list[str]:
    """Keep only target items whose dataset prevalence clears the configured ratio."""
    min_target_count = max(1, math.ceil(config.min_target_prevalence_ratio * n_total))
    return sorted(
        item
        for item in observed_items
        if is_target_label(parse_item(item)[1], config.include_normal_targets)
        and len(item_to_indices[item]) >= min_target_count
    )


def summarize_pair_labels(dataset: FingerprintDataset) -> tuple[list[dict[str, object]], list[str], set[str]]:
    observed_labels: set[str] = set()
    for row in dataset.rows:
        for label in row.values():
            if label:
                observed_labels.add(label)

    canonical_label_order = sorted(
        observed_labels,
        key=lambda label: (
            0 if label == "Normal" else 1 if label == "Hybrid" else 2,
            label,
        ),
    )
    extra_count_columns = [f"count__{sanitize_label_for_column(label)}" for label in canonical_label_order]
    summary_rows: list[dict[str, object]] = []

    for pair_id in dataset.pair_ids:
        label_counter = Counter(row[pair_id] for row in dataset.rows if row[pair_id])
        n_trajectories = len(dataset.rows)
        n_missing = sum(1 for row in dataset.rows if not row[pair_id])
        n_normal = label_counter.get("Normal", 0)
        n_hybrid = label_counter.get("Hybrid", 0)
        n_uncommon_total = sum(
            count for label, count in label_counter.items() if label_type(label) == "uncommon"
        )
        non_missing = n_trajectories - n_missing
        dominant_label = ""
        dominant_label_ratio = 0.0
        if label_counter:
            dominant_label, dominant_count = max(
                label_counter.items(), key=lambda item: (item[1], item[0])
            )
            dominant_label_ratio = dominant_count / non_missing if non_missing else 0.0

        row_summary: dict[str, object] = {
            "pair_id": pair_id,
            "n_trajectories": n_trajectories,
            "n_missing": n_missing,
            "n_normal": n_normal,
            "n_hybrid": n_hybrid,
            "n_uncommon_total": n_uncommon_total,
            "pct_normal": n_normal / non_missing if non_missing else 0.0,
            "pct_hybrid": n_hybrid / non_missing if non_missing else 0.0,
            "pct_uncommon_total": n_uncommon_total / non_missing if non_missing else 0.0,
            "entropy": entropy_from_counts(list(label_counter.values())),
            "non_normal_ratio": (non_missing - n_normal) / non_missing if non_missing else 0.0,
            "dominant_label": dominant_label,
            "dominant_label_ratio": dominant_label_ratio,
        }
        for label in canonical_label_order:
            row_summary[f"count__{sanitize_label_for_column(label)}"] = label_counter.get(label, 0)
        summary_rows.append(row_summary)

    fieldnames = [
        "pair_id",
        "n_trajectories",
        "n_missing",
        "n_normal",
        "n_hybrid",
        "n_uncommon_total",
        "pct_normal",
        "pct_hybrid",
        "pct_uncommon_total",
        "entropy",
        "non_normal_ratio",
        "dominant_label",
        "dominant_label_ratio",
        *extra_count_columns,
    ]
    uncommon_nodes = {uncommon_node_from_label(label) for label in observed_labels if label_type(label) == "uncommon"}
    uncommon_nodes.discard("")
    return summary_rows, fieldnames, uncommon_nodes


def summarize_trajectory_fingerprints(
    dataset: FingerprintDataset,
    uncommon_nodes: set[str],
) -> tuple[list[dict[str, object]], list[str]]:
    uncommon_node_list = sorted(uncommon_nodes)
    rows: list[dict[str, object]] = []

    for trajectory_id, row in zip(dataset.trajectory_ids, dataset.rows, strict=False):
        available_labels = [label for label in row.values() if label]
        n_pair_labels = len(available_labels)
        n_normal = sum(1 for label in available_labels if label_type(label) == "normal")
        n_hybrid = sum(1 for label in available_labels if label_type(label) == "hybrid")
        n_uncommon = sum(1 for label in available_labels if label_type(label) == "uncommon")
        non_normal_count = n_pair_labels - n_normal
        uncommon_counter = Counter(
            uncommon_node_from_label(label)
            for label in available_labels
            if label_type(label) == "uncommon"
        )
        uncommon_counter.pop("", None)
        dominant_uncommon_node = ""
        dominant_uncommon_node_count = 0
        dominant_uncommon_node_ratio = 0.0
        if uncommon_counter:
            dominant_uncommon_node, dominant_uncommon_node_count = max(
                uncommon_counter.items(), key=lambda item: (item[1], item[0])
            )
            dominant_uncommon_node_ratio = (
                dominant_uncommon_node_count / n_uncommon if n_uncommon else 0.0
            )

        trajectory_row: dict[str, object] = {
            "trajectory_id": trajectory_id,
            "n_pair_labels": n_pair_labels,
            "n_normal": n_normal,
            "n_hybrid": n_hybrid,
            "n_uncommon": n_uncommon,
            "non_normal_count": non_normal_count,
            "non_normal_ratio": non_normal_count / n_pair_labels if n_pair_labels else 0.0,
            "hybrid_ratio": n_hybrid / n_pair_labels if n_pair_labels else 0.0,
            "uncommon_ratio": n_uncommon / n_pair_labels if n_pair_labels else 0.0,
            "dominant_uncommon_node": dominant_uncommon_node,
            "dominant_uncommon_node_count": dominant_uncommon_node_count,
            "dominant_uncommon_node_ratio": dominant_uncommon_node_ratio,
            "fingerprint_entropy": entropy_from_counts(list(Counter(available_labels).values())),
        }
        for node in uncommon_node_list:
            trajectory_row[f"count_uncommon_{sanitize_label_for_column(node)}"] = uncommon_counter.get(node, 0)
        rows.append(trajectory_row)

    fieldnames = [
        "trajectory_id",
        "n_pair_labels",
        "n_normal",
        "n_hybrid",
        "n_uncommon",
        "non_normal_count",
        "non_normal_ratio",
        "hybrid_ratio",
        "uncommon_ratio",
        "dominant_uncommon_node",
        "dominant_uncommon_node_count",
        "dominant_uncommon_node_ratio",
        "fingerprint_entropy",
        *[f"count_uncommon_{sanitize_label_for_column(node)}" for node in uncommon_node_list],
    ]
    return rows, fieldnames


def summarize_item_cooccurrence(
    dataset: FingerprintDataset,
    transactions: list[dict[str, str]],
) -> list[dict[str, object]]:
    n_total = len(dataset.trajectory_ids)
    item_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for transaction in transactions:
        items = sorted(transaction.values())
        item_counts.update(items)
        for item_a, item_b in combinations(items, 2):
            pair_counts[(item_a, item_b)] += 1

    rows: list[dict[str, object]] = []
    for (item_a, item_b), count_ab in sorted(pair_counts.items()):
        count_a = item_counts[item_a]
        count_b = item_counts[item_b]
        support_a = count_a / n_total if n_total else 0.0
        support_b = count_b / n_total if n_total else 0.0
        support_ab = count_ab / n_total if n_total else 0.0
        confidence_a_to_b = count_ab / count_a if count_a else 0.0
        confidence_b_to_a = count_ab / count_b if count_b else 0.0
        lift = support_ab / (support_a * support_b) if support_a and support_b else 0.0
        union = count_a + count_b - count_ab
        jaccard = count_ab / union if union else 0.0
        rows.append(
            {
                "item_a": item_a,
                "item_b": item_b,
                "count_a": count_a,
                "count_b": count_b,
                "count_ab": count_ab,
                "support_a": support_a,
                "support_b": support_b,
                "support_ab": support_ab,
                "confidence_a_to_b": confidence_a_to_b,
                "confidence_b_to_a": confidence_b_to_a,
                "lift": lift,
                "jaccard": jaccard,
            }
        )
    return rows


@dataclass
class FPNode:
    """Node in the FP-tree used for high-level pseudo-label itemsets."""

    item: str | None
    count: int
    parent: "FPNode | None"
    children: dict[str, "FPNode"] = field(default_factory=dict)
    link: "FPNode | None" = None


@dataclass
class HeaderEntry:
    """Header-table entry linking FP-tree nodes for one pseudo-label item."""

    support_count: int
    first_node: FPNode | None = None


def min_rule_support_count(config: Step2Config, n_total: int) -> int:
    """Dataset-level support threshold shared by FP-growth and rule generation."""
    return max(config.min_support_count, math.ceil(config.min_support_ratio * n_total))


def _append_header_node(header_entry: HeaderEntry, node: FPNode) -> None:
    if header_entry.first_node is None:
        header_entry.first_node = node
        return
    current = header_entry.first_node
    while current.link is not None:
        current = current.link
    current.link = node


def _insert_fp_tree_path(
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
    min_support: int,
) -> dict[str, HeaderEntry]:
    support_counts: Counter[str] = Counter()
    for transaction, transaction_count in weighted_transactions:
        for item in transaction:
            support_counts[item] += transaction_count

    frequent_supports = {
        item: support_count
        for item, support_count in support_counts.items()
        if support_count >= min_support
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
            _insert_fp_tree_path(root, ordered_items, transaction_count, header_table)
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
    min_support: int,
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
        itemset_counts[itemset] = header_table[item].support_count
        if len(new_prefix) >= max_itemset_length:
            continue
        conditional_patterns = _conditional_pattern_base(header_table[item])
        conditional_header = _build_fp_tree(conditional_patterns, min_support)
        if conditional_header:
            _mine_fp_tree(
                conditional_header,
                min_support,
                max_itemset_length,
                new_prefix,
                itemset_counts,
            )


def mine_frequent_pseudo_label_itemsets(
    transactions: list[dict[str, str]],
    min_support: int,
    max_itemset_length: int,
) -> dict[tuple[str, ...], int]:
    """Mine high-level pseudo-label itemsets with FP-growth."""
    weighted_transactions = [
        (tuple(sorted(transaction.values())), 1)
        for transaction in transactions
        if transaction
    ]
    header_table = _build_fp_tree(weighted_transactions, min_support)
    itemset_counts: dict[tuple[str, ...], int] = {}
    if header_table:
        _mine_fp_tree(
            header_table,
            min_support,
            max_itemset_length,
            (),
            itemset_counts,
        )
    return itemset_counts


def valid_antecedent_for_target(
    antecedent_items: tuple[str, ...],
    target_pair: str,
    target_label: str,
    config: Step2Config,
) -> bool:
    """Apply taxonomy-pair constraints after FP-growth discovers an itemset."""
    seen_pairs: set[str] = set()
    for item in antecedent_items:
        item_pair, item_label = parse_item(item)
        if item_pair == target_pair:
            return False
        if item_pair in seen_pairs:
            return False
        if config.exclude_target_label_echoes and item_label == target_label:
            return False
        seen_pairs.add(item_pair)
    return True

def itemset_signature(itemset: tuple[str, ...], target_item: str) -> str:
    return " AND ".join(sorted(itemset)) + "=>" + target_item


def growth_rate_value(support_target_ratio: float, support_contrast_ratio: float) -> float:
    # Zero contrast support gives an infinite target-vs-contrast growth rate.
    if support_contrast_ratio == 0:
        return math.inf if support_target_ratio > 0 else 0.0
    return support_target_ratio / support_contrast_ratio


def format_metric(value: float | None) -> float | str:
    # Blank cells are clearer than "inf" or "nan" in the exported CSVs.
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def metric_status(value: float, *, infinite_reason: str = "", zero_reason: str = "") -> str:
    # We keep a companion status field so blanked metrics still explain themselves.
    if math.isnan(value):
        return "undefined_nan"
    if math.isinf(value):
        return infinite_reason or "infinite"
    if value == 0.0 and zero_reason:
        return zero_reason
    return "finite"


def antecedent_satisfied(transaction: dict[str, str], antecedent_items: tuple[str, ...]) -> bool:
    values = set(transaction.values())
    return all(item in values for item in antecedent_items)


def prune_redundant_association_rules(
    rules: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep a longer rule only when it improves on every accepted subset rule."""
    retained: list[dict[str, object]] = []
    rules_by_target: dict[str, list[dict[str, object]]] = defaultdict(list)

    for rule in sorted(
        rules,
        key=lambda candidate: (
            int(candidate["antecedent_length"]),
            -float(candidate["confidence"]),
            -float(candidate["lift"]),
            -int(candidate["rule_support_count"]),
        ),
    ):
        antecedent = set(rule["antecedent_items_tuple"])
        is_redundant = any(
            set(shorter_rule["antecedent_items_tuple"]).issubset(antecedent)
            and float(shorter_rule["confidence"]) >= float(rule["confidence"])
            and float(shorter_rule["lift"]) >= float(rule["lift"])
            for shorter_rule in rules_by_target[str(rule["target_item"])]
        )
        if is_redundant:
            continue
        retained.append(rule)
        rules_by_target[str(rule["target_item"])].append(rule)
    return retained


def mine_high_level_rules(
    dataset: FingerprintDataset,
    config: Step2Config,
    transactions: list[dict[str, str]] | None = None,
    include_coverage: bool = True,
    progress_task: ProgressTask | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if transactions is None:
        transactions = build_transactions(
            dataset,
            exclude_hybrid_items=config.exclude_hybrid_antecedents,
        )
    n_total = len(dataset.trajectory_ids)
    min_support = min_rule_support_count(config, n_total)
    item_to_indices: dict[str, list[int]] = defaultdict(list)
    observed_items: set[str] = set()

    for index, transaction in enumerate(transactions):
        for item in transaction.values():
            observed_items.add(item)
            item_to_indices[item].append(index)

    target_items = set(
        filter_target_items_by_prevalence(
            observed_items,
            item_to_indices,
            n_total,
            config,
        )
    )
    if not target_items:
        if progress_task is not None:
            progress_task.complete("no targets")
        return [], []

    max_itemset_length = config.max_high_level_rule_length + 1
    itemset_counts = mine_frequent_pseudo_label_itemsets(
        transactions,
        min_support,
        max_itemset_length,
    )
    if progress_task is not None:
        progress_task.refresh(note=f"fp-growth itemsets {len(itemset_counts)}")

    raw_rules: list[dict[str, object]] = []
    for itemset, support_target_count in itemset_counts.items():
        if len(itemset) < 2:
            continue
        itemset_items = set(itemset)
        for target_item in sorted(item for item in itemset if item in target_items):
            target_pair, target_label = parse_item(target_item)
            antecedent_items = tuple(sorted(itemset_items.difference({target_item})))
            if not antecedent_items:
                continue
            if len(antecedent_items) > config.max_high_level_rule_length:
                continue
            if not valid_antecedent_for_target(
                antecedent_items,
                target_pair,
                target_label,
                config,
            ):
                continue

            antecedent_support_count = itemset_counts.get(antecedent_items, 0)
            if antecedent_support_count < min_support:
                continue
            n_target = len(item_to_indices[target_item])
            n_contrast = n_total - n_target
            support_contrast_count = antecedent_support_count - support_target_count
            support_target_ratio = support_target_count / n_target if n_target else 0.0
            support_contrast_ratio = support_contrast_count / n_contrast if n_contrast else 0.0
            confidence = support_target_count / antecedent_support_count if antecedent_support_count else 0.0
            if confidence < config.min_confidence:
                continue
            target_base_rate = n_target / n_total if n_total else 0.0
            lift = confidence / target_base_rate if target_base_rate else 0.0
            if lift < config.min_lift:
                continue
            if support_target_ratio < config.min_target_coverage:
                continue
            growth_rate = growth_rate_value(support_target_ratio, support_contrast_ratio)
            support_pt = support_target_count / n_total if n_total else 0.0
            support_p = antecedent_support_count / n_total if n_total else 0.0
            leverage = support_pt - (support_p * target_base_rate)

            raw_rules.append(
                {
                    "target_item": target_item,
                    "target_pair": target_pair,
                    "target_label": target_label,
                    "antecedent_items_tuple": antecedent_items,
                    "antecedent_items": ";".join(antecedent_items),
                    "antecedent_length": len(antecedent_items),
                    "n_total": n_total,
                    "n_target": n_target,
                    "n_contrast": n_contrast,
                    "rule_support_count": support_target_count,
                    "rule_support_ratio": support_pt,
                    "target_coverage": support_target_ratio,
                    "antecedent_support_count": antecedent_support_count,
                    "antecedent_support_ratio": support_p,
                    "confidence": confidence,
                    "target_base_rate": target_base_rate,
                    "lift": lift,
                    "growth_rate": growth_rate,
                    "leverage": leverage,
                    "rule_signature": itemset_signature(antecedent_items, target_item),
                }
            )

    if progress_task is not None:
        progress_task.refresh(note=f"candidate rules {len(raw_rules)}")

    minimal_rules = prune_redundant_association_rules(raw_rules)
    minimal_rules.sort(
        key=lambda rule: (
            -float(rule["confidence"]),
            -float(rule["lift"]),
            -int(rule["rule_support_count"]),
            -float(rule["target_coverage"]),
            -1 if math.isinf(float(rule["growth_rate"])) else 0,
            -float(rule["growth_rate"]) if not math.isinf(float(rule["growth_rate"])) else 0.0,
            int(rule["antecedent_length"]),
            str(rule["target_item"]),
            str(rule["antecedent_items"]),
        )
    )

    rules: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for index, rule in enumerate(minimal_rules, start=1):
        rule_id = f"R{index:05d}"
        rule["rule_id"] = rule_id
        rules.append(rule)
        if not include_coverage:
            continue
        antecedent_items = tuple(rule["antecedent_items_tuple"])
        target_item = str(rule["target_item"])
        for row_index, transaction in enumerate(transactions):
            if not antecedent_satisfied(transaction, antecedent_items):
                continue
            target_satisfied = target_item in transaction.values()
            coverage_rows.append(
                {
                    "rule_id": rule_id,
                    "trajectory_id": dataset.trajectory_ids[row_index],
                    "target_item": target_item,
                    "is_target": target_satisfied,
                    "is_contrast": not target_satisfied,
                    "antecedent_satisfied": True,
                    "target_satisfied": target_satisfied,
                }
            )
    if progress_task is not None:
        progress_task.complete("fp-growth completed")
    return rules, coverage_rows

def pair_pretty(pair_id: str) -> str:
    return pair_id.replace("__x__", " × ")


def humanize_node_name(node_name: str) -> str:
    return node_name.replace("_", " ")


def item_to_plain_language_phrase(item: str, *, as_outcome: bool = False) -> str:
    pair_id, label = parse_item(item)
    left_node, right_node = parse_pair_id(pair_id)
    left_name = humanize_node_name(left_node)
    right_name = humanize_node_name(right_node)
    if label == "Normal":
        if as_outcome:
            return f"the {left_name}–{right_name} comparison is also typical"
        return f"the {left_name}–{right_name} comparison is typical"
    if label == "Hybrid":
        if as_outcome:
            return f"{left_name} and {right_name} also show a shared unusual pattern"
        return f"{left_name} and {right_name} show a shared unusual pattern"
    if label.startswith("Uncommon "):
        node = uncommon_node_from_label(label)
        node_name = humanize_node_name(node)
        if node == left_node:
            if as_outcome:
                return f"{node_name} is also unusual relative to {right_name}"
            return f"{node_name} is unusual relative to {right_name}"
        if node == right_node:
            if as_outcome:
                return f"{node_name} is also unusual relative to {left_name}"
            return f"{node_name} is unusual relative to {left_name}"
        if as_outcome:
            return f"{node_name} is also unusual in the {left_name}–{right_name} comparison"
        return f"{node_name} is unusual in the {left_name}–{right_name} comparison"
    return f"{label} in the {left_name}–{right_name} comparison"


def build_behavioral_catalogue(
    rules: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    coverage_by_rule: dict[str, list[str]] = defaultdict(list)
    for row in coverage_rows:
        coverage_by_rule[str(row["rule_id"])].append(str(row["trajectory_id"]))

    rows: list[dict[str, object]] = []
    for rule in rules:
        antecedent_items = str(rule["antecedent_items"]).split(";") if str(rule["antecedent_items"]) else []
        antecedent_text = " and ".join(
            item_to_plain_language_phrase(item) for item in antecedent_items
        )
        target_text = item_to_plain_language_phrase(str(rule["target_item"]), as_outcome=True)
        covered_ids = coverage_by_rule.get(str(rule["rule_id"]), [])
        rows.append(
            {
                "rule_id": rule["rule_id"],
                "pattern": (
                    f"When {antecedent_text}, {target_text}."
                ),
                "evidence": (
                    f"{rule['rule_support_count']} of {rule['antecedent_support_count']} "
                    f"matching trajectories ({float(rule['confidence']):.1%}) also showed "
                    f"this outcome; this is {float(rule['lift']):.1f}× its overall rate."
                ),
                "n_covered": len(covered_ids),
                "rule_support_count": rule["rule_support_count"],
                "rule_support_ratio": rule["rule_support_ratio"],
                "target_coverage": rule["target_coverage"],
                "confidence": rule["confidence"],
                "lift": rule["lift"],
                "growth_rate": format_metric(float(rule["growth_rate"])),
                "growth_rate_status": metric_status(
                    float(rule["growth_rate"]),
                    infinite_reason="zero_contrast",
                ),
                "covered_trajectory_ids": ";".join(covered_ids),
            }
        )
    return rows


def rule_semantic_required_counts(rule: dict[str, object]) -> Counter[str]:
    antecedent_items = [
        item for item in str(rule["antecedent_items"]).split(";") if item
    ]
    return meta.semantic_counts_from_items(
        [*antecedent_items, str(rule["target_item"])]
    )


def rule_importance_score(rule: dict[str, object]) -> float:
    return (
        float(rule["confidence"])
        * float(rule["lift"])
        * math.log1p(float(rule["rule_support_count"]))
    )


def build_semantic_meta_patterns(
    dataset: FingerprintDataset,
    rules: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pattern_rule_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    pattern_required_counts: dict[str, dict[str, int]] = {}
    for rule in rules:
        required_counts = rule_semantic_required_counts(rule)
        if not required_counts:
            continue
        signature = meta.format_required_counts(required_counts)
        pattern_rule_groups[signature].append(rule)
        pattern_required_counts[signature] = dict(required_counts)

    trajectory_counts = [
        meta.row_uncommon_node_counts(row, dataset.pair_ids)
        for row in dataset.rows
    ]

    pattern_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    source_rule_rows: list[dict[str, object]] = []
    catalogue_rows: list[dict[str, object]] = []
    staged_patterns: list[dict[str, object]] = []

    for signature, source_rules in pattern_rule_groups.items():
        required_counts = pattern_required_counts[signature]
        nodes = sorted(required_counts)
        covered_trajectory_ids: list[str] = []
        for trajectory_id, observed_counts in zip(
            dataset.trajectory_ids,
            trajectory_counts,
            strict=False,
        ):
            if meta.pattern_satisfied(observed_counts, required_counts):
                covered_trajectory_ids.append(trajectory_id)

        best_rule = max(
            source_rules,
            key=lambda rule: (
                rule_importance_score(rule),
                float(rule["confidence"]),
                float(rule["lift"]),
                int(rule["rule_support_count"]),
                str(rule["rule_id"]),
            ),
        )
        mean_confidence = sum(float(rule["confidence"]) for rule in source_rules) / len(source_rules)
        mean_lift = sum(float(rule["lift"]) for rule in source_rules) / len(source_rules)
        mean_target_coverage = (
            sum(float(rule["target_coverage"]) for rule in source_rules) / len(source_rules)
        )
        max_confidence = max(float(rule["confidence"]) for rule in source_rules)
        max_lift = max(float(rule["lift"]) for rule in source_rules)
        max_target_coverage = max(float(rule["target_coverage"]) for rule in source_rules)
        pattern_importance_score = (
            mean_confidence
            * mean_lift
            * math.log1p(len(covered_trajectory_ids))
            * math.log1p(len(source_rules))
        )
        staged_patterns.append(
            {
                "signature": signature,
                "required_counts": required_counts,
                "nodes": nodes,
                "source_rules": source_rules,
                "covered_trajectory_ids": covered_trajectory_ids,
                "best_rule": best_rule,
                "mean_confidence": mean_confidence,
                "mean_lift": mean_lift,
                "mean_target_coverage": mean_target_coverage,
                "max_confidence": max_confidence,
                "max_lift": max_lift,
                "max_target_coverage": max_target_coverage,
                "pattern_importance_score": pattern_importance_score,
            }
        )

    staged_patterns.sort(
        key=lambda pattern: (
            -float(pattern["pattern_importance_score"]),
            -len(pattern["source_rules"]),
            -len(pattern["covered_trajectory_ids"]),
            str(pattern["signature"]),
        )
    )

    for index, pattern in enumerate(staged_patterns, start=1):
        meta_pattern_id = f"MP{index:05d}"
        required_counts = pattern["required_counts"]
        nodes = pattern["nodes"]
        source_rules = pattern["source_rules"]
        covered_trajectory_ids = pattern["covered_trajectory_ids"]
        best_rule = pattern["best_rule"]
        readable_pattern = meta.readable_required_counts(required_counts)
        source_rule_ids = [str(rule["rule_id"]) for rule in source_rules]
        source_target_items = sorted({str(rule["target_item"]) for rule in source_rules})

        pattern_rows.append(
            {
                "meta_pattern_id": meta_pattern_id,
                "pattern_signature": pattern["signature"],
                "required_uncommon_counts": pattern["signature"],
                "nodes": ";".join(nodes),
                "n_nodes": len(nodes),
                "total_required_uncommon_evidence": sum(required_counts.values()),
                "n_source_rules": len(source_rules),
                "source_rule_ids": ";".join(source_rule_ids),
                "source_target_items": ";".join(source_target_items),
                "best_source_rule_id": best_rule["rule_id"],
                "best_source_rule_confidence": best_rule["confidence"],
                "best_source_rule_lift": best_rule["lift"],
                "best_source_rule_support_count": best_rule["rule_support_count"],
                "mean_source_confidence": pattern["mean_confidence"],
                "max_source_confidence": pattern["max_confidence"],
                "mean_source_lift": pattern["mean_lift"],
                "max_source_lift": pattern["max_lift"],
                "mean_source_target_coverage": pattern["mean_target_coverage"],
                "max_source_target_coverage": pattern["max_target_coverage"],
                "n_covered": len(covered_trajectory_ids),
                "coverage_ratio": (
                    len(covered_trajectory_ids) / len(dataset.trajectory_ids)
                    if dataset.trajectory_ids
                    else 0.0
                ),
                "pattern_importance_score": pattern["pattern_importance_score"],
                "readable_pattern": readable_pattern,
                "covered_trajectory_ids": ";".join(covered_trajectory_ids),
            }
        )

        for trajectory_id, observed_counts in zip(
            dataset.trajectory_ids,
            trajectory_counts,
            strict=False,
        ):
            if not meta.pattern_satisfied(observed_counts, required_counts):
                continue
            coverage_rows.append(
                {
                    "meta_pattern_id": meta_pattern_id,
                    "trajectory_id": trajectory_id,
                    "required_uncommon_counts": pattern["signature"],
                    "observed_uncommon_counts": meta.format_observed_counts(
                        observed_counts,
                        nodes,
                    ),
                }
            )

        for rule in source_rules:
            source_rule_rows.append(
                {
                    "meta_pattern_id": meta_pattern_id,
                    "rule_id": rule["rule_id"],
                    "target_item": rule["target_item"],
                    "target_pair": rule["target_pair"],
                    "target_label": rule["target_label"],
                    "antecedent_items": rule["antecedent_items"],
                    "rule_support_count": rule["rule_support_count"],
                    "target_coverage": rule["target_coverage"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                }
            )

        catalogue_rows.append(
            {
                "meta_pattern_id": meta_pattern_id,
                "pattern": readable_pattern,
                "interpretation": (
                    "This meta-pattern summarizes recurring pairwise rules where "
                    f"{readable_pattern}. It should be read as persistent taxonomy-level "
                    "uncommonness rather than a single isolated comparison."
                ),
                "n_covered": len(covered_trajectory_ids),
                "coverage_ratio": (
                    len(covered_trajectory_ids) / len(dataset.trajectory_ids)
                    if dataset.trajectory_ids
                    else 0.0
                ),
                "n_source_rules": len(source_rules),
                "representative_rule_id": best_rule["rule_id"],
                "mean_source_confidence": pattern["mean_confidence"],
                "mean_source_lift": pattern["mean_lift"],
                "pattern_importance_score": pattern["pattern_importance_score"],
                "covered_trajectory_ids": ";".join(covered_trajectory_ids),
            }
        )

    return pattern_rows, coverage_rows, source_rule_rows, catalogue_rows


def export_rule_rows(rules: list[dict[str, object]]) -> list[dict[str, object]]:
    exported_rows: list[dict[str, object]] = []
    for rule in rules:
        growth_rate = float(rule["growth_rate"])
        exported_row = dict(rule)
        exported_row["growth_rate"] = format_metric(growth_rate)
        exported_row["growth_rate_status"] = metric_status(
            growth_rate,
            infinite_reason="zero_contrast",
        )
        exported_rows.append(exported_row)
    return exported_rows


def write_step2_outputs(
    output_dir: Path,
    pair_summary_rows: list[dict[str, object]],
    pair_summary_fieldnames: list[str],
    trajectory_summary_rows: list[dict[str, object]],
    trajectory_summary_fieldnames: list[str],
    item_cooccurrence_rows: list[dict[str, object]],
    rules: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    behavioural_rows: list[dict[str, object]],
    meta_pattern_rows: list[dict[str, object]],
    meta_pattern_coverage_rows: list[dict[str, object]],
    meta_pattern_source_rule_rows: list[dict[str, object]],
    semantic_catalogue_rows: list[dict[str, object]],
) -> None:
    write_csv(output_dir / "pair_label_summary.csv", pair_summary_fieldnames, pair_summary_rows)
    write_csv(
        output_dir / "trajectory_fingerprint_summary.csv",
        trajectory_summary_fieldnames,
        trajectory_summary_rows,
    )
    write_csv(
        output_dir / "high_level_item_cooccurrence.csv",
        [
            "item_a",
            "item_b",
            "count_a",
            "count_b",
            "count_ab",
            "support_a",
            "support_b",
            "support_ab",
            "confidence_a_to_b",
            "confidence_b_to_a",
            "lift",
            "jaccard",
        ],
        item_cooccurrence_rows,
    )
    write_csv(
        output_dir / "high_level_association_rules.csv",
        [
            "rule_id",
            "target_item",
            "target_pair",
            "target_label",
            "antecedent_items",
            "antecedent_length",
            "n_total",
            "n_target",
            "rule_support_count",
            "rule_support_ratio",
            "target_coverage",
            "antecedent_support_count",
            "antecedent_support_ratio",
            "confidence",
            "target_base_rate",
            "lift",
            "growth_rate",
            "growth_rate_status",
            "leverage",
        ],
        export_rule_rows(rules),
    )
    write_csv(
        output_dir / "high_level_rule_coverage.csv",
        [
            "rule_id",
            "trajectory_id",
            "target_item",
            "is_target",
            "is_contrast",
            "antecedent_satisfied",
            "target_satisfied",
        ],
        coverage_rows,
    )
    write_csv(
        output_dir / "behavioral_pattern_catalogue.csv",
        [
            "rule_id",
            "pattern",
            "evidence",
            "n_covered",
            "rule_support_count",
            "rule_support_ratio",
            "target_coverage",
            "confidence",
            "lift",
            "growth_rate",
            "growth_rate_status",
            "covered_trajectory_ids",
        ],
        behavioural_rows,
    )
    write_csv(
        output_dir / "semantic_meta_patterns.csv",
        [
            "meta_pattern_id",
            "pattern_signature",
            "required_uncommon_counts",
            "nodes",
            "n_nodes",
            "total_required_uncommon_evidence",
            "n_source_rules",
            "source_rule_ids",
            "source_target_items",
            "best_source_rule_id",
            "best_source_rule_confidence",
            "best_source_rule_lift",
            "best_source_rule_support_count",
            "mean_source_confidence",
            "max_source_confidence",
            "mean_source_lift",
            "max_source_lift",
            "mean_source_target_coverage",
            "max_source_target_coverage",
            "n_covered",
            "coverage_ratio",
            "pattern_importance_score",
            "readable_pattern",
            "covered_trajectory_ids",
        ],
        meta_pattern_rows,
    )
    write_csv(
        output_dir / "semantic_meta_pattern_coverage.csv",
        [
            "meta_pattern_id",
            "trajectory_id",
            "required_uncommon_counts",
            "observed_uncommon_counts",
        ],
        meta_pattern_coverage_rows,
    )
    write_csv(
        output_dir / "semantic_meta_pattern_source_rules.csv",
        [
            "meta_pattern_id",
            "rule_id",
            "target_item",
            "target_pair",
            "target_label",
            "antecedent_items",
            "rule_support_count",
            "target_coverage",
            "confidence",
            "lift",
        ],
        meta_pattern_source_rule_rows,
    )
    write_csv(
        output_dir / "semantic_behavioral_pattern_catalogue.csv",
        [
            "meta_pattern_id",
            "pattern",
            "interpretation",
            "n_covered",
            "coverage_ratio",
            "n_source_rules",
            "representative_rule_id",
            "mean_source_confidence",
            "mean_source_lift",
            "pattern_importance_score",
            "covered_trajectory_ids",
        ],
        semantic_catalogue_rows,
    )


def write_step2_summary(
    output_dir: Path,
    fingerprint_path: Path,
    dataset: FingerprintDataset,
    depth_filter: DepthFilterSummary,
    config: Step2Config,
    rules: list[dict[str, object]],
    target_item_count: int,
    meta_pattern_rows: list[dict[str, object]],
) -> None:
    payload = {
        "step": "step2",
        "status": "completed",
        "inputs": {
            "fingerprint": str(fingerprint_path),
            "trajectory_id": dataset.trajectory_id_column,
        },
        "dataset": {
            "row_count": len(dataset.trajectory_ids),
            "pair_column_count": len(dataset.pair_ids),
            "original_pair_column_count": len(depth_filter.original_pair_ids),
            "retained_same_depth_pair_count": len(depth_filter.retained_pair_ids),
            "skipped_mixed_depth_pair_count": len(depth_filter.skipped_pair_ids),
            "skipped_out_of_depth_range_pair_count": len(depth_filter.skipped_depth_range_pair_ids),
        },
        "parameters": {
            "mining_algorithm": "fp_growth_association_rules",
            "include_normal_targets": config.include_normal_targets,
            "target_label_scope": "uncommon_only"
            if not config.include_normal_targets
            else "uncommon_and_normal",
            "hybrid_items_excluded_from_antecedents": config.exclude_hybrid_antecedents,
            "target_label_echoes_excluded_from_antecedents": config.exclude_target_label_echoes,
            "max_high_level_rule_length": config.max_high_level_rule_length,
            "max_fp_growth_itemset_length": config.max_high_level_rule_length + 1,
            "min_support_count": config.min_support_count,
            "min_support_ratio": config.min_support_ratio,
            "min_confidence": config.min_confidence,
            "min_lift": config.min_lift,
            "min_target_coverage": config.min_target_coverage,
            "min_target_prevalence_ratio": config.min_target_prevalence_ratio,
            "same_depth_only": True,
            "min_node_depth": config.min_node_depth,
            "max_node_depth": config.max_node_depth,
        },
        "depth_filter": {
            "retained_pair_ids": depth_filter.retained_pair_ids,
            "skipped_pair_ids": depth_filter.skipped_pair_ids,
            "skipped_depth_range_pair_ids": depth_filter.skipped_depth_range_pair_ids,
        },
        "results": {
            "target_item_count": target_item_count,
            "rule_count": len(rules),
            "semantic_meta_pattern_count": len(meta_pattern_rows),
            "rule_type": "fp_growth_association_rule",
            "meta_pattern_type": "node_uncommonness_count_compression",
        },
        "generated_files": {
            filename: str(output_dir / filename) for filename in STEP2_OUTPUT_FILES
        },
    }
    write_json(output_dir / "step2_summary.json", payload)


def run_step2(
    fingerprint_path: Path,
    output_dir: Path,
    trajectory_id: str,
    config: Step2Config,
) -> None:
    progress = ProgressReporter()
    progress.stage("Step 2: loading trajectory fingerprint table")
    original_dataset = load_fingerprint_dataset(fingerprint_path, trajectory_id)
    progress.stage("Step 2: loading taxonomy depths from Step 1 metadata")
    node_depths = load_node_depths(fingerprint_path.with_name("node_feature_sets.csv"))
    progress.stage("Step 2: filtering taxonomy pairs by depth")
    dataset, depth_filter = filter_dataset_to_same_depth_pairs(original_dataset, node_depths, config)
    progress.stage("Step 2: building transaction fingerprints")
    transactions = build_transactions(dataset)
    mining_transactions = build_transactions(
        dataset,
        exclude_hybrid_items=config.exclude_hybrid_antecedents,
    )
    progress.stage("Step 2: summarizing pair labels")
    pair_summary_rows, pair_summary_fieldnames, uncommon_nodes = summarize_pair_labels(dataset)
    progress.stage("Step 2: summarizing trajectory fingerprints")
    trajectory_summary_rows, trajectory_summary_fieldnames = summarize_trajectory_fingerprints(
        dataset,
        uncommon_nodes,
    )
    progress.stage("Step 2: computing item co-occurrence summary")
    item_cooccurrence_rows = summarize_item_cooccurrence(dataset, transactions)
    progress.stage("Step 2: mining high-level association rules")
    item_to_indices: dict[str, list[int]] = defaultdict(list)
    observed_items: set[str] = set()
    for index, transaction in enumerate(mining_transactions):
        for item in transaction.values():
            observed_items.add(item)
            item_to_indices[item].append(index)
    target_items = filter_target_items_by_prevalence(
        observed_items,
        item_to_indices,
        len(dataset.trajectory_ids),
        config,
    )
    observed_target_items = sorted(
        item
        for transaction in mining_transactions
        for item in transaction.values()
        if is_target_label(parse_item(item)[1], config.include_normal_targets)
    )
    progress.stage(
        "Step 2: "
        f"{len(target_items)} target items retained after prevalence filtering "
        f"(from {len(sorted(set(observed_target_items)))})"
    )
    mining_task = progress.task(
        "FP-growth rule mining",
        1,
    )
    rules, coverage_rows = mine_high_level_rules(
        dataset,
        config,
        transactions=mining_transactions,
        include_coverage=True,
        progress_task=mining_task,
    )
    mining_task.complete("completed")
    progress.stage("Step 2: building behavioral catalogue")
    behavioural_rows = build_behavioral_catalogue(rules, coverage_rows)
    progress.stage("Step 2: compressing rules into semantic meta-patterns")
    (
        meta_pattern_rows,
        meta_pattern_coverage_rows,
        meta_pattern_source_rule_rows,
        semantic_catalogue_rows,
    ) = build_semantic_meta_patterns(dataset, rules)
    progress.stage("Step 2: writing Step 2 outputs")
    write_step2_outputs(
        output_dir,
        pair_summary_rows,
        pair_summary_fieldnames,
        trajectory_summary_rows,
        trajectory_summary_fieldnames,
        item_cooccurrence_rows,
        rules,
        coverage_rows,
        behavioural_rows,
        meta_pattern_rows,
        meta_pattern_coverage_rows,
        meta_pattern_source_rule_rows,
        semantic_catalogue_rows,
    )
    progress.stage("Step 2: writing Step 2 summary")
    write_step2_summary(
        output_dir,
        fingerprint_path,
        dataset,
        depth_filter,
        config,
        rules,
        len(target_items),
        meta_pattern_rows,
    )
    LOGGER.info(
        "Prepared Step 2 analysis for %d trajectories across %d retained pair columns "
        "(filtered from %d total pair columns) with %d rules and %d semantic meta-patterns.",
        len(dataset.trajectory_ids),
        len(dataset.pair_ids),
        len(depth_filter.original_pair_ids),
        len(rules),
        len(meta_pattern_rows),
    )
