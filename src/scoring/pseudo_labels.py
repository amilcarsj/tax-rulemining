"""Pseudo-label generation from node score pairs."""

from __future__ import annotations

from core.progress import ProgressTask


def assign_zone(score_a: float, score_b: float) -> int:
    if score_a < 0.5 and score_b < 0.5:
        return 0
    if score_a < 0.5 and score_b > 0.5 and score_a < (score_b - 0.5):
        return 1
    if score_a > 0.5 and score_b < (score_a - 0.5):
        return 2
    return 3


def zone_label(zone: int, node_a_name: str, node_b_name: str) -> str:
    if zone == 0:
        return f"Normal {node_a_name}-{node_b_name}"
    if zone == 1:
        return f"Uncommon {node_b_name}"
    if zone == 2:
        return f"Uncommon {node_a_name}"
    return f"Hybrid {node_a_name}-{node_b_name}"


def generate_pair_pseudo_labels(
    trajectory_ids: list[str],
    pairs: list[dict[str, str | int]],
    score_lookup: dict[str, dict[str, float]],
    node_name_lookup: dict[str, str],
    progress_task: ProgressTask | None = None,
) -> list[dict[str, str | int | float]]:
    pseudo_label_rows: list[dict[str, str | int | float]] = []

    for pair in pairs:
        node_a = str(pair["node_a"])
        node_b = str(pair["node_b"])
        node_a_name = node_name_lookup.get(node_a, node_a)
        node_b_name = node_name_lookup.get(node_b, node_b)
        pair_id = str(pair["pair_id"])
        scores_a = score_lookup.get(node_a)
        scores_b = score_lookup.get(node_b)
        if scores_a is None or scores_b is None:
            continue

        for trajectory_id in trajectory_ids:
            score_a = scores_a[trajectory_id]
            score_b = scores_b[trajectory_id]
            label_code = assign_zone(score_a, score_b)
            pseudo_label_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "pair_id": pair_id,
                    "node_a": node_a,
                    "node_b": node_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "pseudo_label_code": label_code,
                    "pseudo_label": zone_label(label_code, node_a_name, node_b_name),
                }
            )
        if progress_task is not None:
            # Each processed taxonomy pair advances the pseudo-label generation stage.
            progress_task.advance(1, note=pair_id)

    return pseudo_label_rows


def build_trajectory_pseudo_label_table(
    trajectory_ids: list[str],
    pairs: list[dict[str, str | int]],
    pseudo_label_rows: list[dict[str, str | int | float]],
) -> list[dict[str, str]]:
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    table_by_trajectory = {
        trajectory_id: {"trajectory_id": trajectory_id, **{pair_id: "" for pair_id in pair_ids}}
        for trajectory_id in trajectory_ids
    }

    for row in pseudo_label_rows:
        trajectory_id = str(row["trajectory_id"])
        pair_id = str(row["pair_id"])
        table_by_trajectory[trajectory_id][pair_id] = str(row["pseudo_label"])

    return [table_by_trajectory[trajectory_id] for trajectory_id in trajectory_ids]
