"""Shared spatio-temporal feature generation from point-level trajectories."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """Input/output paths for one dataset."""

    name: str
    point_features_path: Path
    trajectory_features_path: Path


def parse_time(value: str) -> datetime:
    """Parse nanosecond timestamp strings by trimming to microseconds."""
    text = str(value).strip()
    if "." in text:
        prefix, suffix = text.split(".", 1)
        text = f"{prefix}.{suffix[:6].ljust(6, '0')}"
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def to_float(value: str) -> float:
    """Convert a CSV field to float with a robust zero fallback."""
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters."""
    earth_radius_m = 6_371_000.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a_value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.asin(math.sqrt(a_value))


def mean(values: list[float]) -> float:
    """Return an arithmetic mean or zero for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    """Return population standard deviation or zero for tiny lists."""
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def median(sorted_values: list[float]) -> float:
    """Return the median of a pre-sorted list."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def quantile(sorted_values: list[float], probability: float) -> float:
    """Return a linear-interpolated quantile."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if count == 1:
        return sorted_values[0]
    position = (count - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def iqr(values: list[float]) -> float:
    """Return the interquartile range."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    return quantile(sorted_values, 0.75) - quantile(sorted_values, 0.25)


def progress_slope(progress: list[float], values: list[float]) -> float:
    """Estimate a least-squares slope over normalized trajectory progress."""
    if len(progress) < 2 or len(values) < 2:
        return 0.0
    progress_mean = mean(progress)
    values_mean = mean(values)
    denominator = sum((value - progress_mean) ** 2 for value in progress)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (x_value - progress_mean) * (y_value - values_mean)
        for x_value, y_value in zip(progress, values, strict=False)
    )
    return numerator / denominator


def entropy_from_counts(counts: list[int]) -> float:
    """Compute Shannon entropy with base-2 logarithm."""
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


def relative_grid_cells(latitudes: list[float], longitudes: list[float], grid_size: int = 3) -> list[str]:
    """Encode each point into a trajectory-relative grid cell to support scale-free complexity features."""
    if not latitudes or not longitudes:
        return []
    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    cells: list[str] = []
    for lat_value, lon_value in zip(latitudes, longitudes, strict=False):
        if lat_span == 0.0:
            lat_bin = grid_size // 2
        else:
            lat_position = (lat_value - min_lat) / lat_span
            lat_bin = min(grid_size - 1, int(lat_position * grid_size))
        if lon_span == 0.0:
            lon_bin = grid_size // 2
        else:
            lon_position = (lon_value - min_lon) / lon_span
            lon_bin = min(grid_size - 1, int(lon_position * grid_size))
        cells.append(f"{lat_bin}_{lon_bin}")
    return cells


def transition_entropy(states: list[str]) -> float:
    """Compute entropy over adjacent state transitions."""
    if len(states) < 2:
        return 0.0
    counts: dict[str, int] = {}
    for left_state, right_state in zip(states[:-1], states[1:], strict=False):
        transition = f"{left_state}->{right_state}"
        counts[transition] = counts.get(transition, 0) + 1
    return entropy_from_counts(list(counts.values()))


def phase_split(progress: list[float], fallback_size: int) -> tuple[list[int], list[int]]:
    """Split points into early and late phases using normalized time or index fallback."""
    early = [index for index, value in enumerate(progress) if value <= 0.5]
    late = [index for index, value in enumerate(progress) if value > 0.5]
    if not early or not late:
        split_index = max(1, fallback_size // 2)
        early = list(range(split_index))
        late = list(range(split_index, fallback_size))
    return early, late


def list_mean(values: list[float], indices: list[int]) -> float:
    """Return the mean of the selected indices."""
    if not indices:
        return 0.0
    return mean([values[index] for index in indices])


def aggregate_trajectory(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    """Build one augmented trajectory-level feature row from point-level observations."""
    ordered = sorted(rows, key=lambda row: row["time"])
    times = [parse_time(row["time"]) for row in ordered]
    latitudes = [to_float(row["lat"]) for row in ordered]
    longitudes = [to_float(row["lon"]) for row in ordered]
    speeds = [to_float(row["speed"]) for row in ordered]
    accelerations = [to_float(row["acceleration"]) for row in ordered]
    angles = [to_float(row["angle"]) for row in ordered]
    distances = [max(0.0, to_float(row["distance"])) for row in ordered]
    point_count = len(ordered)

    elapsed_seconds = [
        max(0.0, (time_value - times[0]).total_seconds())
        for time_value in times
    ]
    duration_total_seconds = elapsed_seconds[-1] if elapsed_seconds else 0.0
    if duration_total_seconds > 0.0:
        progress = [value / duration_total_seconds for value in elapsed_seconds]
    elif point_count > 1:
        progress = [index / (point_count - 1) for index in range(point_count)]
    else:
        progress = [0.0]

    gap_seconds = [
        max(0.0, (right_time - left_time).total_seconds())
        for left_time, right_time in zip(times[:-1], times[1:], strict=False)
    ]
    early_indices, late_indices = phase_split(progress, point_count)

    min_lat = min(latitudes) if latitudes else 0.0
    max_lat = max(latitudes) if latitudes else 0.0
    min_lon = min(longitudes) if longitudes else 0.0
    max_lon = max(longitudes) if longitudes else 0.0
    lat_span_deg = max_lat - min_lat
    lon_span_deg = max_lon - min_lon
    mean_lat = mean(latitudes)
    mean_lon = mean(longitudes)
    bbox_height_m = haversine_m(min_lat, mean_lon, max_lat, mean_lon) if latitudes else 0.0
    bbox_width_m = haversine_m(mean_lat, min_lon, mean_lat, max_lon) if longitudes else 0.0
    bbox_diagonal_m = haversine_m(min_lat, min_lon, max_lat, max_lon) if latitudes and longitudes else 0.0
    bbox_area_km2 = (bbox_height_m * bbox_width_m) / 1_000_000.0

    path_length_m = sum(distances)
    net_displacement_m = (
        haversine_m(latitudes[0], longitudes[0], latitudes[-1], longitudes[-1])
        if point_count >= 2
        else 0.0
    )
    directness_ratio = net_displacement_m / path_length_m if path_length_m > 0.0 else 0.0

    centroid_distances_m = [
        haversine_m(lat_value, lon_value, mean_lat, mean_lon)
        for lat_value, lon_value in zip(latitudes, longitudes, strict=False)
    ]
    sorted_centroid_distances = sorted(centroid_distances_m)
    radius_of_gyration_m = (
        math.sqrt(sum(distance_value ** 2 for distance_value in centroid_distances_m) / len(centroid_distances_m))
        if centroid_distances_m
        else 0.0
    )

    relative_cells = relative_grid_cells(latitudes, longitudes, grid_size=3)
    cell_counts: dict[str, int] = {}
    for cell in relative_cells:
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    unique_relative_grid_cells = len(cell_counts)
    relative_grid_entropy = entropy_from_counts(list(cell_counts.values()))
    relative_transition_entropy = transition_entropy(relative_cells)
    relative_grid_revisit_ratio = (
        1.0 - (unique_relative_grid_cells / len(relative_cells))
        if relative_cells
        else 0.0
    )

    sampling_gap_mean = mean(gap_seconds)
    sampling_gap_sd = std(gap_seconds)
    row: dict[str, float | int | str] = {
        "trajectory_id": ordered[0]["trajectory_id"],
        "object_id": ordered[0]["object_id"],
        "point_count": point_count,
        "duration_total_seconds": duration_total_seconds,
        "sampling_gap_mean_seconds": sampling_gap_mean,
        "sampling_gap_sd_seconds": sampling_gap_sd,
        "sampling_gap_iqr_seconds": iqr(gap_seconds),
        "sampling_gap_cv": sampling_gap_sd / sampling_gap_mean if sampling_gap_mean > 0.0 else 0.0,
        "speed_mean_first_half": list_mean(speeds, early_indices),
        "speed_mean_second_half": list_mean(speeds, late_indices),
        "speed_mean_half_diff": list_mean(speeds, late_indices) - list_mean(speeds, early_indices),
        "acceleration_mean_first_half": list_mean(accelerations, early_indices),
        "acceleration_mean_second_half": list_mean(accelerations, late_indices),
        "acceleration_mean_half_diff": (
            list_mean(accelerations, late_indices) - list_mean(accelerations, early_indices)
        ),
        "bbox_lat_span_deg": lat_span_deg,
        "bbox_lon_span_deg": lon_span_deg,
        "bbox_height_m": bbox_height_m,
        "bbox_width_m": bbox_width_m,
        "bbox_diagonal_m": bbox_diagonal_m,
        "bbox_area_km2": bbox_area_km2,
        "path_length_m": path_length_m,
        "net_displacement_m": net_displacement_m,
        "directness_ratio": directness_ratio,
        "mean_distance_to_centroid_m": mean(centroid_distances_m),
        "median_distance_to_centroid_m": median(sorted_centroid_distances),
        "max_distance_to_centroid_m": max(centroid_distances_m) if centroid_distances_m else 0.0,
        "radius_of_gyration_m": radius_of_gyration_m,
        "speed_progress_slope": progress_slope(progress, speeds),
        "acceleration_progress_slope": progress_slope(progress, accelerations),
        "angle_progress_slope": progress_slope(progress, angles),
        "relative_grid_unique_cells_3x3": unique_relative_grid_cells,
        "relative_grid_entropy_3x3": relative_grid_entropy,
        "relative_transition_entropy_3x3": relative_transition_entropy,
        "relative_grid_revisit_ratio_3x3": relative_grid_revisit_ratio,
    }
    return row


def build_augmented_trajectory_features(point_features_path: Path) -> tuple[list[str], list[dict[str, float | int | str]]]:
    """Aggregate every trajectory into the shared spatio-temporal feature set."""
    groups: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    with point_features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trajectory_id = row["trajectory_id"]
            if trajectory_id not in groups:
                groups[trajectory_id] = []
                order.append(trajectory_id)
            groups[trajectory_id].append(row)
    aggregated_rows = [aggregate_trajectory(groups[trajectory_id]) for trajectory_id in order]
    fieldnames = list(aggregated_rows[0].keys()) if aggregated_rows else []
    return fieldnames, aggregated_rows


def augment_existing_trajectory_features(
    existing_trajectory_features_path: Path,
    point_features_path: Path,
) -> tuple[list[str], list[dict[str, str | float | int]]]:
    """Append shared spatio-temporal features to the existing trajectory-level feature table."""
    _, augmented_rows = build_augmented_trajectory_features(point_features_path)
    augmented_by_id = {
        str(row["trajectory_id"]): row
        for row in augmented_rows
    }

    with existing_trajectory_features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    extra_columns = [
        column_name
        for column_name in augmented_rows[0].keys()
        if column_name not in {"trajectory_id", "object_id"}
    ] if augmented_rows else []
    # We rewrite the trajectory table idempotently, so any previously generated
    # spatio-temporal columns must be removed before we append the fresh values.
    existing_base_fieldnames = [
        fieldname
        for fieldname in existing_fieldnames
        if fieldname not in extra_columns
    ]
    merged_rows: list[dict[str, str | float | int]] = []
    for row in existing_rows:
        trajectory_id = str(row["trajectory_id"])
        augmented = augmented_by_id.get(trajectory_id)
        if augmented is None:
            raise ValueError(f"Missing augmented features for trajectory_id {trajectory_id}.")
        merged = {fieldname: row.get(fieldname, "") for fieldname in existing_base_fieldnames}
        for column_name in extra_columns:
            merged[column_name] = augmented[column_name]
        merged_rows.append(merged)

    return [*existing_base_fieldnames, *extra_columns], merged_rows


def write_augmented_dataset(paths: DatasetPaths) -> list[dict[str, str | float | int]]:
    """Regenerate one dataset trajectory table in place with the new shared feature family."""
    fieldnames, merged_rows = augment_existing_trajectory_features(
        paths.trajectory_features_path,
        paths.point_features_path,
    )
    with paths.trajectory_features_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return merged_rows
