"""Regenerate shared spatio-temporal trajectory features for bundled datasets."""

from __future__ import annotations

from pathlib import Path

from feature_engineering.spatiotemporal import DatasetPaths, write_augmented_dataset


def dataset_paths() -> list[DatasetPaths]:
    """Return the bundled dataset paths that share the same trajectory schema."""
    base_dir = Path("datasets")
    return [
        DatasetPaths(
            name=dataset_name,
            point_features_path=base_dir / dataset_name / f"{dataset_name}-point-feats.csv",
            trajectory_features_path=base_dir / dataset_name / f"{dataset_name}-traj-feats.csv",
        )
        for dataset_name in ["ais", "fox", "hurricanes"]
    ]


def main() -> int:
    """Rebuild the trajectory-level feature tables for all bundled datasets."""
    for paths in dataset_paths():
        rows = write_augmented_dataset(paths)
        column_count = len(rows[0]) if rows else 0
        print(
            f"{paths.name}: wrote {paths.trajectory_features_path} "
            f"with {column_count} columns and {len(rows)} rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
