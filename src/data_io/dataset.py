"""Dataset loading helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from core.models import DatasetSchema, DatasetTable


def load_dataset_schema(data_path: Path, trajectory_id_column: str) -> DatasetSchema:
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"Dataset CSV is empty: {data_path}")
        if trajectory_id_column not in headers:
            raise ValueError(
                "Dataset is missing the required trajectory identifier column "
                f"'{trajectory_id_column}'."
            )

        candidate_columns = [column for column in headers if column != trajectory_id_column]
        numeric_flags = {column: True for column in candidate_columns}
        row_count = 0

        for row_count, row in enumerate(reader, start=1):
            for column in candidate_columns:
                if not numeric_flags[column]:
                    continue
                value = row.get(column, "")
                if value is None:
                    continue
                stripped = value.strip()
                if stripped == "":
                    continue
                try:
                    float(stripped)
                except ValueError:
                    numeric_flags[column] = False

    numeric_columns = {column for column, is_numeric in numeric_flags.items() if is_numeric}
    return DatasetSchema(
        trajectory_id_column=trajectory_id_column,
        headers=headers,
        numeric_columns=numeric_columns,
        row_count=row_count,
    )


def load_dataset_table(data_path: Path, trajectory_id_column: str) -> DatasetTable:
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"Dataset CSV is empty: {data_path}")

        columns = {header: [] for header in headers}
        trajectory_ids: list[str] = []

        for row in reader:
            trajectory_id = row.get(trajectory_id_column)
            if trajectory_id is None or str(trajectory_id).strip() == "":
                raise ValueError(
                    "Dataset contains an empty trajectory identifier in column "
                    f"'{trajectory_id_column}'."
                )

            trajectory_ids.append(str(trajectory_id))
            for header in headers:
                value = row.get(header, "")
                columns[header].append("" if value is None else str(value))

    return DatasetTable(
        trajectory_ids=trajectory_ids,
        columns=columns,
        row_count=len(trajectory_ids),
    )
