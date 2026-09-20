"""Append-only CSV logging for single-run AURA results."""

import csv
import os
from typing import Any, Dict, Mapping

RESULT_FIELDS = (
    "dataset_name",
    "bottleneck_depth",
    "loss_fn",
    "img_size",
    "category",
    "auroc",
    "timestamp",
)


def log_result(csv_path: str, row_dict: Mapping[str, Any]) -> None:
    """Append one result row to ``csv_path``, writing a header if the file is new."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    row = {field: row_dict.get(field, "") for field in RESULT_FIELDS}

    # Sequential appends: open/write/close per call avoids leaving a stale handle.
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
