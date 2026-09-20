"""Append-only CSV logging for single-run AURA results."""

import csv
import os
from typing import Any, Mapping, Optional, Sequence, Set, Tuple

RESULT_FIELDS = (
    "dataset_name",
    "bottleneck_depth",
    "loss_fn",
    "img_size",
    "category",
    "auroc",
    "timestamp",
)

# Key used for resume-safety lookups.
ResultKey = Tuple[str, str, str, str]  # category, depth, loss_fn, img_size


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


def _row_key(row: Mapping[str, Any]) -> ResultKey:
    return (
        str(row.get("category", "")),
        str(row.get("bottleneck_depth", "")),
        str(row.get("loss_fn", "")),
        str(row.get("img_size", "")),
    )


def load_completed_keys(csv_path: str) -> Set[ResultKey]:
    """Return set of (category, depth, loss_fn, img_size) keys already in the CSV."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    keys: Set[ResultKey] = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(_row_key(row))
    return keys


def has_result(
    csv_path: str,
    category: str,
    bottleneck_depth: int,
    loss_fn: str,
    img_size: int,
) -> bool:
    """True if a matching result row already exists (resume-safety check)."""
    key = (str(category), str(bottleneck_depth), str(loss_fn), str(img_size))
    return key in load_completed_keys(csv_path)
