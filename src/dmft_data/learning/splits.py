"""Explicit three-way blocked splits, keeping all replicas in one partition."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def block_key(value) -> str:
    """Keep numeric beta labels consistent across JSON floats and YAML integers."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, ".15g")
    return str(value)


def blocked_partition(records: list[dict], specification: dict) -> dict[str, list[int]]:
    dimension = specification.get("dimension", "material")
    if dimension not in {"material", "beta", "parameter_block", "group_id"}:
        raise ValueError(
            "Split dimension must be material, beta, parameter_block, or group_id"
        )
    validation = {block_key(v) for v in specification["validation"]}
    test = {block_key(v) for v in specification["test"]}
    if not validation or not test or validation & test:
        raise ValueError("Validation and test blocks must be non-empty and disjoint")
    partitions = {"train": [], "validation": [], "test": []}
    group_roles: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        value = record.get(dimension)
        if value is None or str(value).lower() in {"unknown", "unresolved", ""}:
            raise ValueError(
                f"Missing split metadata {dimension}: {record['relative_source_path']}"
            )
        role = (
            "test"
            if block_key(value) in test
            else "validation"
            if block_key(value) in validation
            else "train"
        )
        partitions[role].append(index)
        group_roles[record["group_id"]].add(role)
    if any(len(roles) > 1 for roles in group_roles.values()):
        raise ValueError("Replica/physical-state group crosses partition boundaries")
    if any(not indices for indices in partitions.values()):
        raise ValueError("Train, validation, and test must all have samples")
    return partitions


def aggregate_groups(
    x: np.ndarray, y: np.ndarray, records: list[dict], indices: list[int]
):
    """Arithmetic run means per reviewed group; retain channels and frequencies."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[records[index]["group_id"]].append(index)
    return (
        np.stack([x[members].mean(axis=0) for members in groups.values()]),
        np.stack([y[members].mean(axis=0) for members in groups.values()]),
        list(groups),
    )
