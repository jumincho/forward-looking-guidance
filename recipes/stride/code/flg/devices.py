"""Preserve scheduler device masks when launching vLLM workers."""
from __future__ import annotations

import os
import subprocess


def numeric_device_mask(mask: str) -> str:
    """vLLM 0.11 expects physical numeric IDs rather than GPU UUIDs."""
    devices = [d.strip() for d in mask.split(",")]
    if all(d.isdigit() for d in devices):
        return ",".join(devices)
    if not all(d.startswith("GPU-") for d in devices):
        raise ValueError("Expected a numeric or full GPU-UUID device mask; MIG is not supported")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    mapping = {}
    for line in result.stdout.splitlines():
        index, uuid = (v.strip() for v in line.split(",", 1))
        mapping[uuid] = index
    if any(d not in mapping for d in devices):
        raise ValueError("An admitted GPU UUID is absent from the current device inventory")
    return ",".join(mapping[d] for d in devices)


def normalize_vllm_mask() -> None:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask and "GPU-" in mask:
        os.environ["CUDA_VISIBLE_DEVICES"] = numeric_device_mask(mask)


def worker_mask(group: list[int], inherited: str | None) -> str:
    """Assignment indices are relative to the parent's visible device list."""
    if not group or any(type(g) is not int or g < 0 for g in group):
        raise ValueError("GPU assignments must contain nonnegative integer indices")
    if inherited is None:
        return ",".join(map(str, group))
    available = numeric_device_mask(inherited).split(",")
    if max(group) >= len(available):
        raise ValueError("GPU assignment exceeds the inherited allocation")
    return ",".join(available[g] for g in group)
