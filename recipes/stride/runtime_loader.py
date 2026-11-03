"""Load the pinned CUDA runtime required by the measured xFormers binary."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path

LIBRARY_SHA256 = "5a077122071ec6479e814be154a546a51d6b631ccb2718f08427ea5c0a5e6c0a"
_loaded = None


def load_cuda_runtime(required: bool = True) -> dict | None:
    global _loaded
    explicit = os.environ.get("FLG_CUDA12_LIBRARY")
    if explicit:
        path = Path(explicit).resolve()
    else:
        spec = importlib.util.find_spec("torch")
        path = (Path(spec.origin).parent.parent / "cuda12_overlay/lib/libcudart.so.12"
                if spec and spec.origin else Path("/nonexistent-flg-runtime"))
    if not path.is_file():
        if not required and not explicit:
            return None
        raise ValueError("Missing pinned CUDA 12 library; run install_runtime.py first")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != LIBRARY_SHA256:
        raise ValueError("CUDA 12 runtime differs from the measured dependency")
    # The absolute path is inherited by spawned vLLM workers, whose import of
    # run.py calls this loader before importing CUDA-dependent libraries.
    os.environ["FLG_CUDA12_LIBRARY"] = str(path)
    if _loaded is None:
        _loaded = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return dict(version="12.0.107", sha256=actual, bytes=path.stat().st_size)
