"""
Linear-algebra backend for the KFC family: NumPy by default, optional CUDA.

Opt-in via environment variable ``KFC_GPU=1``. The GPU path is used only when
torch imports, CUDA is available, and the matrix is at least ``KFC_GPU_MIN_N``
on a side (default 1500 — below that the transfer overhead loses to NumPy).
Precision on GPU is float64 unless ``KFC_GPU_FP32=1`` (only enable fp32 after
the equivalence gate in ``check_gpu_equivalence.py`` passes). Any GPU failure
falls back to NumPy and disables the GPU for the rest of the process — the
caller never sees an exception from the backend.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_MIN_N = int(os.environ.get("KFC_GPU_MIN_N", "1500"))
_state: dict = {"checked": False, "torch": None, "disabled": False}


def _torch():
    """Import torch once; None if unavailable, CUDA missing, or opted out."""
    if _state["checked"]:
        return _state["torch"]
    _state["checked"] = True
    if os.environ.get("KFC_GPU") != "1":
        return None
    try:
        import torch
        if torch.cuda.is_available():
            _state["torch"] = torch
    except Exception as exc:  # noqa: BLE001
        print(f"[kfc_linalg] GPU requested but unusable ({exc}); "
              "using NumPy.", file=sys.stderr)
    return _state["torch"]


def gpu_active() -> bool:
    """True when the CUDA path is available and not disabled by an error."""
    return _torch() is not None and not _state["disabled"]


def inv(a: np.ndarray) -> np.ndarray:
    """Dense inverse; CUDA-accelerated for large matrices when enabled."""
    torch = _torch()
    if torch is None or _state["disabled"] or a.shape[0] < _MIN_N:
        return np.linalg.inv(a)
    try:
        dtype = (torch.float32 if os.environ.get("KFC_GPU_FP32") == "1"
                 else torch.float64)
        t = torch.as_tensor(a, dtype=dtype, device="cuda")
        out = torch.linalg.inv(t).cpu().numpy().astype(np.float64)
        return out
    except Exception as exc:  # noqa: BLE001
        _state["disabled"] = True
        print(f"[kfc_linalg] GPU inv failed ({exc}); NumPy fallback for the "
              "rest of this process.", file=sys.stderr)
        return np.linalg.inv(a)
