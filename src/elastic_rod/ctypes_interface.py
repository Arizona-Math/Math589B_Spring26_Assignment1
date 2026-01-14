from __future__ import annotations
import ctypes
import os
from pathlib import Path
import numpy as np

class RodLib:
    """
    ctypes interface to the C++ energy+gradient kernel.

    Exposes:
        energy_grad(N, x, kb, ks, l0, q, kappa) -> (E, g)
    """
    def __init__(self, lib_path: str | os.PathLike | None = None):
        if lib_path is None:
            # default: alongside csrc/
            root = Path(__file__).resolve().parents[2]
            csrc = root / "csrc"
            # prefer .so, then .dylib
            candidates = [csrc / "librod.so", csrc / "librod.dylib", csrc / "rod.dll"]
            for p in candidates:
                if p.exists():
                    lib_path = p
                    break
            else:
                raise FileNotFoundError("Could not find shared library. Run: bash csrc/build.sh")
        self.lib = ctypes.CDLL(str(lib_path))

        # void rod_energy_grad(int N, const double* x, double kb, double ks, double l0,
        #                      double q, double kappa, double* energy_out, double* grad_out)
        self.lib.rod_energy_grad.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.rod_energy_grad.restype = None

    def energy_grad(self, x: np.ndarray, kb: float, ks: float, l0: float, q: float, kappa: float):
        x = np.ascontiguousarray(x, dtype=np.float64)
        assert x.ndim == 1 and x.size % 3 == 0
        N = x.size // 3
        energy = ctypes.c_double(0.0)
        g = np.zeros_like(x)
        self.lib.rod_energy_grad(
            ctypes.c_int(N),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(kb),
            ctypes.c_double(ks),
            ctypes.c_double(l0),
            ctypes.c_double(q),
            ctypes.c_double(kappa),
            ctypes.byref(energy),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return float(energy.value), g
