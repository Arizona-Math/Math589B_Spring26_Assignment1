from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ctypes_interface import RodLib

@dataclass(frozen=True)
class RodParams:
    kb: float = 1.0     # bending stiffness
    ks: float = 80.0    # stretching stiffness
    l0: float = 0.5     # rest length
    kc: float = 0.02    # confinement (small >0 creates coiling/packing)
    eps: float = 1.0    # WCA epsilon
    sigma: float = 0.35 # WCA sigma

class RodEnergy:
    def __init__(self, params: RodParams | None = None, lib_path=None):
        self.params = params or RodParams()
        self.lib = RodLib(lib_path=lib_path)

    def value_and_grad(self, x: np.ndarray):
        p = self.params
        return self.lib.energy_grad(x, p.kb, p.ks, p.l0, p.kc, p.eps, p.sigma)
