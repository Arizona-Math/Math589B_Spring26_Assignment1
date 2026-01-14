from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ctypes_interface import RodLib

@dataclass(frozen=True)
class RodParams:
    kb: float = 2.0      # bending stiffness
    ks: float = 50.0     # stretching stiffness (big -> near-inextensible)
    l0: float = 0.5      # rest segment length
    q: float = 1.0       # charge magnitude per node
    kappa: float = 0.3   # screening (0 = unscreened)

class RodEnergy:
    def __init__(self, params: RodParams | None = None, lib_path=None):
        self.params = params or RodParams()
        self.lib = RodLib(lib_path=lib_path)

    def value_and_grad(self, x: np.ndarray):
        p = self.params
        return self.lib.energy_grad(x, p.kb, p.ks, p.l0, p.q, p.kappa)
