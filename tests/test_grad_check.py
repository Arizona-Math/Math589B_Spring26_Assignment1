from __future__ import annotations
import numpy as np
import pytest

from elastic_rod.model import RodEnergy, RodParams
from elastic_rod.utils import random_loop, pack

def finite_diff(f, x, eps=1e-6, k=25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=min(k, x.size), replace=False)
    g = np.zeros_like(x)
    for i in idx:
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        fp, _ = f(xp)
        fm, _ = f(xm)
        g[i] = (fp - fm) / (2*eps)
    return idx, g

@pytest.mark.parametrize("N", [50])
def test_gradient_matches_finite_difference(N):
    # Parameters chosen to keep things stable for finite differences.
    params = RodParams(kb=1.0, ks=60.0, l0=0.5, kc=0.01, eps=1.0, sigma=0.35)
    model = RodEnergy(params)
    x0 = pack(random_loop(N, radius=6.5, noise=0.3, seed=1))

    f, g = model.value_and_grad(x0)
    idx, g_fd = finite_diff(model.value_and_grad, x0, eps=1e-6, k=30, seed=2)

    denom = np.maximum(1.0, np.abs(g_fd[idx]) + np.abs(g[idx]))
    rel = np.abs(g[idx] - g_fd[idx]) / denom

    assert np.max(rel) < 8e-4
