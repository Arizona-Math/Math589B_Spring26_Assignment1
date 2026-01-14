from __future__ import annotations
import numpy as np

def pack(X: np.ndarray) -> np.ndarray:
    """(N,3) -> (3N,) contiguous float64"""
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == 3
    return np.ascontiguousarray(X.reshape(-1))

def unpack(x: np.ndarray) -> np.ndarray:
    """(3N,) -> (N,3)"""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and (x.size % 3 == 0)
    return x.reshape((-1, 3))

def random_loop(N: int, radius: float = 5.0, noise: float = 0.2, seed: int = 0) -> np.ndarray:
    """A noisy circle in the xy-plane as a reasonable initial closed curve."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    X = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    X += noise * rng.normal(size=X.shape)
    # enforce exact closure by shifting mean (not strictly necessary, but helps)
    X -= X.mean(axis=0, keepdims=True)
    return X
