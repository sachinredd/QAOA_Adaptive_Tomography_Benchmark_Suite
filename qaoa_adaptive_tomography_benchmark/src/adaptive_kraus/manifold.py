"""Complex Stiefel geometry with real inner product Re Tr(A^dagger B)."""
from __future__ import annotations
import numpy as np


def inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.vdot(a, b).real)


def polar(a: np.ndarray) -> np.ndarray:
    """Nearest column isometry; used as a retraction R_K(V)=polar(K+V)."""
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    return u @ vh


def project_tangent(k: np.ndarray, g: np.ndarray) -> np.ndarray:
    product = k.conj().T @ g
    return g - k @ ((product + product.conj().T) / 2)


def cayley_step(k: np.ndarray, g: np.ndarray, step: float) -> np.ndarray:
    """Paper Eq. (3): K - eta A (I+eta B^dagger A/2)^-1 B^dagger K.

    A=[G,K], B=[K,-G]. A small 2d by 2d solve replaces an rd by rd solve.
    Caller chooses whether to normalise G. This is a feasible Cayley descent
    curve; its initial velocity is -(G-K G^dagger K).
    """
    a = np.concatenate((g, k), axis=1)
    b = np.concatenate((k, -g), axis=1)
    return k - step * a @ np.linalg.solve(np.eye(a.shape[1]) + step / 2 * b.conj().T @ a,
                                         b.conj().T @ k)


def random_isometry(d: int, rank: int, seed: int = 1, near_identity: bool = True) -> np.ndarray:
    if d < 2 or not 1 <= rank <= d * d:
        raise ValueError("Require d>=2 and 1<=rank<=d^2")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rank * d, d)) + 1j * rng.normal(size=(rank * d, d))
    if near_identity:
        a *= .12 / np.sqrt(rank * d)
        a[:d] += np.eye(d)
    return polar(a)

