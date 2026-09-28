"""Column-vectorisation convention; qubit 0 is the rightmost tensor factor."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
DOWN = np.array([[0, 1], [0, 0]], dtype=complex)


def local_operator(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed an operator using Qiskit's little-endian qubit convention."""
    if not 0 <= qubit < n_qubits:
        raise ValueError("qubit must be in range")
    result = np.array([[1]], dtype=complex)
    for q in reversed(range(n_qubits)):
        result = np.kron(result, op if q == qubit else I)
    return result


def lindbladian(hamiltonian: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    """Return L such that vec_F(d rho/dt) = L vec_F(rho).

    Jump operators already include sqrt(rate). No Trotter approximation.
    """
    h = np.asarray(hamiltonian, complex)
    d = h.shape[0]
    eye = np.eye(d)
    if h.shape != (d, d) or not np.allclose(h, h.conj().T):
        raise ValueError("Hamiltonian must be square and Hermitian")
    out = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for l in jumps:
        a = l.conj().T @ l
        out += np.kron(l.conj(), l) - .5 * (np.kron(eye, a) + np.kron(a.T, eye))
    return out


def gksl_channel(dt: float = 1.0, omega=(.25, -.16, .32),
                 rates=(.12, .035, .045), n_qubits: int = 1,
                 zz: float = 0.0) -> np.ndarray:
    """Kraus operators for exp(dt L), with local down/up/Z dissipators.

    omega and rates can be length 3 or shape (n_qubits, 3). omega is in
    radians/time; rates are in 1/time. H has a factor 1/2, D[Z] does not.
    zz couples adjacent qubits via zz*Z_q Z_(q+1)/2.
    """
    from qiskit.quantum_info import Kraus, SuperOp
    if n_qubits < 1 or dt <= 0:
        raise ValueError("n_qubits and dt must be positive")
    omega = np.broadcast_to(np.asarray(omega, float), (n_qubits, 3))
    rates = np.broadcast_to(np.asarray(rates, float), (n_qubits, 3))
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Finite frequencies and nonnegative rates are required")
    d = 2 ** n_qubits
    h = np.zeros((d, d), complex)
    jumps = []
    for q in range(n_qubits):
        for w, op in zip(omega[q], (X, Y, Z)):
            h += .5 * w * local_operator(op, q, n_qubits)
        for rate, op in zip(rates[q], (DOWN, DOWN.conj().T, Z)):
            if rate > 0:
                jumps.append(np.sqrt(rate) * local_operator(op, q, n_qubits))
    for q in range(n_qubits - 1):
        h += .5 * zz * local_operator(Z, q, n_qubits) @ local_operator(Z, q + 1, n_qubits)
    channel = np.asarray(Kraus(SuperOp(expm(dt * lindbladian(h, jumps)))).data)
    return channel


def blocks(stack: np.ndarray) -> np.ndarray:
    """View an (r*d, d) Stiefel matrix as r Kraus operators."""
    d = stack.shape[1]
    return stack.reshape(-1, d, d)


def apply_channel(kraus: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply a Kraus channel to one density matrix or a leading batch."""
    return np.einsum("kab,...bc,kdc->...ad", kraus, rho, kraus.conj(), optimize=True)


def adjoint_channel(kraus: np.ndarray, observable: np.ndarray) -> np.ndarray:
    return np.einsum("kab,...ac,kcd->...bd", kraus.conj(), observable, kraus, optimize=True)


def superoperator(kraus: np.ndarray) -> np.ndarray:
    return sum(np.kron(k.conj(), k) for k in kraus)


def choi(kraus: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Input tensor output ordering; trace is d (or 1 if normalized)."""
    v = np.stack([k.reshape(-1, order="F") for k in kraus])
    j = v.T @ v.conj()
    return j / kraus.shape[1] if normalized else j


def cptp_diagnostics(kraus: np.ndarray) -> dict:
    d = kraus.shape[1]
    tp = sum(k.conj().T @ k for k in kraus)
    j = choi(kraus)
    return {"tp_error_fro": float(np.linalg.norm(tp - np.eye(d))),
            "choi_min_eigenvalue": float(np.linalg.eigvalsh(j).min()),
            "choi_trace": float(np.trace(j).real)}


def channel_metrics(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Gauge-invariant metrics: never compare individual Kraus matrices."""
    a, b = choi(estimated, True), choi(reference, True)
    vals, vecs = np.linalg.eigh(a)
    root = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.conj().T
    middle = root @ b @ root
    fidelity = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.conj().T) / 2), 0)).sum() ** 2
    out = cptp_diagnostics(estimated)
    out.update(process_fidelity_squared=float(np.clip(fidelity, 0, 1)),
               normalized_choi_trace_distance=float(.5 * np.abs(np.linalg.eigvalsh(a - b)).sum()),
               superoperator_relative_fro=float(np.linalg.norm(superoperator(estimated) - superoperator(reference))
                                               / np.linalg.norm(superoperator(reference))))
    return out
