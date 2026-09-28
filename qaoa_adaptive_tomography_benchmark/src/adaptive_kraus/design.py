"""Gauge-independent local Fisher information and a batch selection surrogate."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .channels import apply_channel
from .experiments import pauli_basis


def pauli_transfer(kraus: np.ndarray) -> np.ndarray:
    """T_ab=Tr[P_a E(P_b)]/d. The identity row is fixed for a TP channel."""
    d = kraus.shape[1]
    basis = pauli_basis(int(np.log2(d)))
    evolved = apply_channel(kraus, basis)
    return np.einsum("aij,bji->ab", basis, evolved).real/d


def transfer_predictions_jacobian(transfer: np.ndarray, pool):
    """Differentiate probabilities w.r.t. all T[a,b], a>0, in row-major order.

    These d^4-d^2 real coordinates describe trace-preserving Hermiticity-
    preserving perturbations. They remove Kraus gauge redundancy, but the
    design chart itself does not enforce CP near the channel boundary.
    The learned channel used to evaluate this chart remains CPTP.
    """
    m = transfer.shape[0]
    d = int(np.sqrt(m))
    basis = pauli_basis(int(np.log2(d)))
    states = np.einsum("aij,sji->sa", basis, np.asarray([e.rho for e in pool])).real
    effect_coeff = np.einsum("soij,aji->soa", np.asarray([e.effects for e in pool]), basis).real/d
    aa = np.repeat(np.arange(1, m), m)
    bb = np.tile(np.arange(m), m-1)
    tangent = np.zeros((len(pool), m, len(aa)))
    p = np.zeros((len(pool), d))
    jac = np.zeros((len(pool), d, len(aa)))
    steps = np.asarray([e.steps for e in pool])
    for t in range(1, int(steps.max())+1):
        source = np.zeros_like(tangent)
        source[:, aa, np.arange(len(aa))] = states[:, bb]
        tangent = np.einsum("ab,sbp->sap", transfer, tangent)+source
        states = np.einsum("ab,sb->sa", transfer, states)
        mask = steps == t
        p[mask] = np.einsum("soa,sa->so", effect_coeff[mask], states[mask])
        jac[mask] = np.einsum("soa,sap->sop", effect_coeff[mask], tangent[mask])
    return p, jac


def fisher_information(probabilities, jacobian, probability_floor=1e-6):
    """Expected multinomial Fisher information per shot, including all outcomes."""
    if not 0 < probability_floor < 1:
        raise ValueError("Fisher probability floor must lie in (0,1)")
    weights = 1/np.maximum(probabilities, probability_floor)
    return np.einsum("sop,so,soq->spq", jacobian, weights, jacobian, optimize=True)


def logdet_spd(matrix: np.ndarray) -> float:
    """Cholesky log determinant; fail rather than silently accept an indefinite F."""
    lower = np.linalg.cholesky((matrix+matrix.T)/2)
    return float(2*np.log(np.diag(lower)).sum())


@dataclass
class DesignState:
    probabilities: np.ndarray
    information: np.ndarray
    accumulated: np.ndarray
    whitened: np.ndarray
    singleton_values: np.ndarray
    condition_number: float


def build_design(kraus, pool, observations, shots_per_experiment, ridge=1., probability_floor=1e-6):
    """Recompute past and prospective information at the current learned map.

    ridge is an isotropic regularisation in the fixed PTM chart, not a fitted
    posterior or a confidence interval. Candidate blocks already include the
    acquisition shot count. No true model or held-out outcomes are accepted.
    """
    shots = np.asarray(shots_per_experiment)
    if (shots.ndim not in (0, 1) or (shots.ndim == 1 and shots.shape != (len(pool),))
            or not np.isfinite(shots).all() or np.any(shots < 1) or np.any(shots != np.floor(shots))):
        raise ValueError("Shot counts must be positive integers, scalar or one per candidate")
    if ridge <= 0:
        raise ValueError("Positive ridge and shot count are required")
    p, jac = transfer_predictions_jacobian(pauli_transfer(kraus), pool)
    per_shot = fisher_information(p, jac, probability_floor)
    accumulated = ridge*np.eye(jac.shape[-1])
    for obs in observations:
        accumulated += obs.shots*per_shot[obs.experiment_id]
    vals, vecs = np.linalg.eigh(accumulated)
    root_inv = (vecs/np.sqrt(vals))@vecs.T
    info = (shots if shots.ndim == 0 else shots[:, None, None])*per_shot
    whitened = np.einsum("ab,sbc,cd->sad", root_inv, info, root_inv, optimize=True)
    eye = np.eye(jac.shape[-1])
    values = np.array([logdet_spd(eye+a) for a in whitened])
    return DesignState(p, info, accumulated, whitened, values, float(vals[-1]/vals[0]))


def batch_gain(design: DesignState, selected) -> float:
    """Exact local log-det gain for the selected batch, relative to current F."""
    return logdet_spd(np.eye(design.whitened.shape[-1])+design.whitened[list(selected)].sum(axis=0))


def greedy_batch(design: DesignState, batch_size: int) -> list[int]:
    """Sequentially maximise true local log-det gain across the full candidate pool."""
    chosen = []
    remaining = list(range(len(design.whitened)))
    for _ in range(batch_size):
        pick = max(remaining, key=lambda i: batch_gain(design, chosen+[i]))
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


@dataclass
class BinaryDesign:
    """E(q)=linear.q + sum_(i<j) pairs_ij q_i q_j + constant.

    For an unpenalised design E=-surrogate_gain. The QAOA register has one
    qubit per shortlisted setting, unrelated to system qubits or Kraus rank.
    """
    linear: np.ndarray
    pairs: np.ndarray
    batch_size: int
    experiment_ids: np.ndarray
    constant: float = 0.

    @property
    def n_bits(self):
        return len(self.linear)

    def energy(self, q):
        q = np.asarray(q)
        return q@self.linear+np.einsum("...i,ij,...j->...", q, self.pairs, q)+self.constant

    def ising(self):
        h = -self.linear/2-(self.pairs.sum(0)+self.pairs.sum(1))/4
        return h, self.pairs/4, float(self.constant+self.linear.sum()/2+self.pairs.sum()/4)

    def with_cardinality_penalty(self, strength=None):
        """Equivalent unconstrained QUBO with a sufficient feasibility penalty."""
        if strength is None:
            strength = float(np.abs(self.linear).sum()+np.abs(self.pairs).sum()+1.)
        if strength <= 0:
            raise ValueError("Penalty strength must be positive")
        return BinaryDesign(self.linear+strength*(1-2*self.batch_size),
            self.pairs+2*strength*np.triu(np.ones_like(self.pairs), 1), self.batch_size,
            self.experiment_ids.copy(), self.constant+strength*self.batch_size**2)


def make_binary_design(design: DesignState, shortlist_size: int, batch_size: int) -> BinaryDesign:
    """Match log-det gains exactly for singleton and pair selections.

    r_ij=v_i+v_j-gain({i,j}) measures redundant information. Higher-order
    interactions are omitted, so batches of three or more are approximate.
    Shortlisting uses only the current model, with stable ID tie breaking.
    """
    if not 1 <= batch_size <= shortlist_size <= len(design.whitened):
        raise ValueError("Require batch <= shortlist <= pool size")
    ids = np.argsort(-design.singleton_values, kind="stable")[:shortlist_size]
    values = design.singleton_values[ids]
    pairs = np.zeros((shortlist_size, shortlist_size))
    for i, j in combinations(range(shortlist_size), 2):
        pairs[i, j] = max(0., values[i]+values[j]-batch_gain(design, [ids[i], ids[j]]))
    return BinaryDesign(-values, pairs, batch_size, ids)


def feasible_bits(n_bits: int, batch_size: int) -> np.ndarray:
    """Enumerate only fixed-cardinality subsets for the explicit exact baseline."""
    from math import comb
    if not 0 <= batch_size <= n_bits or comb(n_bits, batch_size) > 200000:
        raise ValueError("Exact enumeration guard exceeded or invalid cardinality")
    out = np.zeros((comb(n_bits, batch_size), n_bits))
    for row, subset in enumerate(combinations(range(n_bits), batch_size)):
        out[row, list(subset)] = 1.
    return out


def classical_select(model: BinaryDesign, method: str, rng, samples=256):
    """Exact surrogate minimisation or best of uniform feasible random proposals."""
    if method == "exact_qubo":
        candidates = feasible_bits(model.n_bits, model.batch_size)
    elif method == "random_qubo":
        if samples < 1:
            raise ValueError("Positive random sample count required")
        candidates = np.zeros((samples, model.n_bits))
        for row in candidates:
            row[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    else:
        raise ValueError("Unknown classical selector")
    index = int(np.argmin(model.energy(candidates)))
    return candidates[index], dict(solver=method, candidate_evaluations=len(candidates),
        objective_shots=0, sample_shots=0, circuit_executions=0)
