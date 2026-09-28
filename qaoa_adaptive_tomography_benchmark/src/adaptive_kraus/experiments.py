"""Known experimental settings, count-only observations and an isolated Aer oracle."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from .channels import I, X, Y, Z, apply_channel


@dataclass(frozen=True)
class Experiment:
    """One full measurement setting; axes[0] addresses Qiskit qubit zero."""
    id: int
    preparation: str
    rho: np.ndarray
    axes: tuple[str, ...]
    steps: int
    effects: np.ndarray

    @property
    def label(self):
        return f"{self.preparation}|{''.join(reversed(self.axes))}|t={self.steps}"


@dataclass(frozen=True)
class Observation:
    """One actual acquisition. No reference probabilities or channel are stored."""
    experiment_id: int
    counts: tuple[int, ...]
    shots: int

    def __post_init__(self):
        if (self.shots < 1 or int(self.shots) != self.shots or not self.counts
                or any(c < 0 or int(c) != c for c in self.counts)
                or sum(self.counts) != self.shots):
            raise ValueError("Counts must be nonnegative integers summing to positive shots")


def pauli_basis(n_qubits: int) -> np.ndarray:
    """Unnormalised tensor Paulis, identity first; Tr(P_a P_b)=d delta_ab."""
    out = []
    for factors in product((I, X, Y, Z), repeat=n_qubits):
        p = np.ones((1, 1), complex)
        for factor in factors:
            p = np.kron(p, factor)
        out.append(p)
    return np.asarray(out)


def product_probes(n_qubits: int, pilot: bool = False):
    """Six eigenstates per qubit, or the four-state IC pilot set X+,Y+,Z+,Z-."""
    one = [(axis+sign, (I+s*p)/2) for axis, p in zip("XYZ", (X, Y, Z))
           for sign, s in (("+", 1), ("-", -1))]
    if pilot:
        one = [x for x in one if x[0] in ("X+", "Y+", "Z+", "Z-")]
    out = []
    for factors in product(one, repeat=n_qubits):
        rho = np.ones((1, 1), complex)
        for _, state in factors:
            rho = np.kron(rho, state)
        out.append(("".join(label for label, _ in factors), rho))
    return out


def confusion_matrix(n_qubits: int, readout=(0., 0.)) -> np.ndarray:
    """Column x contains P(reported outcome y | ideal outcome x)."""
    if len(readout) != 2 or not np.all(np.isfinite(readout)) or any(x < 0 or x > 1 for x in readout):
        raise ValueError("readout requires two finite probabilities")
    r01, r10 = readout
    one = np.array([[1-r01, r10], [r01, 1-r10]])
    result = np.ones((1, 1))
    for _ in range(n_qubits):
        result = np.kron(result, one)
    return result


def measurement_effects(axes, readout=(0., 0.)) -> np.ndarray:
    """Full POVM including a known, independent assignment-error calibration."""
    paul = dict(zip("XYZ", (X, Y, Z)))
    effects = []
    for outcome in range(2**len(axes)):
        m = np.ones((1, 1), complex)
        for q in reversed(range(len(axes))):
            m = np.kron(m, (I+(1-2*((outcome >> q) & 1))*paul[axes[q]])/2)
        effects.append(m)
    return np.einsum("yx,xab->yab", confusion_matrix(len(axes), readout), effects)


def make_pool(n_qubits: int = 1, steps=(1, 2, 4, 6, 8), readout=(0., 0.)):
    """Stable contiguous IDs for all product-Pauli probes, times and bases."""
    if n_qubits not in (1, 2) or not steps or len(set(steps)) != len(steps) or any(t < 1 or int(t) != t for t in steps):
        raise ValueError("Use one/two qubits and unique positive integer steps")
    pool = []
    for label, rho in product_probes(n_qubits):
        for step in steps:
            for axes in product("XYZ", repeat=n_qubits):
                pool.append(Experiment(len(pool), label, rho, axes, int(step), measurement_effects(axes, readout)))
    return pool


def pilot_ids(pool: list[Experiment]) -> list[int]:
    """All bases for an IC preparation set at the shortest candidate time."""
    nq = len(pool[0].axes)
    labels = {x[0] for x in product_probes(nq, pilot=True)}
    step = min(e.steps for e in pool)
    return [e.id for e in pool if e.preparation in labels and e.steps == step]


def heldout_pool(n_qubits=1, steps=(3, 5, 9), probes=8, seed=917, readout=(0., 0.)):
    """Seeded random pure states and separate integer times for scoring only."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(probes):
        v = rng.normal(size=2**n_qubits)+1j*rng.normal(size=2**n_qubits)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        for step in steps:
            for axes in product("XYZ", repeat=n_qubits):
                pool.append(Experiment(len(pool), f"heldout_{i}", rho, axes, step, measurement_effects(axes, readout)))
    return pool


def predict_probabilities(kraus: np.ndarray, experiments: list[Experiment]) -> np.ndarray:
    """Predict complete outcome distributions; no observed counts are needed."""
    states = np.asarray([e.rho for e in experiments])
    trajectory = [states]
    for _ in range(max(e.steps for e in experiments)):
        trajectory.append(apply_channel(kraus, trajectory[-1]))
    selected = np.asarray(trajectory)[[e.steps for e in experiments], np.arange(len(experiments))]
    return np.einsum("soij,sji->so", np.asarray([e.effects for e in experiments]), selected).real


def aggregate_counts(observations: list[Observation], pool: list[Experiment]):
    """Merge repeated settings, retaining shot weights exactly."""
    merged = {}
    for obs in observations:
        if obs.experiment_id < 0 or obs.experiment_id >= len(pool):
            raise ValueError("Observation refers to an unknown experiment")
        if len(obs.counts) != len(pool[obs.experiment_id].effects):
            raise ValueError("Outcome count disagrees with measurement dimension")
        merged.setdefault(obs.experiment_id, np.zeros(len(obs.counts), dtype=np.int64))
        merged[obs.experiment_id] += np.asarray(obs.counts, dtype=np.int64)
    ids = sorted(merged)
    if not ids:
        raise ValueError("At least one acquisition is required")
    return [pool[i] for i in ids], np.stack([merged[i] for i in ids])


class AerOracle:
    """Only measure() is consumed by acquisition. Truth stays inside this object.

    State preparation and measurement rotations are ideal. Kraus noise is
    explicitly inserted at each time step. Seeds depend on setting and visit,
    not policy, providing common random numbers across matched acquisitions.
    """
    def __init__(self, kraus, readout=(0., 0.), seed=1):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, kraus_error
        self.n_qubits = int(np.log2(kraus.shape[1]))
        self.seed = int(seed)
        self.visits = {}
        self.circuit_executions = 0
        self.measurement_shots = 0
        self.channel_applications = 0
        self._instruction = kraus_error(list(kraus)).to_instruction()
        noise = NoiseModel()
        if any(readout):
            noise.add_all_qubit_readout_error(ReadoutError(confusion_matrix(1, readout).T))
        self._backend = AerSimulator(method="density_matrix", noise_model=noise, max_parallel_threads=1)

    def measure(self, experiments: list[Experiment], shots: int) -> list[Observation]:
        from qiskit import QuantumCircuit
        if shots < 1 or int(shots) != shots:
            raise ValueError("Acquisition requires positive integer shots")
        out = []
        for experiment in experiments:
            circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
            circuit.set_density_matrix(experiment.rho)
            for _ in range(experiment.steps):
                circuit.append(self._instruction, range(self.n_qubits))
            for q, axis in enumerate(experiment.axes):
                if axis == "Y":
                    circuit.sdg(q)
                if axis in ("X", "Y"):
                    circuit.h(q)
            circuit.measure(range(self.n_qubits), range(self.n_qubits))
            visit = self.visits.get(experiment.id, 0)
            run_seed = int(np.random.SeedSequence([self.seed, experiment.id, visit]).generate_state(1)[0])
            result = self._backend.run(circuit, shots=int(shots), seed_simulator=run_seed).result()
            if not result.success:
                raise RuntimeError(f"Aer acquisition failed: {result.status}")
            counts = result.get_counts()
            values = tuple(int(counts.get(format(i, f"0{self.n_qubits}b"), 0)) for i in range(2**self.n_qubits))
            out.append(Observation(experiment.id, values, int(shots)))
            self.visits[experiment.id] = visit+1
            self.circuit_executions += 1
            self.measurement_shots += int(shots)
            self.channel_applications += int(shots)*experiment.steps
        return out
