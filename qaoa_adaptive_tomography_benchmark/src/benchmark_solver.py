"""Noiseless fixed-cardinality circuit simulation and sampled QAOA optimization.

The feasible-subspace engine implements the same RZ/RZZ and ordered RXX/RYY
circuit as adaptive_kraus.qaoa.build_circuit. Enumeration is a simulator
representation, not a candidate solution supplied to the optimizer. Final
proposals are restricted to sampled outcomes. Exact optima are audit inputs
only, computed outside solve(). There is no classical repair or replacement.
"""
from functools import lru_cache
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize


@lru_cache(None)
def basis(n, b):
    integers = np.array(sorted(sum(1 << j for j in c) for c in combinations(range(n), b)), dtype=np.int64)
    bits = ((integers[:, None] >> np.arange(n)) & 1).astype(float)
    index = {int(x): i for i, x in enumerate(integers)}
    edges = []
    for parity in (0, 1):
        for j in range(parity, n-1, 2):
            a = np.where((bits[:, j] == 0) & (bits[:, j+1] == 1))[0]
            c = np.array([index[int(integers[i]) ^ (1 << j) ^ (1 << (j+1))] for i in a])
            edges.append((a, c))
    return integers, bits, index, edges


class Circuit:
    def __init__(self, model, depth, initial_bits):
        self.model, self.depth = model, depth
        self.integers, self.bits, self.index, self.edges = basis(model.n_bits, model.batch_size)
        h, j, offset = model.ising()
        self.scale = max(float(abs(h).max()), float(abs(j).max()), 1e-12)
        self.energy = model.energy(self.bits)
        self.diagonal = (self.energy-offset)/self.scale
        self.initial = np.zeros(len(self.bits), complex)
        self.initial[self.index[sum(int(v) << i for i, v in enumerate(initial_bits))]] = 1
        for _ in range(2): self.mix(self.initial, np.pi/4)

    def mix(self, state, angle):
        c, s = np.cos(angle), -1j*np.sin(angle)
        for a, b in self.edges:
            x, y = state[a].copy(), state[b].copy()
            state[a], state[b] = c*x+s*y, s*x+c*y

    def probabilities(self, angles):
        state = self.initial.copy()
        for k in range(self.depth):
            state *= np.exp(-1j*angles[k]*self.diagonal)
            self.mix(state, 2*angles[self.depth+k])
        p = abs(state)**2
        return p/p.sum()


def solve(model, depth, maxiter, seed, warm_angles=None, shots=128, sample_shots=256):
    started = perf_counter()
    rng = np.random.default_rng(seed)
    initial = np.zeros(model.n_bits)
    initial[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    random_angles = np.r_[rng.uniform(0., 1., depth), rng.uniform(.1, 1.2, depth)]
    angles = random_angles if warm_angles is None else np.asarray(warm_angles, float)
    if angles.shape != (2*depth,): raise ValueError('Wrong transferred angle dimension')
    circuit = Circuit(model, depth, initial)
    best_value, best_angles, evaluations = np.inf, None, 0
    trace = []
    def objective(theta):
        nonlocal best_value, best_angles, evaluations
        p = circuit.probabilities(theta)
        draws = rng.choice(len(p), size=shots, p=p)
        value = float(circuit.diagonal[draws].mean())
        evaluations += 1
        trace.append({'angles': np.asarray(theta).tolist(), 'sampled_scaled_energy': value})
        if value < best_value:
            best_value, best_angles = value, np.asarray(theta).copy()
        return value
    result = minimize(objective, angles, method='COBYLA', options={'maxiter': maxiter, 'rhobeg': .35, 'tol': 1e-3})
    probabilities = circuit.probabilities(best_angles)
    final = rng.choice(len(probabilities), size=sample_shots, p=probabilities)
    selected = {}
    for count in (8, 32, sample_shots):
        candidates = np.unique(final[:count])
        winner = min(candidates, key=lambda i: (circuit.energy[i], i))
        selected[str(count)] = {'bits': circuit.bits[winner].astype(int).tolist(),
                               'energy': float(circuit.energy[winner])}
    return dict(selected=selected, angles=best_angles.tolist(), initial_bits=initial.astype(int).tolist(),
                start_angles=angles.tolist(), transferred=warm_angles is not None,
                objective_evaluations=evaluations, objective_shots=evaluations*shots,
                sample_shots=sample_shots, total_shots=evaluations*shots+sample_shots,
                expected_energy=float(probabilities@circuit.energy),
                final_sample_integers=circuit.integers[final].tolist(),
                trace=trace, probabilities=probabilities.tolist(),
                optimizer_status=int(result.status), seconds=perf_counter()-started)


def random_search(model, seed, count=256):
    started = perf_counter(); rng = np.random.default_rng(seed)
    qs = np.zeros((count, model.n_bits))
    for q in qs: q[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    energies = model.energy(qs)
    i = int(np.argmin(energies))
    return qs[i], dict(candidate_evaluations=count, energy=float(energies[i]), seconds=perf_counter()-started)


def swap_search(model, seed, budget=256):
    """First-improvement swaps with random restarts, capped by energy evaluations."""
    started = perf_counter(); rng = np.random.default_rng(seed)
    q = np.zeros(model.n_bits); q[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    current = float(model.energy(q)); best, value = q.copy(), current; used, restarts = 1, 0
    while used < budget:
        pairs = [(i, j) for i in np.flatnonzero(q) for j in np.flatnonzero(1-q)]
        rng.shuffle(pairs); improved = False
        for i, j in pairs:
            if used >= budget: break
            candidate = q.copy(); candidate[i], candidate[j] = 0, 1
            energy = float(model.energy(candidate)); used += 1
            if energy < value: best, value = candidate.copy(), energy
            if energy < current-1e-12:
                q, current, improved = candidate, energy, True
                break
        if not improved and used < budget:
            q[:] = 0; q[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
            current = float(model.energy(q)); used += 1; restarts += 1
            if current < value: best, value = q.copy(), current
    return best, dict(candidate_evaluations=used, restarts=restarts, energy=value, seconds=perf_counter()-started)
