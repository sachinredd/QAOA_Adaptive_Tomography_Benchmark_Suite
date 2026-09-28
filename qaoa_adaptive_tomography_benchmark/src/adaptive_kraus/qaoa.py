"""Fixed-cardinality QAOA with explicit Aer cost and XY mixer circuits."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .design import BinaryDesign


def integer_bits(value: int, n: int) -> np.ndarray:
    """Little-endian coordinate bits corresponding to Qiskit register qubits."""
    return ((int(value) >> np.arange(n)) & 1).astype(float)


def build_circuit(model: BinaryDesign, depth: int, initial_bits):
    """Cost RZ/RZZ layers and paired RXX/RYY gates on a connected chain.

    RXX(2 beta) RYY(2 beta)=exp[-i beta(XX+YY)] preserves Hamming weight
    ideally. The sequential chain is an alternating-operator mixer, not an
    exact exponential of the sum of all overlapping edge Hamiltonians.
    Initialisation uses X gates and two fixed XY sweeps to create a feasible
    superposition. This makes the first cost layer nontrivial.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    if depth < 1 or len(initial_bits) != model.n_bits or sum(initial_bits) != model.batch_size:
        raise ValueError("Invalid depth or initial feasible bitstring")
    h, j, _ = model.ising()
    scale = max(float(np.max(abs(h))), float(np.max(abs(j))), 1e-12)
    h, j = h/scale, j/scale
    angles = ParameterVector("theta", 2*depth)
    qc = QuantumCircuit(model.n_bits)
    for i, bit in enumerate(initial_bits):
        if bit:
            qc.x(i)
    for _ in range(2):
        for parity in (0, 1):
            for i in range(parity, model.n_bits-1, 2):
                qc.rxx(np.pi/4, i, i+1)
                qc.ryy(np.pi/4, i, i+1)
    for layer in range(depth):
        gamma, beta = angles[layer], angles[depth+layer]
        for i in range(model.n_bits):
            if abs(h[i]) > 1e-14:
                qc.rz(2*gamma*h[i], i)
            for other in range(i+1, model.n_bits):
                if abs(j[i, other]) > 1e-14:
                    qc.rzz(2*gamma*j[i, other], i, other)
        for parity in (0, 1):
            for i in range(parity, model.n_bits-1, 2):
                qc.rxx(2*beta, i, i+1)
                qc.ryy(2*beta, i, i+1)
    return qc, list(angles), scale


def noise_model(one_qubit=0., two_qubit=0.):
    """Optional depolarisation on the explicitly simulated primitive gates."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in (one_qubit, two_qubit)):
        raise ValueError("Noise strengths must be probabilities")
    noise = NoiseModel()
    if one_qubit:
        noise.add_all_qubit_quantum_error(depolarizing_error(one_qubit, 1), ["x", "rz"])
    if two_qubit:
        noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit, 2), ["rzz", "rxx", "ryy"])
    return noise


def solve_qaoa(model: BinaryDesign, depth=2, maxiter=45, shots=256, sample_shots=256,
               restarts=1, seed=1, one_qubit_noise=0., two_qubit_noise=0.):
    """Minimise conditional mean energy among feasible outcomes, then sample.

    shots=0 is a simulator-only exact-probability objective. All final proposals
    come from actual Aer measurement counts. In noisy runs, postselection
    rejects invalid cardinalities and all attempted shots are charged. If no
    feasible final sample exists, use the initial bitstring and report it.
    No exact solution, greedy solution or local repair replaces QAOA output.
    """
    from qiskit_aer import AerSimulator
    noisy = bool(one_qubit_noise or two_qubit_noise)
    if (depth < 1 or maxiter < 1 or shots < 0 or sample_shots < 1 or restarts < 1
            or model.n_bits > (8 if noisy else 14)):
        raise ValueError("Invalid QAOA budget; maximum 14 ideal or 8 noisy qubits")
    rng = np.random.default_rng(seed)
    initial = np.zeros(model.n_bits)
    initial[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    qc, parameters, scale = build_circuit(model, depth, initial)
    exact_circuit = qc.copy()
    exact_circuit.save_probabilities(label="probabilities")
    measured = qc.copy()
    measured.measure_all()
    backend = AerSimulator(method="density_matrix" if noisy else "statevector",
        noise_model=noise_model(one_qubit_noise, two_qubit_noise), max_parallel_threads=1)
    # This lookup scores possible outcomes; it never chooses an optimum.
    # Exact-probability mode needs full support. Finite-shot mode decodes only
    # observed strings, so it does not enumerate or score unsampled subsets.
    lookup = None
    if shots == 0:
        support = np.asarray([integer_bits(i, model.n_bits) for i in range(2**model.n_bits)])
        valid = support.sum(1) == model.batch_size
        lookup = (valid, model.energy(support))
    evaluations = 0
    feasible_objective_shots = 0
    best_value, best_angles = np.inf, None
    offset = model.ising()[2]
    empty_cost = (np.abs(model.linear).sum()+np.abs(model.pairs).sum()+1.)/scale

    def evaluate(angles):
        nonlocal evaluations, feasible_objective_shots, best_value, best_angles
        template = measured if shots else exact_circuit
        bound = template.assign_parameters(dict(zip(parameters, angles)))
        result = backend.run(bound, shots=max(1, shots), seed_simulator=seed+evaluations).result()
        if not result.success:
            raise RuntimeError(f"QAOA failed: {result.status}")
        if shots:
            count_total, weighted = 0, 0.
            for key, count in result.get_counts().items():
                q = integer_bits(int(key.replace(" ", ""), 2), model.n_bits)
                if int(q.sum()) == model.batch_size:
                    count_total += count
                    weighted += count*(float(model.energy(q))-offset)/scale
            feasible_objective_shots += count_total
            value = weighted/count_total if count_total else empty_cost
        else:
            probabilities = np.asarray(result.data(0)["probabilities"])
            valid, energy = lookup
            mass = float(probabilities[valid].sum())
            value = float(probabilities[valid]@(energy[valid]-offset)/scale/mass) if mass > 1e-15 else empty_cost
        evaluations += 1
        if value < best_value:
            best_value, best_angles = value, np.asarray(angles).copy()
        return value

    for _ in range(restarts):
        angles = np.r_[rng.uniform(0., 1., depth), rng.uniform(.1, 1.2, depth)]
        minimize(evaluate, angles, method="COBYLA", options={"maxiter": maxiter, "rhobeg": .35, "tol": 1e-3})
    result = backend.run(measured.assign_parameters(dict(zip(parameters, best_angles))),
        shots=sample_shots, seed_simulator=seed+evaluations+1).result()
    if not result.success:
        raise RuntimeError(f"QAOA sampling failed: {result.status}")
    candidates, feasible_count = [], 0
    for key, count in result.get_counts().items():
        q = integer_bits(int(key.replace(" ", ""), 2), model.n_bits)
        if int(q.sum()) == model.batch_size:
            candidates.append(q)
            feasible_count += count
    fallback = not candidates
    if fallback:
        candidates = [initial]
    chosen = min(candidates, key=lambda q: float(model.energy(q)))
    return chosen, dict(solver="qaoa", n_qubits=model.n_bits, depth=depth,
        circuit_executions=evaluations+1, objective_evaluations=evaluations,
        objective_shots=evaluations*shots, sample_shots=sample_shots,
        feasible_objective_shots=feasible_objective_shots if shots else None,
        feasible_final_shots=feasible_count, distinct_feasible_samples=len(candidates) if not fallback else 0,
        initial_fallback=fallback, initial_bits=initial.astype(int).tolist(), angles=best_angles.tolist(),
        best_conditional_scaled_cost=float(best_value), circuit_depth=qc.depth(),
        gate_counts=dict(qc.count_ops()), exact_objective=(shots == 0),
        one_qubit_noise=one_qubit_noise, two_qubit_noise=two_qubit_noise)
