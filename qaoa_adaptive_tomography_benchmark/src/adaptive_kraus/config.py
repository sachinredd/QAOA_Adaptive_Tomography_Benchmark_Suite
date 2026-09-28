"""Strict JSON configuration with conservative, reproducible defaults."""
from copy import deepcopy
import json
import numpy as np

DEFAULT = {
    "seeds": [11, 29, 47],
    "policies": ["random", "greedy", "exact_qubo", "random_qubo", "qaoa"],
    "channel": {"n_qubits": 1, "dt": 1., "omega": [.25, -.16, .32], "rates": [.12, .035, .045], "zz": 0.},
    "rank": 4, "pool_steps": [1, 2, 4, 6, 8], "validation_steps": [3, 5, 9],
    "validation_probes": 8, "readout": [0., 0.], "pilot_shots": 64,
    "applications_per_setting": 0, "round_shots": 128, "rounds": 5, "batch_size": 3, "shortlist_size": 10,
    "ridge": 1., "fisher_probability_floor": 1e-6,
    "fit": {"iterations": 180, "learning_rate": 1., "tolerance": 1e-7},
    "qaoa": {"depth": 2, "maxiter": 40, "shots": 256, "sample_shots": 256,
             "restarts": 1, "one_qubit_noise": 0., "two_qubit_noise": 0.},
    "random_qubo_samples": 256, "audit_exact_gap": True,
    "target_rmse": .02
}


def merge_config(update, base=None):
    """Recursively apply known keys only; catch misspelled configuration options."""
    result = deepcopy(DEFAULT if base is None else base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected object for {key}")
            result[key] = merge_config(value, result[key])
        else:
            result[key] = value
    return result


def validate_config(config):
    """Reject infeasible budgets, unsupported dimensions, and time-set leakage."""
    c = config
    if not np.all(np.isfinite([c["ridge"], c["fisher_probability_floor"], c["target_rmse"],
            c["fit"]["learning_rate"], c["fit"]["tolerance"], *c["readout"],
            c["qaoa"]["one_qubit_noise"], c["qaoa"]["two_qubit_noise"]])):
        raise ValueError("All real-valued options must be finite")
    nq = c["channel"]["n_qubits"]
    if nq not in (1, 2) or not 1 <= c["rank"] <= 4**nq:
        raise ValueError("Use one/two system qubits and rank between 1 and d^2")
    if not c["seeds"] or any(int(x) != x or x < 0 for x in c["seeds"]) or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Use distinct nonnegative integer seeds")
    allowed = {"random", "greedy", "exact_qubo", "random_qubo", "qaoa"}
    if not c["policies"] or not set(c["policies"]) <= allowed or len(set(c["policies"])) != len(c["policies"]):
        raise ValueError("Unknown, duplicate or missing policies")
    for key in ("rank", "pilot_shots", "round_shots", "rounds", "batch_size", "shortlist_size", "validation_probes", "random_qubo_samples"):
        if int(c[key]) != c[key] or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("pool_steps", "validation_steps"):
        if not c[key] or len(set(c[key])) != len(c[key]) or any(int(t) != t or t < 1 for t in c[key]):
            raise ValueError(f"{key} must contain distinct positive integers")
    cap = c.get("applications_per_setting", 0)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("applications_per_setting must be a nonnegative integer")
    if cap and any(cap % t for t in c["pool_steps"]):
        raise ValueError("Application budget must be divisible by every candidate step")
    if min(c["pool_steps"]) != 1:
        raise ValueError("The IC pilot must include the one-step channel")
    if set(c["pool_steps"]) & set(c["validation_steps"]):
        raise ValueError("Held-out times must be absent from the candidate pool")
    size = 18**nq*len(c["pool_steps"])
    if not c["batch_size"] <= c["shortlist_size"] <= size:
        raise ValueError("Require batch <= shortlist <= candidate count")
    if c["ridge"] <= 0 or not 0 < c["fisher_probability_floor"] < 1 or c["target_rmse"] <= 0:
        raise ValueError("Invalid information regularisation or target")
    q = c["qaoa"]
    if any(x < 0 or x > 1 for x in (q["one_qubit_noise"], q["two_qubit_noise"])):
        raise ValueError("QAOA noise strengths must be probabilities")
    for key in ("depth", "maxiter", "sample_shots", "restarts"):
        if int(q[key]) != q[key] or q[key] < 1:
            raise ValueError(f"qaoa.{key} must be a positive integer")
    if int(q["shots"]) != q["shots"] or q["shots"] < 0:
        raise ValueError("qaoa.shots must be a nonnegative integer")
    if "qaoa" in c["policies"] and c["shortlist_size"] > (8 if q["one_qubit_noise"] or q["two_qubit_noise"] else 14):
        raise ValueError("QAOA simulator guard: 14 ideal or 8 noisy register qubits")
    if len(c["readout"]) != 2 or any(not 0 <= x <= 1 for x in c["readout"]):
        raise ValueError("Invalid readout probabilities")
    if c["fit"]["iterations"] < 1 or int(c["fit"]["iterations"]) != c["fit"]["iterations"] or c["fit"]["learning_rate"] <= 0 or c["fit"]["tolerance"] < 0:
        raise ValueError("Invalid fitting options")
    # Full Fisher blocks scale as candidate_count * (d^4-d^2)^2.
    p = 16**nq-4**nq
    if size*p*p*8*3 > 2_000_000_000:
        raise ValueError("Design-memory guard exceeded; reduce the number of times")
    return c


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return validate_config(merge_config(json.load(stream)))
