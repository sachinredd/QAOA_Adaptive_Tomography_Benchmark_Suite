"""Adaptive acquisition, separate truth-based scoring, and paired benchmarks."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import importlib.metadata
import json
import platform
import numpy as np
from .cost_budget import prospective_shots, measure_selected
from .channels import gksl_channel, blocks, channel_metrics
from .manifold import random_isometry
from .experiments import (AerOracle, Observation, make_pool, pilot_ids, heldout_pool,
                          predict_probabilities)
from .learning import fit_channel
from .design import build_design, greedy_batch, make_binary_design, classical_select, batch_gain
from .qaoa import solve_qaoa
from .config import validate_config


def write_json(path, value):
    """Atomically replace a JSON checkpoint after serialization has succeeded."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def choose_batch(policy, stack, pool, observations, config, seed):
    """Select using observed counts and a learned channel only; no oracle input."""
    started = perf_counter()
    rng = np.random.default_rng(seed)
    batch = config["batch_size"]
    stats = dict(solver=policy, objective_shots=0, sample_shots=0, circuit_executions=0)
    model = None
    if policy == "random":
        selected = rng.choice(len(pool), batch, replace=False).tolist()
    else:
        design = build_design(blocks(stack), pool, observations, prospective_shots(pool, config),
                              config["ridge"], config["fisher_probability_floor"])
        stats["information_condition_number"] = design.condition_number
        if policy == "greedy":
            selected = greedy_batch(design, batch)
        else:
            model = make_binary_design(design, config["shortlist_size"], batch)
            if policy == "qaoa":
                q, solver_stats = solve_qaoa(model, seed=seed, **config["qaoa"])
            else:
                q, solver_stats = classical_select(model, policy, rng, config["random_qubo_samples"])
            stats.update(solver_stats)
            selected = model.experiment_ids[np.flatnonzero(q)].tolist()
            stats.update(shortlist_ids=model.experiment_ids.tolist(), selected_bits=q.astype(int).tolist(),
                         surrogate_gain=-float(model.energy(q)))
        stats["local_logdet_gain"] = batch_gain(design, selected)
    stats["selection_seconds"] = perf_counter()-started
    # Explicit audit after the selection. Its output cannot alter the choice.
    stats["audit_seconds"] = 0.
    if model is not None and config["audit_exact_gap"]:
        audit_start = perf_counter()
        best, _ = classical_select(model, "exact_qubo", np.random.default_rng(0))
        stats["surrogate_optimality_gap"] = float(-stats["surrogate_gain"]-model.energy(best))
        stats["exact_surrogate_best_gain"] = -float(model.energy(best))
        stats["audit_seconds"] = perf_counter()-audit_start
    return selected, stats, model


def run_policy(policy, pool, oracle, initial, config, seed, output):
    """Acquire, refit and checkpoint. Reference-channel scoring occurs elsewhere."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    observations = []
    history = []
    stack = initial.copy()
    np.savez_compressed(output/"initial.npz", stack=initial)
    cumulative_fit = cumulative_design = cumulative_acquisition = cumulative_audit = 0.
    objective_shots = sample_shots = optimizer_circuits = 0
    for round_index in range(config["rounds"]+1):
        model = None
        if round_index == 0:
            selected = pilot_ids(pool)
            shots = config["pilot_shots"]
            selection = dict(solver="common_pilot", selection_seconds=0., audit_seconds=0.,
                             objective_shots=0, sample_shots=0, circuit_executions=0)
        else:
            selected, selection, model = choose_batch(policy, stack, pool, observations,
                                                       config, seed+10007*round_index)
            shots = prospective_shots(pool, config)
        started = perf_counter()
        observations.extend(measure_selected(oracle, pool, selected, shots))
        cumulative_acquisition += perf_counter()-started
        fit = fit_channel(pool, observations, stack, **config["fit"])
        stack = fit.stack
        cumulative_fit += fit.diagnostics["seconds"]
        cumulative_design += selection["selection_seconds"]
        cumulative_audit += selection["audit_seconds"]
        objective_shots += selection["objective_shots"]
        sample_shots += selection["sample_shots"]
        optimizer_circuits += selection["circuit_executions"]
        row = dict(round=round_index, selected_ids=selected,
            selected_labels=[pool[i].label for i in selected], selection=selection, fit=fit.diagnostics,
            characterization_shots=oracle.measurement_shots,
            characterization_circuits=oracle.circuit_executions,
            channel_applications=oracle.channel_applications,
            optimizer_objective_shots=objective_shots, optimizer_sample_shots=sample_shots,
            optimizer_circuits=optimizer_circuits,
            total_attempted_shots=oracle.measurement_shots+objective_shots+sample_shots,
            fit_seconds=cumulative_fit, selection_seconds=cumulative_design,
            acquisition_seconds=cumulative_acquisition, audit_seconds=cumulative_audit,
            algorithm_seconds=cumulative_fit+cumulative_design+cumulative_acquisition)
        history.append(row)
        np.savez_compressed(output/f"model_round_{round_index:03d}.npz", stack=stack)
        write_json(output/f"fit_round_{round_index:03d}.json", fit.history)
        if model is not None:
            np.savez_compressed(output/f"design_round_{round_index:03d}.npz", linear=model.linear,
                pairs=model.pairs, experiment_ids=model.experiment_ids, batch_size=model.batch_size)
        write_json(output/"observations.json", [asdict(o) for o in observations])
        write_json(output/"history.json", history)
    return history


def score_history(history, output, reference, validation):
    """Add synthetic diagnostics after all adaptive decisions have completed."""
    truth = predict_probabilities(reference, validation)
    for row in history:
        with np.load(Path(output)/f"model_round_{row['round']:03d}.npz", allow_pickle=False) as saved:
            estimate = blocks(saved["stack"])
        prediction = predict_probabilities(estimate, validation)
        row["metrics"] = channel_metrics(estimate, reference)
        row["metrics"]["heldout_rmse"] = float(np.sqrt(np.mean((prediction-truth)**2)))
        row["metrics"]["heldout_max_absolute_error"] = float(np.max(np.abs(prediction-truth)))
    write_json(Path(output)/"history.json", history)
    return history


def environment_info():
    """Versions actually used by the local run, without freezing platform packages."""
    return dict(python=platform.python_version(), platform=platform.platform(), packages={
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "qiskit", "qiskit-aer", "matplotlib")})


def run_benchmark(config, output):
    """Paired seeds, equal acquisition budgets and a shared pilot for all policies."""
    config = validate_config(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    write_json(output/"environment.json", environment_info())
    reference = gksl_channel(**config["channel"])
    np.savez_compressed(output/"reference_for_scoring_only.npz", kraus=reference)
    nq = config["channel"]["n_qubits"]
    pool = make_pool(nq, config["pool_steps"], config["readout"])
    write_json(output/"pool.json", [dict(id=e.id, preparation=e.preparation, axes=list(e.axes),
                                       steps=e.steps, label=e.label) for e in pool])
    validation = heldout_pool(nq, config["validation_steps"], config["validation_probes"], readout=config["readout"])
    results = []
    for seed in config["seeds"]:
        initial = random_isometry(2**nq, config["rank"], seed+4001)
        for policy in config["policies"]:
            directory = output/f"seed_{seed}"/policy
            oracle = AerOracle(reference, config["readout"], seed)
            history = run_policy(policy, pool, oracle, initial, config, seed, directory)
            score_history(history, directory, reference, validation)
            final = history[-1]
            reached = next((r["characterization_shots"] for r in history if r["metrics"]["heldout_rmse"] <= config["target_rmse"]), None)
            result = dict(seed=seed, policy=policy, final=final, first_target_shots=reached)
            results.append(result)
            print(f"seed={seed} policy={policy} RMSE={final['metrics']['heldout_rmse']:.5f} "
                  f"fidelity={final['metrics']['process_fidelity_squared']:.6f} "
                  f"shots={final['characterization_shots']} total={final['total_attempted_shots']}", flush=True)
            write_json(output/"summary.json", results)
    from .reporting import plot_benchmark, write_summary_csv
    plot_benchmark(output)
    write_summary_csv(output)
    return results
