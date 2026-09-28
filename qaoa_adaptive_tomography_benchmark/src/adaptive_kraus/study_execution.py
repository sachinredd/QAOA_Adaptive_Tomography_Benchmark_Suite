"""Paired adaptive trajectories with round-level deterministic restart."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import tempfile
import numpy as np
from .cost_budget import prospective_shots, measure_selected
from .channels import gksl_channel, blocks
from .experiments import AerOracle, Observation, make_pool, pilot_ids, heldout_pool
from .multistart import fit_candidates
from .manifold import random_isometry
from .runner import choose_batch
from .config import merge_config
from .design import build_design
from .study_protocol import stable_seed
from .study_io import (read_json, atomic_json, load_plan, task_lock, commit_round,
                       verify_round, file_digest)
from .study_diagnostics import (stationarity, convergence_check, prediction_metrics,
                                exact_gain_optimum, audit_design)


def load_observations(path):
    return [Observation(r['experiment_id'], tuple(r['counts']), r['shots']) for r in read_json(path)]


def fixed_batch(pool, batch_size, round_index, seed, policy):
    """A predetermined cyclic schedule, balanced over each completed cycle.

    fixed_ic repeats the 12 informationally complete one-step settings.
    fixed_pool covers all candidate settings once per cycle. The permutation
    is chosen before data and repeated unchanged; neither policy uses a fit.
    """
    ids = pilot_ids(pool) if policy == 'fixed_ic' else list(range(len(pool)))
    if batch_size > len(ids): raise ValueError('Fixed batch exceeds schedule length')
    order = np.random.default_rng(seed).permutation(ids)
    offset = (round_index-1)*batch_size
    return [int(order[(offset+j) % len(order)]) for j in range(batch_size)]


def restore_oracle(oracle, observations, pool):
    """Restore deterministic visit streams and counters from committed counts."""
    oracle.visits = {}
    oracle.measurement_shots = oracle.circuit_executions = oracle.channel_applications = 0
    for obs in observations:
        oracle.visits[obs.experiment_id] = oracle.visits.get(obs.experiment_id, 0)+1
        oracle.measurement_shots += obs.shots
        oracle.circuit_executions += 1
        oracle.channel_applications += obs.shots*pool[obs.experiment_id].steps


def completed_result(directory, manifest):
    directory = Path(directory)
    marker = directory/'task_complete.json'
    if not marker.exists(): return None
    record = read_json(marker)
    if record['manifest'] != manifest['fingerprint']:
        raise ValueError('Task belongs to a different manifest')
    if file_digest(directory/'result.json') != record['result_sha256']:
        raise ValueError('Task result checksum mismatch')
    return read_json(directory/'result.json')


def run_task(study, index, recover_lock=False, stop_after_round=None):
    """Execute one trajectory; an interrupted round is repeated deterministically.

    stop_after_round is a development/recovery-test hook. Only committed
    rounds count toward retained resource totals; abandoned attempts are
    identified in documentation as unmeasured interruption overhead.
    """
    study = Path(study); manifest = load_plan(study, executing=True)
    if index < 0 or index >= len(manifest['tasks']): raise IndexError('Task index out of range')
    task = manifest['tasks'][index]; c = task['config']; seeds = task['seeds']
    directory = study/'tasks'/task['id']
    with task_lock(directory, recover=recover_lock):
        done = completed_result(directory, manifest)
        if done is not None: return {'index': index, 'status': 'already_complete', 'id': task['id']}
        atomic_json(directory/'task.json', task)
        pool = make_pool(1, c['pool_steps'], c['readout'])
        reference = gksl_channel(**c['channel'])
        oracle = AerOracle(reference, task['true_readout'], seeds['acquisition'])
        initial = random_isometry(2, c['rank'], seeds['initialization'])
        stack, observations, history = initial, [], []
        committed = sorted(directory.glob('round_[0-9][0-9][0-9]'))
        for expected, folder in enumerate(committed):
            if folder.name != f'round_{expected:03d}': raise ValueError('Noncontiguous saved rounds')
            row = verify_round(folder)
            if row['round'] != expected: raise ValueError('Round index mismatch')
            history.append(row)
        if committed:
            observations = load_observations(committed[-1]/'observations.json')
            with np.load(committed[-1]/'model.npz', allow_pickle=False) as saved: stack = saved['stack']
            restore_oracle(oracle, observations, pool)
        for r in range(len(history), c['rounds']+1):
            model = None
            if r == 0:
                selected, shots = pilot_ids(pool), c['pilot_shots']
                selection = {'solver': 'common_pilot', 'selection_seconds': 0., 'audit_seconds': 0.,
                             'objective_shots': 0, 'sample_shots': 0, 'circuit_executions': 0}
            else:
                shots = prospective_shots(pool, c)
                if task['policy'] in ('fixed_ic', 'fixed_pool', 'fixed_prior'):
                    started = perf_counter()
                    selected = (task['fixed_prior_schedule'][r-1] if task['policy'] == 'fixed_prior' else
                                fixed_batch(pool, c['batch_size'], r, seeds['selection'], task['policy']))
                    selection = {'solver': task['policy'], 'selection_seconds': perf_counter()-started,
                                 'audit_seconds': 0., 'objective_shots': 0, 'sample_shots': 0, 'circuit_executions': 0}
                else:
                    selection_config = {k: v for k, v in c.items() if k not in
                                        ('channel', 'readout', 'seeds', 'validation_steps', 'validation_probes', 'target_rmse')}
                    selected, selection, model = choose_batch(task['policy'], stack, pool, observations,
                                                              selection_config, stable_seed(seeds['selection'], r))
            started = perf_counter()
            observations.extend(measure_selected(oracle, pool, selected, shots))
            acquisition_time = perf_counter()-started
            fit, fit_trials, candidate_models = fit_candidates(
                pool, observations, stack, c['fit'], task.get('acquisition_fitting', {}).get('fresh_starts', 0),
                seeds['initialization'], r)
            stack = fit.stack
            started = perf_counter(); check = stationarity(pool, observations, stack)
            monitor_time = perf_counter()-started
            previous = history[-1] if history else {}
            row = {'round': r, 'selected_ids': selected, 'selected_labels': [pool[i].label for i in selected],
                   'fit': dict(fit.diagnostics, final_gradient_norm=check['gradient_norm']), 'selection': selection,
                   'selected_shots': [int(shots if np.ndim(shots) == 0 else shots[i]) for i in selected],
                   'characterization_shots': oracle.measurement_shots,
                   'characterization_circuits': oracle.circuit_executions, 'channel_applications': oracle.channel_applications}
            increments = {'acquisition_seconds': acquisition_time, 'fit_seconds': fit.diagnostics['seconds']+monitor_time,
                          'selection_seconds': selection['selection_seconds'],
                          'optimizer_objective_shots': selection['objective_shots'],
                          'optimizer_sample_shots': selection['sample_shots'],
                          'optimizer_circuits': selection['circuit_executions']}
            for key, increment in increments.items(): row[key] = previous.get(key, 0)+increment
            row['total_attempted_shots'] = row['characterization_shots']+row['optimizer_objective_shots']+row['optimizer_sample_shots']
            row['algorithm_seconds'] = sum(row[k] for k in ['fit_seconds', 'acquisition_seconds', 'selection_seconds'])
            staging = Path(tempfile.mkdtemp(prefix=f'.round_{r:03d}_', dir=directory))
            np.savez_compressed(staging/'model.npz', stack=stack)
            atomic_json(staging/'observations.json', [asdict(o) for o in observations])
            atomic_json(staging/'fit_history.json', fit.history)
            if fit_trials is not None:
                atomic_json(staging/'fit_candidates.json', fit_trials)
                np.savez_compressed(staging/'candidate_models.npz', stacks=candidate_models)
            atomic_json(staging/'row.json', row)
            if model is not None:
                np.savez_compressed(staging/'design.npz', linear=model.linear, pairs=model.pairs,
                                    experiment_ids=model.experiment_ids, batch_size=model.batch_size)
            commit_round(staging, directory/f'round_{r:03d}')
            history.append(row)
            if stop_after_round == r: return {'index': index, 'status': 'paused', 'round': r}
        # Entire trajectory is now frozen. Only here is truth-based scoring permitted.
        validation = heldout_pool(1, c['validation_steps'], c['validation_probes'], seeds['validation'], c['readout'])
        true_validation = heldout_pool(1, c['validation_steps'], c['validation_probes'], seeds['validation'], task['true_readout'])
        for row in history:
            with np.load(directory/f"round_{row['round']:03d}"/'model.npz', allow_pickle=False) as saved:
                row['metrics'] = prediction_metrics(saved['stack'], reference, validation, true_validation)
        convergence = None
        if manifest['spec']['convergence']['enabled']:
            convergence, refitted = convergence_check(pool, observations, stack, manifest['spec']['convergence'],
                                                     c['fit'], seeds['refit'])
            convergence['refit_metrics'] = prediction_metrics(refitted, reference, validation, true_validation)
            np.savez_compressed(directory/'offline_refit.npz', stack=refitted)
        hit = next((h['characterization_shots'] for h in history if h['metrics']['heldout_rmse'] <= c['target_rmse']), None)
        result = {'task_id': task['id'], 'manifest': manifest['fingerprint'], 'history': history,
                  'first_target_shots': hit, 'convergence': convergence}
        atomic_json(directory/'result.json', result)
        atomic_json(directory/'task_complete.json', {'manifest': manifest['fingerprint'],
                                                     'result_sha256': file_digest(directory/'result.json')})
        return {'index': index, 'status': 'complete', 'id': task['id']}


def run_audit_task(study, index, recover_lock=False):
    """Audit common frozen snapshots, with separately recorded quantum resources."""
    study = Path(study); manifest = load_plan(study, executing=True)
    task = manifest['tasks'][index]; settings = manifest['spec']['audits']
    if task['policy'] != settings['source_policy']:
        return {'index': index, 'status': 'not_audit_source'}
    directory = study/'tasks'/task['id']
    if completed_result(directory, manifest) is None:
        raise RuntimeError('Finish the source trajectory before auditing')
    output = study/'audits'/task['id']
    with task_lock(output, recover=recover_lock):
        invocation_started = perf_counter()
        c = task['config']; pool = make_pool(1, c['pool_steps'], c['readout'])
        results = []
        for r in settings['rounds']:
            folder = directory/f'round_{r-1:03d}'; verify_round(folder)
            observations = load_observations(folder/'observations.json')
            with np.load(folder/'model.npz', allow_pickle=False) as saved: stack = saved['stack']
            design = build_design(blocks(stack), pool, observations, prospective_shots(pool, c), c['ridge'], c['fisher_probability_floor'])
            cache = {}
            for vi, variant in enumerate(settings['variants']):
                path = output/f'round_{r:03d}_variant_{vi:03d}.json'
                if path.exists():
                    record = read_json(path)
                    if record.get('manifest') != manifest['fingerprint']:
                        raise ValueError('Audit manifest mismatch')
                else:
                    batch = variant['batch_size']
                    if batch not in cache:
                        cache[batch] = exact_gain_optimum(design, range(len(pool)), batch, settings['max_combinations'])
                    options = merge_config({'qaoa': variant['qaoa']}, c)['qaoa']
                    seed = stable_seed(task['seeds']['selection'], r, variant['id'], 'offline_audit')
                    record = audit_design(design, variant, options, seed, settings['max_combinations'], cache[batch])
                    record.update(manifest=manifest['fingerprint'], task_id=task['id'], round=r)
                    atomic_json(path, record)
                results.append(record)
        atomic_json(output/'summary.json', results)
        atomic_json(output/'execution.json', {'manifest': manifest['fingerprint'],
                    'last_invocation_seconds': perf_counter()-invocation_started,
                    'note': 'Includes information construction and exact enumeration in this invocation; cached records are not rerun.'})
    return {'index': index, 'status': 'audited', 'snapshots': len(results)}
