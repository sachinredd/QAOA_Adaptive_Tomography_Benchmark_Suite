"""Offline fitting checks and exact decomposition of batch-design losses."""
from itertools import combinations, islice
from math import comb
from time import perf_counter
import numpy as np
from .channels import blocks, channel_metrics
from .experiments import predict_probabilities
from .learning import CountObjective, fit_channel
from .manifold import project_tangent, random_isometry
from .design import batch_gain, make_binary_design, classical_select, greedy_batch
from .qaoa import solve_qaoa


def stationarity(pool, observations, stack):
    objective = CountObjective(pool, observations)
    loss, gradient = objective.value_grad(stack)
    return {'nll': loss, 'gradient_norm': float(np.linalg.norm(project_tangent(stack, gradient)))}


def convergence_check(pool, observations, stack, settings, fit_options, seed):
    """Frozen-data continuation and fresh starts. Select solely by training NLL.

    The return value is an offline diagnostic and must never replace the
    adaptive trajectory's model. Stationarity is not a global certificate.
    """
    started = perf_counter()
    original = stationarity(pool, observations, stack)
    best_stack, best_loss = stack.copy(), original['nll']
    trials = []
    options = dict(fit_options, iterations=settings['extra_iterations'])
    for i in range(settings['restarts']+1):
        initial = stack if i == 0 else random_isometry(stack.shape[1], stack.shape[0]//stack.shape[1],
                                                       seed+i, near_identity=False)
        fit = fit_channel(pool, observations, initial, **options)
        check = stationarity(pool, observations, fit.stack)
        trials.append(dict(start='continuation' if i == 0 else f'fresh_{i}', **fit.diagnostics,
                           final_gradient_norm=check['gradient_norm']))
        if check['nll'] < best_loss:
            best_stack, best_loss = fit.stack.copy(), check['nll']
    best_check = stationarity(pool, observations, best_stack)
    gain = original['nll']-best_loss
    result = {'original': original, 'best': best_check, 'nll_improvement': gain,
              'original_stationary': original['gradient_norm'] <= settings['gradient_tolerance'],
              'material_refit_improvement': gain > settings['nll_gain_tolerance'],
              'trials': trials, 'seconds': perf_counter()-started,
              'qualification': 'Offline sensitivity check; no global optimality claim; no model replacement.'}
    return result, best_stack


def exact_gain_optimum(design, candidates, batch_size, max_combinations=150000, chunk_size=512):
    """Enumerate bounded subsets in chunks, using batched Cholesky log-dets."""
    ids = list(map(int, candidates)); count = comb(len(ids), batch_size)
    if count > max_combinations:
        return {'status': 'skipped_guard', 'combinations': count, 'gain': None, 'selected': None}
    if not 1 <= batch_size <= len(ids): raise ValueError('Invalid exact-design batch')
    iterator = combinations(ids, batch_size)
    best_gain, best = -np.inf, None
    eye = np.eye(design.whitened.shape[-1])
    while chunk := list(islice(iterator, chunk_size)):
        index = np.asarray(chunk)
        matrices = eye+design.whitened[index].sum(axis=1)
        matrices = (matrices+matrices.swapaxes(-1, -2))/2
        lower = np.linalg.cholesky(matrices)
        gains = 2*np.log(np.diagonal(lower, axis1=-2, axis2=-1)).sum(axis=1)
        winner = int(np.argmax(gains))
        if gains[winner] > best_gain:
            best_gain, best = float(gains[winner]), list(map(int, chunk[winner]))
    return {'status': 'exact', 'combinations': count, 'gain': best_gain, 'selected': best}


def audit_design(design, variant, qaoa_options, seed, max_combinations=150000, full_optimum=None):
    """Same frozen information matrices for every selector and approximation.

    The QAOA contribution in *true gain* is signed. Only its QUBO energy gap
    is guaranteed nonnegative. A solver error can accidentally improve the
    true objective when the pair surrogate misranks batches.
    """
    started = perf_counter()
    batch, size = variant['batch_size'], variant['shortlist_size']
    binary = make_binary_design(design, size, batch)
    full = full_optimum or exact_gain_optimum(design, range(len(design.whitened)), batch, max_combinations)
    short = exact_gain_optimum(design, binary.experiment_ids, batch, max_combinations)
    if short['status'] != 'exact':
        raise ValueError('Shortlist exact audit exceeds guard; reduce shortlist or batch size')
    surrogate, _ = classical_select(binary, 'exact_qubo', np.random.default_rng(0))
    # Audit truth is never supplied to solve_qaoa.
    bits, solver = solve_qaoa(binary, seed=seed, **qaoa_options)
    surrogate_ids = binary.experiment_ids[np.flatnonzero(surrogate)].tolist()
    selected = binary.experiment_ids[np.flatnonzero(bits)].tolist()
    gs, gq = batch_gain(design, surrogate_ids), batch_gain(design, selected)
    greedy = greedy_batch(design, batch)
    result = {'variant': variant['id'], 'batch_size': batch, 'shortlist_size': size,
              'qaoa_options': qaoa_options, 'seed': seed, 'full_optimum': full, 'shortlist_optimum': short,
              'surrogate_optimum_ids': surrogate_ids, 'qaoa_ids': selected,
              'surrogate_optimum_true_gain': gs, 'qaoa_true_gain': gq,
              'greedy_true_gain': batch_gain(design, greedy),
              'shortlist_regret': None if full['gain'] is None else full['gain']-short['gain'],
              'surrogate_regret': short['gain']-gs,
              'qaoa_signed_true_gain_loss': gs-gq,
              'qaoa_surrogate_energy_gap': float(binary.energy(bits)-binary.energy(surrogate)),
              'total_true_gain_regret': None if full['gain'] is None else full['gain']-gq,
              'qaoa_resources': solver, 'seconds': perf_counter()-started}
    if full['gain'] is not None:
        parts = result['shortlist_regret']+result['surrogate_regret']+result['qaoa_signed_true_gain_loss']
        result['decomposition_residual'] = parts-result['total_true_gain_regret']
    else:
        result['decomposition_residual'] = None
    return result


def prediction_metrics(stack, reference, assumed_validation, true_validation):
    """Truth is used after acquisition, never in fitting or experimental design."""
    predicted = predict_probabilities(blocks(stack), assumed_validation)
    expected = predict_probabilities(reference, true_validation)
    metrics = channel_metrics(blocks(stack), reference)
    metrics['heldout_rmse'] = float(np.sqrt(np.mean((predicted-expected)**2)))
    metrics['heldout_max_absolute_error'] = float(np.max(np.abs(predicted-expected)))
    return metrics
