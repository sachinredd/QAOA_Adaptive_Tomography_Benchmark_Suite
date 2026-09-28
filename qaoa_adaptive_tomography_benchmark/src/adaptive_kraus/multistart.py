"""Training-only choice among reproducible acquisition-time channel fits."""
from time import perf_counter
import numpy as np
from .learning import fit_channel, FitResult
from .manifold import random_isometry
from .study_protocol import stable_seed


def fit_candidates(pool, observations, initial, fit_options, fresh_starts, seed, round_index):
    """Fit the incumbent and fresh global isometries; choose minimum training NLL.

    Seeds depend on the paired initialization stream, round and candidate, never
    on policy, task scheduling or held-out scores. Ties prefer the incumbent,
    then the earliest fresh candidate. All computation is charged to the fit.
    The function takes no reference channel or validation data.
    """
    if isinstance(fresh_starts, bool) or not isinstance(fresh_starts, int) or fresh_starts < 0:
        raise ValueError('fresh_starts must be a nonnegative integer')
    if fresh_starts == 0:
        return fit_channel(pool, observations, initial, **fit_options), None, None
    started = perf_counter()
    results, trials = [], []
    d = initial.shape[1]; rank = initial.shape[0] // d
    for index in range(fresh_starts + 1):
        candidate_seed = None if index == 0 else stable_seed(seed, 'acquisition_fit', round_index, index)
        start = initial if index == 0 else random_isometry(d, rank, candidate_seed, near_identity=False)
        fit = fit_channel(pool, observations, start, **fit_options)
        if not np.isfinite(fit.diagnostics['nll']) or not np.all(np.isfinite(fit.stack)):
            raise FloatingPointError(f'Nonfinite acquisition fit at candidate {index}')
        results.append(fit)
        trials.append({'candidate': index, 'start': 'incumbent' if index == 0 else 'fresh',
                       'seed': candidate_seed, 'diagnostics': dict(fit.diagnostics), 'history': fit.history})
    winner = min(range(len(results)), key=lambda i: results[i].diagnostics['nll'])
    chosen = results[winner]
    diagnostics = dict(chosen.diagnostics)
    for key in ['evaluations', 'gradient_evaluations']:
        diagnostics['selected_' + key] = diagnostics[key]
        diagnostics[key] = sum(fit.diagnostics[key] for fit in results)
    diagnostics['selected_seconds'] = diagnostics['seconds']
    diagnostics['seconds'] = perf_counter() - started
    diagnostics['multistart'] = {'candidates': len(results), 'selected_candidate': winner,
                                'incumbent_nll': results[0].diagnostics['nll'],
                                'gain_over_incumbent': results[0].diagnostics['nll'] - diagnostics['nll']}
    return (FitResult(chosen.stack, chosen.history, diagnostics), trials,
            np.asarray([fit.stack for fit in results]))
