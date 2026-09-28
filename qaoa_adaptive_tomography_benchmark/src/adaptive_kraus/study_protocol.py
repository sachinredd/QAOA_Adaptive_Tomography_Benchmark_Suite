"""Deterministic one-qubit benchmark grids, frozen manifests and resource estimates."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import platform
import importlib.metadata
import numpy as np
from .config import merge_config, validate_config

POLICIES = ('fixed_ic', 'fixed_pool', 'fixed_prior', 'random', 'greedy', 'exact_qubo', 'random_qubo', 'qaoa')
FAMILIES = {
    'amplitude_damping': {'omega': [0., 0., 0.], 'rates': [.18, 0., 0.]},
    'thermal_relaxation': {'omega': [0., 0., 0.], 'rates': [.14, .055, 0.]},
    'dephasing': {'omega': [0., 0., 0.], 'rates': [0., 0., .09]},
    'coherent_rotation': {'omega': [.25, -.16, .32], 'rates': [0., 0., 0.]},
    'driven_dissipation': {'omega': [.25, -.16, .32], 'rates': [.12, .035, .045]},
    # Equal population and coherence decay: down=up=a; Z rate=a/2.
    'depolarizing': {'omega': [0., 0., 0.], 'rates': [.10, .10, .05]},
}
DEFAULT = {
    'schema_version': 1, 'name': 'simulation_study', 'channel_seed': 7301,
    'channel_sampling': 'nominal_multipliers',
    'families': list(FAMILIES), 'instances': 1, 'strengths': [.5, 1.5],
    'replicates': list(range(20)), 'policies': [p for p in POLICIES if p != 'fixed_prior'],
    'noise_profiles': [
        {'id': 'ideal', 'true_readout': [0., 0.], 'assumed_readout': [0., 0.]},
        {'id': 'calibrated_readout', 'true_readout': [.03, .06], 'assumed_readout': [.03, .06]}],
    'checkpoints': [0, 1, 3, 5],
    'base': {'fit': {'iterations': 600, 'tolerance': 1e-6}},
    'acquisition_fitting': {'fresh_starts': 0},
    'fixed_prior': {'seed': 713903, 'instances': 5},
    'convergence': {'enabled': True, 'extra_iterations': 1000, 'restarts': 2,
                    'nll_gain_tolerance': 1e-5, 'gradient_tolerance': 1e-4},
    'analysis': {'bootstrap_samples': 2000, 'seed': 8801, 'baseline': 'fixed_ic'},
    'audits': {'source_policy': 'greedy', 'rounds': [1], 'max_combinations': 150000,
               'variants': [
                   {'id': 'pair_m6', 'batch_size': 2, 'shortlist_size': 6, 'qaoa': {'depth': 1}},
                   {'id': 'triple_m6', 'batch_size': 3, 'shortlist_size': 6, 'qaoa': {'depth': 1}},
                   {'id': 'triple_m10', 'batch_size': 3, 'shortlist_size': 10, 'qaoa': {'depth': 1}},
                   {'id': 'triple_m6_p2', 'batch_size': 3, 'shortlist_size': 6, 'qaoa': {'depth': 2}},
                   {'id': 'triple_m6_noisy', 'batch_size': 3, 'shortlist_size': 6,
                    'qaoa': {'depth': 1, 'one_qubit_noise': .001, 'two_qubit_noise': .003}}]},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable_seed(*parts):
    """Independent named streams; never Python's process-randomized hash()."""
    return int(digest(list(parts))[:8], 16)


def source_digest():
    root = Path(__file__).parent
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))})


def environment():
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name)
        for name in ['numpy', 'scipy', 'qiskit', 'qiskit-aer', 'matplotlib']}}


def merge_known(base, update):
    result = deepcopy(base)
    if not isinstance(update, dict):
        raise ValueError('Expected a JSON object')
    for key, value in update.items():
        if key not in base:
            raise ValueError(f'Unknown study key: {key}')
        if key == 'base':
            result[key] = merge_config(value)
        elif isinstance(base[key], dict):
            result[key] = merge_known(base[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_spec(update):
    """Validate the entire grid before any expensive execution or output write."""
    s = merge_known(DEFAULT, update)
    if s['schema_version'] != 1:
        raise ValueError('Unsupported study schema')
    for key in ['families', 'replicates', 'policies', 'strengths', 'checkpoints']:
        if not s[key] or len(set(s[key])) != len(s[key]):
            raise ValueError(f'{key} must be nonempty and unique')
    if not set(s['families']) <= set(FAMILIES) or not set(s['policies']) <= set(POLICIES):
        raise ValueError('Unknown family or policy')
    positive_integer(s['instances'], 'instances')
    positive_integer(s['channel_seed'], 'channel_seed', 0)
    if s['channel_sampling'] not in ['nominal_multipliers', 'broad_v1']:
        raise ValueError('Unknown channel sampling scheme')
    positive_integer(s['acquisition_fitting']['fresh_starts'], 'acquisition_fitting.fresh_starts', 0)
    for x in s['replicates']: positive_integer(x, 'replicate', 0)
    for x in s['checkpoints']: positive_integer(x, 'checkpoint', 0)
    if s['checkpoints'] != sorted(s['checkpoints']) or s['checkpoints'][0] != 0 or s['checkpoints'][-1] < 1:
        raise ValueError('Use sorted checkpoints including pilot 0 and a later round')
    if any(not np.isfinite(x) or x <= 0 for x in s['strengths']):
        raise ValueError('Strengths must be finite positive numbers')
    noise_ids = []
    for noise in s['noise_profiles']:
        if set(noise) != {'id', 'true_readout', 'assumed_readout'}:
            raise ValueError('A noise profile needs id, true_readout and assumed_readout')
        noise_ids.append(noise['id'])
        for key in ['true_readout', 'assumed_readout']:
            if len(noise[key]) != 2 or any(not np.isfinite(x) or not 0 <= x < .5 for x in noise[key]):
                raise ValueError('Study readout errors must be finite probabilities below 0.5')
    if not noise_ids or len(set(noise_ids)) != len(noise_ids):
        raise ValueError('Use nonempty unique noise profiles')
    c = merge_config(s['base'])
    c['rounds'] = s['checkpoints'][-1]
    c['policies'] = [p for p in s['policies'] if p not in ('fixed_ic', 'fixed_pool', 'fixed_prior')] or ['random']
    c['seeds'] = s['replicates']
    validate_config(c)
    if c['channel']['n_qubits'] != 1 or c['batch_size'] > 12:
        raise ValueError('This study grid supports one system qubit and batch size <= 12')
    # Families and profiles own these reference fields; cannot be overridden silently.
    if any(k in update.get('base', {}) for k in ['channel', 'readout', 'seeds', 'policies', 'rounds']):
        raise ValueError('Use study families/profiles/replicates/policies/checkpoints for grid-owned fields')
    for n in s['noise_profiles']:
        trial = deepcopy(c); trial['readout'] = n['assumed_readout']; validate_config(trial)
    positive_integer(s['fixed_prior']['seed'], 'fixed_prior.seed', 0)
    positive_integer(s['fixed_prior']['instances'], 'fixed_prior.instances')
    if 'fixed_prior' in s['policies'] and (not c.get('applications_per_setting') or s['channel_sampling'] != 'broad_v1'):
        raise ValueError('fixed_prior requires an application budget and broad_v1 sampling')
    cv = s['convergence']
    if not isinstance(cv['enabled'], bool): raise ValueError('convergence.enabled must be Boolean')
    positive_integer(cv['extra_iterations'], 'extra_iterations')
    positive_integer(cv['restarts'], 'restarts', 0)
    if any(not np.isfinite(cv[k]) or cv[k] < 0 for k in ['nll_gain_tolerance', 'gradient_tolerance']):
        raise ValueError('Invalid convergence tolerance')
    a = s['analysis']; positive_integer(a['bootstrap_samples'], 'bootstrap_samples', 100)
    positive_integer(a['seed'], 'analysis seed', 0)
    if a['baseline'] not in s['policies']: raise ValueError('Analysis baseline must be a selected policy')
    audits = s['audits']
    if audits['source_policy'] not in s['policies'] and audits['variants']:
        raise ValueError('Audit source policy must be included')
    positive_integer(audits['max_combinations'], 'max_combinations')
    if len(set(audits['rounds'])) != len(audits['rounds']): raise ValueError('Duplicate audit rounds')
    for r in audits['rounds']:
        positive_integer(r, 'audit round')
        if r > c['rounds']: raise ValueError('Audit round exceeds trajectory')
    ids = []
    for variant in audits['variants']:
        if set(variant) != {'id', 'batch_size', 'shortlist_size', 'qaoa'}:
            raise ValueError('Audit variant requires id, batch_size, shortlist_size and qaoa')
        ids.append(variant['id'])
        trial = merge_config({'batch_size': variant['batch_size'], 'shortlist_size': variant['shortlist_size'],
                              'qaoa': variant['qaoa']}, c)
        trial['policies'] = ['qaoa']; validate_config(trial)
    if len(set(ids)) != len(ids): raise ValueError('Duplicate audit variant ID')
    s['base'] = c
    return s


def family_parameters(family, instance, strength, seed, sampling='nominal_multipliers'):
    """Draw an instance once, independently of acquisition/solver replicates.

    Scalar multipliers preserve the family: exact zero rates remain zero and
    the isotropic depolarizing rate ratios remain intact. Strength multiplies
    all frequencies and dissipative rates, so is a dynamical time-scale factor.
    """
    if sampling == 'broad_v1':
        from .study_sampling import broad_parameters
        return broad_parameters(family, instance, strength, seed)
    if sampling != 'nominal_multipliers':
        raise ValueError('Unknown channel sampling scheme')
    rng = np.random.default_rng(stable_seed(seed, family, instance, 'channel'))
    nominal = FAMILIES[family]
    wscale, rscale = rng.uniform(.8, 1.2, size=2)
    return dict(n_qubits=1, dt=1., zz=0.,
                omega=(np.array(nominal['omega'])*strength*wscale).tolist(),
                rates=(np.array(nominal['rates'])*strength*rscale).tolist())


def expand_spec(spec):
    tasks = []
    schedules = {}
    if 'fixed_prior' in spec['policies']:
        from .cost_budget import prior_design
        schedules = prior_design(spec)['schedules']
    for family in spec['families']:
        for instance in range(spec['instances']):
            for strength in spec['strengths']:
                channel = family_parameters(family, instance, strength, spec['channel_seed'], spec.get('channel_sampling', 'nominal_multipliers'))
                for noise in spec['noise_profiles']:
                    case = dict(family=family, instance=instance, strength=strength, noise=noise['id'])
                    case_id = digest(case)[:16]
                    for replicate in spec['replicates']:
                        c = deepcopy(spec['base']); c['channel'] = channel
                        c['readout'] = noise['assumed_readout']; c['audit_exact_gap'] = False
                        seeds = {name: stable_seed(spec['channel_seed'], case_id, replicate, name)
                                 for name in ['acquisition', 'initialization', 'selection', 'refit']}
                        # Fixed validation states across policies and replicates in a case.
                        seeds['validation'] = stable_seed(spec['channel_seed'], case_id, 'validation')
                        for policy in spec['policies']:
                            task = dict(case=case, case_id=case_id, replicate=replicate, policy=policy,
                                        seeds=seeds, config=c, true_readout=noise['true_readout'])
                            if spec['acquisition_fitting']['fresh_starts']:
                                task['acquisition_fitting'] = deepcopy(spec['acquisition_fitting'])
                            if policy == 'fixed_prior':
                                task['fixed_prior_schedule'] = schedules[noise['id']]
                            task['id'] = digest(task)[:24]
                            task['index'] = len(tasks)
                            tasks.append(task)
    return tasks


def plan_estimate(spec, tasks):
    c = spec['base']; rounds = c['rounds']
    char = 12*c['pilot_shots']+rounds*c['batch_size']*c['round_shots']
    q = c['qaoa']; qtasks = sum(t['policy'] == 'qaoa' for t in tasks)
    # COBYLA enforces an interpolation-set minimum even if maxiter is smaller.
    maxeval = max(q['maxiter'], 2*q['depth']+2)
    estimate = {'tasks': len(tasks), 'cases': len({t['case_id'] for t in tasks}),
            'replicates_per_case': len(spec['replicates']), 'policies': len(spec['policies']),
            'characterization_shots_per_trajectory': char,
            'characterization_shots_all_tasks': char*len(tasks),
            'acquisition_fit_candidates_per_round': 1 + spec['acquisition_fitting']['fresh_starts'],
            'acquisition_fit_iteration_cap_all_tasks': len(tasks)*(rounds+1)*c['fit']['iterations']*(1+spec['acquisition_fitting']['fresh_starts']),
            'checkpoint_shots': [12*c['pilot_shots']+r*c['batch_size']*c['round_shots'] for r in spec['checkpoints']],
            'optimizer_shots_budget_estimate': qtasks*rounds*(maxeval*q['restarts']*q['shots']+q['sample_shots']),
            'full_pool_subsets_per_audit': math.comb(18*len(c['pool_steps']), c['batch_size']),
            'notes': 'No wall-time promise. QAOA counts are a configured estimate; actual counts are recorded. Offline audit solver shots are additional.'}

    if c.get('applications_per_setting'):
        cap = c['applications_per_setting']
        estimate.update(characterization_shots_per_trajectory=None, characterization_shots_all_tasks=None,
            checkpoint_shots=None,
            channel_applications_per_trajectory=12*c['pilot_shots']+rounds*c['batch_size']*cap,
            checkpoint_channel_applications=[12*c['pilot_shots']+r*c['batch_size']*cap for r in spec['checkpoints']],
            characterization_shot_bounds_per_trajectory=[
                12*c['pilot_shots']+rounds*c['batch_size']*(cap//max(c['pool_steps'])),
                12*c['pilot_shots']+rounds*c['batch_size']*(cap//min(c['pool_steps']))])
        estimate['notes'] += ' Characterization shots vary; each selected setting costs the same number of unknown-channel applications. Prior design is offline computation.'
    return estimate
