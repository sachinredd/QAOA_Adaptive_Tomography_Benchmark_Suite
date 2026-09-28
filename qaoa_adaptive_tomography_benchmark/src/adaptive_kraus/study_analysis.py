"""Paired inference over acquisition replicates, conditional on a fixed channel grid."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from .study_io import load_plan, read_json, atomic_json
from .study_execution import completed_result
from .study_protocol import stable_seed, source_digest


def mean_interval(values, samples=2000, seed=1):
    """Percentile bootstrap of the mean; resample independent replicate units."""
    x = np.asarray(values, float)
    if not len(x): return {'n': 0, 'mean': None, 'sd': None, 'ci_low': None, 'ci_high': None}
    result = {'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)) if len(x)>1 else None,
              'ci_low': None, 'ci_high': None}
    if len(x)>1:
        rng = np.random.default_rng(seed)
        boot = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(1)
        result['ci_low'], result['ci_high'] = map(float, np.quantile(boot, [.025, .975]))
    return result


def paired_difference(policy_values, reference_values, samples=2000, seed=1):
    """Join on replicate IDs before subtracting; missing pairs stay excluded."""
    common = sorted(set(policy_values) & set(reference_values))
    differences = [policy_values[k]-reference_values[k] for k in common]
    result = mean_interval(differences, samples, seed)
    result.update(paired_ids=common, missing_policy=len(set(reference_values)-set(policy_values)),
                  missing_reference=len(set(policy_values)-set(reference_values)))
    return result


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def analyse(study, allow_partial=False):
    study = Path(study); manifest = load_plan(study); spec = manifest['spec']
    completed, missing, raw = [], [], []
    for task in manifest['tasks']:
        result = completed_result(study/'tasks'/task['id'], manifest)
        if result is None:
            missing.append(task['index']); continue
        completed.append((task, result))
        for h in result['history']:
            if h['round'] not in spec['checkpoints']: continue
            row = {'task_id': task['id'], 'case_id': task['case_id'], **task['case'],
                   'replicate': task['replicate'], 'policy': task['policy'], 'round': h['round'],
                   **h['metrics'], 'fit_status': h['fit']['status'],
                   'final_gradient_norm': h['fit']['final_gradient_norm']}
            for k in ['characterization_shots', 'total_attempted_shots', 'channel_applications',
                      'optimizer_objective_shots', 'optimizer_sample_shots', 'fit_seconds',
                      'selection_seconds', 'acquisition_seconds', 'algorithm_seconds']:
                row[k] = h[k]
            raw.append(row)
    if missing and not allow_partial:
        raise RuntimeError(f'{len(missing)} trajectories missing; finish the study or explicitly use --allow-partial')
    if not completed: raise RuntimeError('No complete trajectories available')
    output = study/'analysis'; output.mkdir(exist_ok=True)
    metrics = ['heldout_rmse', 'process_fidelity_squared', 'total_attempted_shots',
               'channel_applications', 'algorithm_seconds', 'final_gradient_norm']
    groups = defaultdict(list)
    for row in raw: groups[(row['case_id'], row['policy'], row['round'])].append(row)
    aggregate, paired = [], []
    for (case_id, policy, r), records in sorted(groups.items()):
        for metric in metrics:
            interval = mean_interval([x[metric] for x in records], spec['analysis']['bootstrap_samples'],
                                     stable_seed(spec['analysis']['seed'], case_id, policy, r, metric))
            aggregate.append({'case_id': case_id, 'policy': policy, 'round': r, 'metric': metric, **interval})
    baselines = list(dict.fromkeys([spec['analysis']['baseline'], 'greedy', 'exact_qubo']))
    for (case_id, policy, r), records in sorted(groups.items()):
        for baseline in baselines:
            if baseline == policy or (case_id, baseline, r) not in groups: continue
            reference = groups[(case_id, baseline, r)]
            contrast = paired_difference({x['replicate']: x['heldout_rmse'] for x in records},
                {x['replicate']: x['heldout_rmse'] for x in reference}, spec['analysis']['bootstrap_samples'],
                stable_seed(spec['analysis']['seed'], case_id, policy, baseline, r))
            paired.append({'case_id': case_id, 'policy': policy, 'baseline': baseline, 'round': r,
                           'metric': 'heldout_rmse_difference', **contrast})
    # Fixed-grid summaries average cases *within replicate*, then bootstrap the
    # replicate means. Channel instances and budget checkpoints are not IID rows.
    expected_cases = {t['case_id'] for t in manifest['tasks']}
    lookup = {(x['case_id'], x['replicate'], x['policy'], x['round']): x for x in raw}
    pooled = []; pooled_pairs = []
    for r in spec['checkpoints']:
        common = [seed for seed in spec['replicates'] if all((case, seed, policy, r) in lookup
                  for case in expected_cases for policy in spec['policies'])]
        for metric in metrics:
            values = {}
            for policy in spec['policies']:
                values[policy] = {seed: float(np.mean([lookup[(case, seed, policy, r)][metric]
                                                     for case in sorted(expected_cases)])) for seed in common}
                interval = mean_interval(list(values[policy].values()), spec['analysis']['bootstrap_samples'],
                                         stable_seed(spec['analysis']['seed'], policy, r, metric, 'pooled'))
                pooled.append({'policy': policy, 'round': r, 'metric': metric, 'cases': len(expected_cases),
                               'excluded_replicates': len(spec['replicates'])-len(common), **interval})
            if metric == 'heldout_rmse':
                for baseline in baselines:
                    if baseline not in spec['policies']: continue
                    for policy in spec['policies']:
                        if baseline == policy: continue
                        contrast = paired_difference(values[policy], values[baseline], spec['analysis']['bootstrap_samples'],
                                                     stable_seed(spec['analysis']['seed'], policy, baseline, r, 'pooled'))
                        pooled_pairs.append({'policy': policy, 'baseline': baseline, 'round': r, **contrast})
    convergence = []
    targets = []
    for task, result in completed:
        cv = result['convergence']
        if cv is not None:
            convergence.append({'task_id': task['id'], 'case_id': task['case_id'], 'replicate': task['replicate'],
                'policy': task['policy'], 'nll_improvement': cv['nll_improvement'],
                'original_gradient_norm': cv['original']['gradient_norm'], 'best_gradient_norm': cv['best']['gradient_norm'],
                'original_stationary': cv['original_stationary'], 'material_refit_improvement': cv['material_refit_improvement'],
                'original_rmse': result['history'][-1]['metrics']['heldout_rmse'],
                'offline_refit_rmse': cv['refit_metrics']['heldout_rmse'], 'offline_seconds': cv['seconds']})
        targets.append({'task_id': task['id'], 'case_id': task['case_id'], 'policy': task['policy'],
                        'replicate': task['replicate'], 'target_rmse': task['config']['target_rmse'],
                        'first_target_shots': result['first_target_shots'],
                        'right_censored': result['first_target_shots'] is None,
                        'final_characterization_shots': result['history'][-1]['characterization_shots']})
    audits = []
    for task, _ in completed:
        folder = study/'audits'/task['id']
        for path in sorted(folder.glob('round_*_variant_*.json')):
            a = read_json(path)
            if a['manifest'] != manifest['fingerprint']: raise ValueError('Audit manifest mismatch')
            audits.append({'task_id': task['id'], 'case_id': task['case_id'], 'replicate': task['replicate'],
                **{k: a[k] for k in ['round', 'variant', 'batch_size', 'shortlist_size', 'shortlist_regret',
                    'surrogate_regret', 'qaoa_signed_true_gain_loss', 'qaoa_surrogate_energy_gap',
                    'total_true_gain_regret', 'decomposition_residual']},
                'full_optimum_status': a['full_optimum']['status'],
                'offline_optimizer_shots': a['qaoa_resources']['objective_shots']+a['qaoa_resources']['sample_shots']})
    audit_expected = sum(t['policy'] == spec['audits']['source_policy'] for t in manifest['tasks'])*len(spec['audits']['rounds'])*len(spec['audits']['variants'])
    coverage = {'manifest': manifest['fingerprint'], 'analysis_source_digest': source_digest(),
                'complete_tasks': len(completed), 'expected_tasks': len(manifest['tasks']),
                'missing_indices': missing, 'partial': bool(missing), 'audit_records': len(audits), 'expected_audit_records': audit_expected,
                'inference_scope': 'Percentile 95% intervals over acquisition/initialization/solver replicates, conditional on the fixed channel grid. No multiple-comparison adjustment.'}
    atomic_json(output/'coverage.json', coverage)
    for name, rows in [('records', raw), ('case_summary', aggregate), ('paired_comparisons', paired),
                       ('fixed_grid_summary', pooled), ('fixed_grid_paired', pooled_pairs),
                       ('convergence', convergence), ('target_attainment', targets), ('design_audits', audits)]:
        write_csv(output/f'{name}.csv', rows)
    atomic_json(output/'summary.json', {'coverage': coverage, 'fixed_grid': pooled, 'paired': pooled_pairs})
    make_plots(output, spec, pooled, pooled_pairs, convergence)
    write_report(output, spec, coverage, pooled, pooled_pairs, convergence, targets, audits)
    return coverage


def make_plots(output, spec, pooled, paired, convergence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final = spec['checkpoints'][-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for policy in spec['policies']:
        rows = [x for x in pooled if x['policy']==policy and x['metric']=='heldout_rmse' and x['mean'] is not None]
        if not rows: continue
        x = [12*spec['base']['pilot_shots']+r['round']*spec['base']['batch_size']*spec['base']['round_shots'] for r in rows]
        y = [r['mean'] for r in rows]
        line, = axes[0].plot(x,y,'o-',label=policy,ms=3)
        if all(r['ci_low'] is not None for r in rows):
            axes[0].fill_between(x,[r['ci_low'] for r in rows],[r['ci_high'] for r in rows],alpha=.12,color=line.get_color())
        cost = next((r['mean'] for r in pooled if r['policy']==policy and r['round']==final and r['metric']=='total_attempted_shots'),None)
        last = next((r['mean'] for r in rows if r['round']==final),None)
        if cost is not None and last is not None:
            axes[1].scatter(cost,last,color=line.get_color(),s=35)
    contrasts = [r for r in paired if r['round']==final and r['baseline']==spec['analysis']['baseline'] and r['mean'] is not None]
    for i,row in enumerate(contrasts):
        lo,hi = row['ci_low'],row['ci_high']
        axes[2].plot(row['mean'],i,'o',color='#315971')
        if lo is not None: axes[2].hlines(i,lo,hi,color='#315971',lw=2)
    axes[2].set_yticks(range(len(contrasts)),[r['policy'] for r in contrasts]); axes[2].axvline(0,color='gray',ls='--',lw=1)
    axes[0].set(xlabel='Characterization shots',ylabel='Held-out probability RMSE',title='Fixed-grid mean and 95% bootstrap interval')
    if axes[0].lines: axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Total attempted shots',ylabel='Final held-out RMSE',title='Resource cost; colours match left panel')
    axes[1].set_xscale('log')
    axes[2].set(xlabel=f"RMSE difference versus {spec['analysis']['baseline']}",title='Paired differences; negative favours policy')
    for ax in axes: ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=180); plt.close(fig)
    if convergence:
        fig,ax=plt.subplots(figsize=(6.4,4))
        for policy in spec['policies']:
            rows=[r for r in convergence if r['policy']==policy]
            ax.scatter([r['original_gradient_norm'] for r in rows],[r['nll_improvement'] for r in rows],label=policy,s=18,alpha=.7)
        ax.set(xlabel='Original final projected gradient norm',
               ylabel='NLL improvement under offline refitting',title='Fitting sensitivity on frozen data')
        ax.set_xscale('symlog',linthresh=1e-6); ax.set_yscale('symlog',linthresh=1e-6)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_xticks([0,1e-6,1e-4,1e-2]); ax.set_yticks([0,1e-6,1e-4,1e-2])
        ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(output/'convergence.png',dpi=180);plt.close(fig)


def write_report(output, spec, coverage, pooled, paired, convergence, targets, audits):
    final = spec['checkpoints'][-1]
    lines = [f"# {spec['name']} simulation report", '',
             f"Completed trajectories: {coverage['complete_tasks']} / {coverage['expected_tasks']}.",
             f"Offline design audit records: {coverage['audit_records']} / {coverage['expected_audit_records']}.", '',
             '**Partial study: do not use as a complete comparison.**' if coverage['partial'] else 'The planned acquisition grid is complete.', '',
             coverage['inference_scope'], '',
             'Budget checkpoints come from nested trajectories. Rows across budgets are not independent replicates. Grid summaries give every planned channel/noise case equal weight and use only replicate IDs complete across all cases and policies.', '',
             '| Policy | Mean final RMSE | 95% interval | Replicates |', '|---|---:|---:|---:|']
    for row in pooled:
        if row['round'] != final or row['metric'] != 'heldout_rmse': continue
        mean = 'unavailable' if row['mean'] is None else f"{row['mean']:.5f}"
        ci = 'unavailable' if row['ci_low'] is None else f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]"
        lines.append(f"| {row['policy']} | {mean} | {ci} | {row['n']} |")
    lines += ['', '## Fitting checks', '']
    if convergence:
        material = sum(r['material_refit_improvement'] for r in convergence)
        stationary = sum(r['original_stationary'] for r in convergence)
        lines += [f"{stationary}/{len(convergence)} final fits meet the diagnostic gradient threshold. "
                  f"{material}/{len(convergence)} improve training NLL materially under continuation or fresh starts.",
                  'Offline refits never replace the adaptive models. If improvements are frequent, increase the primary fitting budget in a new plan and reassess policy rankings.']
    else: lines.append('Offline convergence checks were disabled or are unavailable.')
    lines += ['', '## Target attainment', '',
              f"{sum(not r['right_censored'] for r in targets)}/{len(targets)} trajectories attain the configured RMSE target at a saved round. "
              'Unreached targets remain right-censored; they are not assigned a successful hitting time at the final budget.', '',
              '## Design audits', '',
              'Shortlist regret and surrogate regret are measured in the exact local log-determinant objective. '
              'The QAOA energy gap measures solver error on the binary surrogate. Its contribution to true log-determinant loss is signed: an approximate surrogate solution can accidentally improve the true objective. '
              'A skipped full-pool enumeration is recorded as missing, never zero.', '',
              f"Offline selector shots recorded in available audits: {sum(a['offline_optimizer_shots'] for a in audits):,}. These are separate from acquisition-policy costs.", '',
              '## Interpretation', '',
              'These simulations assume stationary repeated channels and ideal state preparation and basis rotations. '
              'Readout may be calibrated or deliberately misspecified according to the frozen profiles. '
              'No hardware speedup or broad population ranking follows from this fixed channel grid. '
              'Inspect paired comparisons against greedy and exact QUBO as well as the fixed baseline. '
              'Wall times depend on worker count and system load; shots and channel applications are more portable.', '',
              'See records.csv, paired_comparisons.csv, fixed_grid_paired.csv, convergence.csv, target_attainment.csv and design_audits.csv for the unrounded data.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
