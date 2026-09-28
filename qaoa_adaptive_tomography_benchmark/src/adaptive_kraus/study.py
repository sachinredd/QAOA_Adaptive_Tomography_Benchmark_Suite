"""Plan, execute, audit and analyse reproducible simulation studies."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import multiprocessing
import sys
from .study_protocol import validate_spec, expand_spec, plan_estimate
from .study_io import read_json, create_plan, load_plan, atomic_json
from .study_execution import run_task, run_audit_task, completed_result
from .study_analysis import analyse


def execute_many(study, indices, workers=1, audit=False, recover=False):
    function = run_audit_task if audit else run_task
    failures = []
    if workers == 1:
        for index in indices:
            try:
                result = function(study, index, recover)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                error = {'index': index, 'error': str(exc), 'type': type(exc).__name__}
                failures.append(error); print(json.dumps(error), file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(function, study, index, recover): index for index in indices}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    error = {'index': futures[future], 'error': str(exc), 'type': type(exc).__name__}
                    failures.append(error); print(json.dumps(error), file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; successful commits were retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Freeze the protocol and deterministic task grid')
    plan.add_argument('--spec', required=True); plan.add_argument('--output')
    plan.add_argument('--dry-run', action='store_true', help='Print budgets without writing a plan')
    for name in ['run', 'audit']:
        p = sub.add_parser(name, help='Resume independent trajectories' if name=='run' else 'Audit saved frozen design snapshots')
        p.add_argument('--study', required=True); p.add_argument('--workers', type=int, default=1)
        p.add_argument('--task-index', type=int, help='Run one manifest task, e.g. an HPC array index')
        p.add_argument('--max-tasks', type=int, help='Limit selected tasks in this invocation')
        p.add_argument('--recover-lock', action='store_true', help='Recover a dead owner on this host only')
    status = sub.add_parser('status', help='Report completed and pending trajectories')
    status.add_argument('--study', required=True)
    analysis = sub.add_parser('analyse', aliases=['analyze'], help='Write paired comparisons, CSV data, plots and report')
    analysis.add_argument('--study', required=True); analysis.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        spec = validate_spec(read_json(args.spec))
        if args.dry_run:
            print(json.dumps(plan_estimate(spec, expand_spec(spec)), indent=2)); return
        if not args.output: parser.error('plan requires --output unless --dry-run is used')
        manifest = create_plan(spec, args.output)
        print(json.dumps({'fingerprint': manifest['fingerprint'], **manifest['estimate']}, indent=2)); return
    manifest = load_plan(args.study)
    if args.command in ['analyse', 'analyze']:
        print(json.dumps(analyse(args.study, args.allow_partial), indent=2)); return
    if args.command == 'status':
        pending = [t['index'] for t in manifest['tasks'] if completed_result(Path(args.study)/'tasks'/t['id'], manifest) is None]
        print(json.dumps({'complete': len(manifest['tasks'])-len(pending), 'expected': len(manifest['tasks']),
                          'pending_indices': pending}, indent=2)); return
    if args.workers < 1 or (args.max_tasks is not None and args.max_tasks < 1):
        parser.error('workers and max-tasks must be positive')
    indices = [args.task_index] if args.task_index is not None else list(range(len(manifest['tasks'])))
    if any(i<0 or i>=len(manifest['tasks']) for i in indices): parser.error('task-index out of range')
    if args.command == 'audit':
        indices = [i for i in indices if manifest['tasks'][i]['policy'] == manifest['spec']['audits']['source_policy']]
    elif args.task_index is None:
        indices = [i for i in indices if completed_result(Path(args.study)/'tasks'/manifest['tasks'][i]['id'], manifest) is None]
    if args.max_tasks is not None: indices = indices[:args.max_tasks]
    execute_many(str(Path(args.study).resolve()), indices, args.workers, args.command=='audit', args.recover_lock)


if __name__ == '__main__':
    main()
