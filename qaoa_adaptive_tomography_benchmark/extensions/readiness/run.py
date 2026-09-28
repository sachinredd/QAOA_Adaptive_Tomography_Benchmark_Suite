"""Frozen-stage driver; run `python .../run.py --help` for the execution order."""
from common import *
from solver import *
from adaptive_kraus.design import BinaryDesign, build_design, make_binary_design
from adaptive_kraus.channels import blocks
from adaptive_kraus.experiments import make_pool
from adaptive_kraus.study_sampling import broad_parameters
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, math, platform, importlib.metadata

def execute(jobs, worker, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            result=f.result()
            if i%12==0 or i==len(jobs): print(i,'/',len(jobs),result,flush=True)

def freeze():
    if (HERE/'protocol.json').exists():
        protocol_guard(); print('Existing protocol verified'); return
    p=dict(name='QAOA initialization and objective mechanism extension',frozen_utc=now(),
        base_commit='58f7a07',archived_source_sha256=source_digest(),extension_code_sha256=code_hash(),
        status='Local timestamped protocol, not external preregistration',
        development='All 36 historical channels (6 archived development and 30 prior evaluation); checkpoints 1 and 2 only. Prior evaluation is now development, never fresh confirmation.',
        factorial={'initialization':['swept','dicke'],'objective':['mean','lower_CVaR_0.2'],
                   'training':['128_shot','exact_diagnostic'],'depth':1,'cap':80,'final_shots':256,'solver_seeds':[8201,8219,8233]},
        selection='Among cvar,dicke,dicke_cvar, lowest channel-averaged expected best-of-256 normalized regret with finite-shot training; tie within 1e-12 prefers cvar,dicke,dicke_cvar in that order. Baseline is retained regardless of selection.',
        sample_size='Use development channel SD of selected-minus-baseline sampled regret: N=6*ceil((1.96*SD/0.01)^2/6), bounded to 30..60. Approximate planning only; report whether the cap limits target precision. No outcome-based expansion.',
        fresh=dict(channel_seed=280920261,acquisition_root=28092026,repetitions=2,families=list(FAMILIES),
                   distribution='unchanged broad_v1',checkpoints=[1,2],solver_seeds=[8401,8419,8433]),
        primary='Fresh frozen-state selected-minus-baseline sampled best-of-256 normalized energy regret; average 3 solver seeds, checkpoints 1 and 2, and 2 acquisition repetitions WITHIN physical channel; equal-family stratified t interval.',
        meaningful_margin=.01,success='95% upper bound below 0 establishes directional improvement; upper bound below -0.01 establishes improvement beyond the prespecified margin.',
        downstream=dict(policies=['greedy','swap','baseline','selected'],rounds=3,applications=4056,
                        primary_status='Descriptive secondary outcomes: RMSE, normalized Choi trace distance, learning curves and resources; no multiplicity-adjusted claims.'),
        landscape='Development only: first named seed; both initial states; 41x41 grid gamma in [-pi,pi], beta in [0,pi]. Bounded search, not global certificate.',
        limitations=['Ideal Dicke amplitudes are an initialization diagnostic; state-preparation gate cost is uncompiled and no hardware advantage is claimed.',
                     'Exact objectives and distribution expectations are offline diagnostics, not finite-shot policy resources.',
                     'All fresh comparisons use the original Aer oracle, readout, fitting and budget. Every fitting flag is retained.'],
        environment={m:importlib.metadata.version(m) for m in ['numpy','scipy','pandas','qiskit','qiskit-aer']})
    write(HERE/'protocol.json',p)
    print('Protocol frozen',digest(HERE/'protocol.json'))

def old_jobs():
    jobs=[]
    for split in ['development','evaluation']:
        for m in read(BASE/f'data/{split}_problems.json'):
            if m['n']==14 and m['b']==3 and m['round'] in [1,2]:
                m=dict(m,id=split+'_'+m['id'],key=split+'_'+m['key'])
                jobs.append(m)
    return jobs

def frozen_job(job):
    split,m,variants,seeds=job
    out=HERE/'results'/split/(m['key']+'.json')
    if out.exists(): return m['key'],'cached'
    protocol_guard()
    with np.load((BASE/m['path']) if split=='development' else HERE/m['path']) as z:
        model=BinaryDesign(z['linear'],z['pairs'],3,z['experiment_ids']);gains=z['gains']
    records=[]
    for s in seeds:
        paired=stable_seed(s,m['id'],m['round'],m.get('rep',0),'mechanism')
        for variant,exact in variants:
            result=run_solver(model,paired,variant,exact)
            p=np.asarray(result['probabilities']);best=best_distribution(model.energy(basis(14,3)[1]),p,256)
            result.update(id=m['id'],family=m['family'],round=m['round'],rep=m.get('rep',0),seed=s,
                true_gain=float(gains[result['selected_index']]),expected_best_gain=float(best@gains),
                gain_regret=float(gains.max()-gains[result['selected_index']]),problem=m['path'])
            records.append(result)
    write(out,dict(protocol_sha256=digest(HERE/'protocol.json'),records=records))
    return m['key'],'complete'

def development(workers):
    variants=[(v,e) for v in VARIANTS for e in [False,True]]
    execute([('development',m,variants,[8201,8219,8233]) for m in old_jobs()],frozen_job,workers)

def choose():
    import pandas as pd
    protocol_guard()
    if (HERE/'selection.json').exists(): raise RuntimeError('Selection already locked')
    rows=[r for p in sorted((HERE/'results/development').glob('*.json')) for r in read(p)['records']]
    assert len(rows)==36*2*3*8
    f=pd.DataFrame(rows);f=f[~f.exact_training]
    c=f.groupby(['id','family','variant'])[['expected_best_regret','sampled_regret']].mean().reset_index()
    scores=c.groupby('variant').expected_best_regret.mean().to_dict()
    candidates=['cvar','dicke','dicke_cvar'];best=min(scores[v] for v in candidates)
    chosen=next(v for v in candidates if scores[v]<=best+1e-12)
    wide=c.pivot(index=['id','family'],columns='variant',values='sampled_regret').reset_index()
    wide['difference']=wide[chosen]-wide.baseline
    # Within-family variance corresponds to the balanced target population.
    pooled_var=float(wide.groupby('family').difference.var(ddof=1).mean())
    requested=6*math.ceil((1.96*np.sqrt(pooled_var)/.01)**2/6)
    n=min(60,max(30,requested))
    locked=dict(locked_utc=now(),selected_variant=chosen,development_scores=scores,
        projected_sd=float(np.sqrt(pooled_var)),requested_n=requested,fresh_channels=n,
        projected_95_halfwidth=float(1.96*np.sqrt(pooled_var/n)),precision_cap_binding=requested>60,
        protocol_sha256=digest(HERE/'protocol.json'),development_records=len(rows))
    write(HERE/'selection.json',locked)
    tasks=[dict(id=f'fresh_{family}_{i:02d}',family=family,instance=i,
        channel=broad_parameters(family,i,1.,280920261)) for family in FAMILIES for i in range(n//6)]
    write(HERE/'data/channels.json',tasks);print(json.dumps(locked,indent=2))

def prepare():
    protocol_guard();pool=make_pool(1,[1,2,4,6,8],[.03,.06]);shots=np.array([408//e.steps for e in pool])
    index=[]
    for task in read(HERE/'data/channels.json'):
        for rep in [0,1]:
            for r in [1,2]:
                folder=HERE/'data/trajectories'/task['id']/f'rep_{rep}'/'greedy'/f'round_{r:03d}'
                stack=np.load(folder/'model.npz')['stack'];obs=observations(folder/'observations.json')
                d=build_design(blocks(stack),pool,obs,shots,1.,1e-6);m=make_binary_design(d,14,3)
                _,bits,_,_=basis(14,3);subsets=np.array([m.experiment_ids[np.flatnonzero(q)] for q in bits])
                matrices=np.eye(12)+d.whitened[subsets].sum(axis=1)
                gains=np.linalg.slogdet(matrices)[1]
                key=f"{task['id']}_rep{rep}_r{r}"
                path=Path('data/problems')/(key+'.npz')
                save_model(HERE/path,linear=m.linear,pairs=m.pairs,experiment_ids=m.experiment_ids,gains=gains)
                index.append(dict(id=task['id'],family=task['family'],round=r,rep=rep,key=key,path=str(path)))
    write(HERE/'data/problems.json',index);print('Prepared',len(index),'fresh frozen problems')

def landscape_job(m):
    out=HERE/'results/landscapes'/(m['key']+'.json')
    if out.exists():return m['key'],'cached'
    with np.load(BASE/m['path']) as z:model=BinaryDesign(z['linear'],z['pairs'],3,z['experiment_ids'])
    rng=np.random.default_rng(stable_seed(8201,m['id'],m['round'],0,'mechanism'))
    initial=np.zeros(14);initial[rng.choice(14,3,replace=False)]=1
    angles=np.array([(g,b) for g in np.linspace(-np.pi,np.pi,41) for b in np.linspace(0,np.pi,41)])
    rows=[]
    for variant in ['baseline','dicke']:
        c=make_circuit(model,initial,variant);p=probabilities_many(c,angles)
        gap=(c.energy-c.energy.min())/np.ptp(c.energy)
        expected=p@gap;best=np.array([best_distribution(c.energy,x,256)@gap for x in p])
        order=np.argsort(c.energy);optimal=c.energy<=c.energy.min()+1e-9
        rows.append(dict(id=m['id'],family=m['family'],round=m['round'],variant=variant,
            grid_min_expected_regret=float(expected.min()),grid_min_best_regret=float(best.min()),
            grid_max_success=float(np.max(1-(1-p[:,optimal].sum(axis=1))**256)),
            grid_optimum_support_anywhere=bool(np.any(p[:,optimal]>1e-12)),
            best_mean_angles=angles[np.argmin(expected)].tolist(),best_sample_angles=angles[np.argmin(best)].tolist()))
        # One compact landscape per family is sufficient for reproducible illustrations.
        if m['id'].startswith('development_'):
            save_model(out.with_name(out.stem+'_'+variant+'.npz'),angles=angles,expected_regret=expected,best_regret=best)
    write(out,{'records':rows});return m['key'],'complete'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','development','choose','scaffold','prepare','confirmation','adaptive','landscapes'])
    parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        protocol_guard()
        if args.stage=='development':development(args.workers)
        elif args.stage=='choose':choose()
        elif args.stage in ('scaffold','adaptive'):
            from trajectory import run_trajectory
            policies=['greedy'] if args.stage=='scaffold' else ['swap','baseline','selected']
            jobs=[(task,rep,p) for task in read(HERE/'data/channels.json') for rep in [0,1] for p in policies]
            execute(jobs,run_trajectory,args.workers)
        elif args.stage=='prepare':prepare()
        elif args.stage=='confirmation':
            variants=[('baseline',False),(read(HERE/'selection.json')['selected_variant'],False)]
            execute([('confirmation',m,variants,[8401,8419,8433]) for m in read(HERE/'data/problems.json')],frozen_job,args.workers)
        elif args.stage=='landscapes':execute(old_jobs(),landscape_job,args.workers)
