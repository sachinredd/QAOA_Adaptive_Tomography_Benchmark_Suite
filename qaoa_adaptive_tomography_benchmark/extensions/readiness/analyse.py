"""Channel-clustered summaries; archived follow-up analyses are labelled post hoc."""
from common import *
from solver import best_distribution, make_circuit
from benchmark_solver import basis
from adaptive_kraus.design import BinaryDesign, build_design, batch_gain
from adaptive_kraus.channels import blocks
from adaptive_kraus.experiments import make_pool
import pandas as pd
from scipy.stats import spearmanr

OUT=HERE/'analysis'

def clean(rows):
    keys=['id','family','round','rep','seed','variant','exact_training','sampled_regret','expected_regret',
          'expected_best_regret','success_probability','optimal','effective_support','selection_noise_penalty',
          'objective_evaluations','objective_shots','final_shots','total_shots','true_gain','gain_regret','expected_best_gain','problem']
    return pd.DataFrame([{k:r[k] for k in keys} for r in rows])

def frozen(split):
    records=[r for p in sorted((HERE/'results'/split).glob('*.json')) for r in read(p)['records']]
    f=clean(records);f.to_csv(OUT/f'{split}_records.csv',index=False)
    metrics=['sampled_regret','expected_regret','expected_best_regret','success_probability','optimal',
             'effective_support','selection_noise_penalty','total_shots','gain_regret']
    channels=f.groupby(['id','family','variant','exact_training'])[metrics].mean().reset_index()
    channels.to_csv(OUT/f'{split}_channel_means.csv',index=False)
    means=[]
    for (v,e),g in channels.groupby(['variant','exact_training']):
        for metric in metrics:
            means.append(dict(variant=v,exact_training=bool(e),metric=metric,**interval(g,metric)))
    pd.DataFrame(means).to_csv(OUT/f'{split}_means.csv',index=False)
    contrasts=[]
    pairs=[('baseline','cvar'),('baseline','dicke'),('dicke','dicke_cvar'),('cvar','dicke_cvar')]
    if split=='confirmation':pairs=[('baseline',read(HERE/'selection.json')['selected_variant'])]
    for exact in sorted(channels.exact_training.unique()):
        subset=channels[channels.exact_training==exact]
        for first,second in pairs:
            for metric in ['sampled_regret','expected_regret','expected_best_regret','gain_regret']:
                w=subset.pivot(index=['id','family'],columns='variant',values=metric).reset_index()
                w['difference']=w[second]-w[first]
                contrasts.append(dict(first=second,second=first,exact_training=bool(exact),metric=metric,**interval(w,'difference')))
    if split=='development':
        for variant,g in channels.groupby('variant'):
            for metric in ['sampled_regret','expected_regret','expected_best_regret']:
                w=g.pivot(index=['id','family'],columns='exact_training',values=metric).reset_index()
                w['difference']=w[True]-w[False]
                contrasts.append(dict(first='exact_'+variant,second='finite_'+variant,exact_training=True,metric=metric,**interval(w,'difference')))
    pd.DataFrame(contrasts).to_csv(OUT/f'{split}_contrasts.csv',index=False)
    # Exact uniform reference is an explicit post hoc distribution diagnostic.
    # It does not receive an exchange rate against quantum objective shots.
    structural=[]
    for r in records:
        if r['variant']!='baseline' or r['exact_training']:continue
        path=(BASE if split=='development' else HERE)/r['problem']
        with np.load(path) as z:m=BinaryDesign(z['linear'],z['pairs'],3,z['experiment_ids'])
        c=make_circuit(m,np.asarray(r['initial_bits']),'baseline')
        support=np.zeros(len(c.integers),bool)
        integer=sum(int(v)<<i for i,v in enumerate(r['initial_bits']))
        support[c.index[integer]]=True
        # Boolean reachability is an upper bound: phases/cancellation can only remove support.
        for _ in range(3):  # two fixed initial sweeps plus one variational sweep
            for a,b in c.edges:
                union=support[a]|support[b];support[a]=union;support[b]=union
        forbidden=float(np.asarray(r['probabilities'])[~support].sum())
        assert forbidden<1e-14
        structural.append(dict(id=r['id'],family=r['family'],round=r['round'],rep=r['rep'],seed=r['seed'],
            reachable_batches=int(support.sum()),optimum_reachable=bool(np.any(support[c.energy<=c.energy.min()+1e-9])),
            forbidden_mass=forbidden))
    st=pd.DataFrame(structural);st.to_csv(OUT/f'{split}_reachability_posthoc.csv',index=False)
    uniform=[]
    seen=set()
    for r in records:
        key=(r['id'],r['round'],r['rep'])
        if key in seen:continue
        seen.add(key)
        path=(BASE if split=='development' else HERE)/r['problem']
        with np.load(path) as z:
            m=BinaryDesign(z['linear'],z['pairs'],3,z['experiment_ids'])
        energies=m.energy(basis(14,3)[1]);p=np.ones(len(energies))/len(energies)
        gap=(energies-energies.min())/np.ptp(energies)
        uniform.append(dict(id=r['id'],family=r['family'],round=r['round'],rep=r['rep'],
            expected_best_regret=float(best_distribution(energies,p,256)@gap),
            success_probability=float(1-(1-np.mean(energies<=energies.min()+1e-9))**256)))
    u=pd.DataFrame(uniform);u.to_csv(OUT/f'{split}_uniform_diagnostic.csv',index=False)
    uniform_channels=u.groupby(['id','family']).expected_best_regret.mean().rename('uniform').reset_index()
    uniform_contrasts=[]
    for variant,g in channels[~channels.exact_training].groupby('variant'):
        joined=g.merge(uniform_channels,on=['id','family'],validate='one_to_one')
        joined['difference']=joined.expected_best_regret-joined.uniform
        uniform_contrasts.append(dict(variant=variant,metric='expected_best_regret_minus_uniform',
                                      status='post_hoc',**interval(joined,'difference')))
    pd.DataFrame(uniform_contrasts).to_csv(OUT/f'{split}_uniform_contrasts_posthoc.csv',index=False)
    return dict(records=len(records),means=means,contrasts=contrasts,
                posthoc_reachable_optimum_fraction=float(st.optimum_reachable.mean()),
                posthoc_mean_reachable_batches=float(st.reachable_batches.mean()),
                posthoc_uniform_contrasts=uniform_contrasts,
                uniform_mean_expected_best_regret=float(u.expected_best_regret.mean()))

def retrospective():
    """Recompute own-model gains, existing channel metrics and first-step paired contrasts."""
    pool=make_pool(1,[1,2,4,6,8],[.03,.06]);shots=np.array([408//e.steps for e in pool])
    rows=[]
    for task in read(BASE/'data/evaluation_channels.json'):
        for policy in read(BASE/'protocol/PROTOCOL.json')['trajectory_stage']['policies']:
            folder=BASE/'data/trajectories'/task['id']/policy
            result=read(folder/'result.json')
            for h in result['history']:
                r=h['round'];gain=None
                if r:
                    previous=folder/f'round_{r-1:03d}'
                    d=build_design(blocks(np.load(previous/'model.npz')['stack']),pool,
                        observations(previous/'observations.json'),shots,1.,1e-6)
                    gain=batch_gain(d,h['selected'])
                rows.append(dict(id=task['id'],family=task['family'],policy=policy,round=r,
                    applications=h['channel_applications'],rmse=h['metrics']['heldout_rmse'],
                    choi_distance=h['metrics']['normalized_choi_trace_distance'],local_gain=gain))
    f=pd.DataFrame(rows);f.to_csv(OUT/'historical_learning_and_gain.csv',index=False)
    curve=[]
    for (policy,r),g in f.groupby(['policy','round']):
        for metric in ['rmse','choi_distance']:
            curve.append(dict(policy=policy,round=int(r),metric=metric,**interval(g,metric)))
    pd.DataFrame(curve).to_csv(OUT/'historical_curves.csv',index=False)
    # Only first acquisition has an identical fitted state across all branches.
    first=f[f['round']==1]
    g=first.pivot(index=['id','family'],columns='policy',values=['rmse','local_gain'])
    dg=g['local_gain']['qaoa_cold']-g['local_gain']['exact_qubo']
    de=g['rmse']['qaoa_cold']-g['rmse']['exact_qubo']
    association=float(spearmanr(dg,de).statistic)
    pairs=[]
    for r in [1,2,3]:
        for metric in ['rmse','choi_distance']:
            w=f[f['round']==r].pivot(index=['id','family'],columns='policy',values=metric).reset_index()
            for a,b in [('qaoa_cold','exact_qubo'),('qaoa_cold','greedy'),('greedy','fixed_prior')]:
                w['difference']=w[a]-w[b]
                pairs.append(dict(first=a,second=b,round=r,metric=metric,**interval(w,'difference')))
    pd.DataFrame(pairs).to_csv(OUT/'historical_paired_contrasts.csv',index=False)
    audits=[x for x in read(BASE/'data/evaluation_problems.json') if x['n']==14 and x['b']==3]
    ratios=np.array([a['full_pool_greedy_gain']/a['full_pool_optimum_gain'] for a in audits])
    final=f[f['round']==3].copy()
    baseline=f[(f['round']==0)&(f.policy=='greedy')].set_index('id').rmse
    final['rmse_reduction_from_pilot']=final.apply(lambda x: baseline[x['id']]-x.rmse,axis=1)
    summary=dict(status='post_hoc_historical_analysis',channels=30,
        first_round_gain_vs_error_spearman=association,
        first_round_mean_qaoa_minus_exact_gain=float(dg.mean()),
        first_round_mean_qaoa_minus_exact_rmse=float(de.mean()),
        greedy_true_gain_ratio_min=float(ratios.min()),greedy_true_gain_ratio_mean=float(ratios.mean()),
        greedy_exact_matches=int(np.sum(abs(ratios-1)<1e-10)),audited_states=len(ratios),
        pilot_rmse_mean=float(baseline.mean()),
        final_reduction_by_policy=final.groupby('policy').rmse_reduction_from_pilot.mean().to_dict(),
        paired_contrasts=pairs)
    write(OUT/'historical_summary.json',summary);return summary

def downstream():
    rows=[];learning=[];flags=[]
    tasks=read(HERE/'data/channels.json')
    for task in tasks:
        for rep in [0,1]:
            for policy in ['greedy','swap','baseline','selected']:
                result=read(HERE/'data/trajectories'/task['id']/f'rep_{rep}'/policy/'result.json')
                row={k:result[k] for k in ['id','family','rep','policy','rmse','choi_distance','applications','characterization_shots','objective_shots','final_shots']}
                row['total_shots']=row['characterization_shots']+row['objective_shots']+row['final_shots']
                check=result['convergence']
                row['stationarity_flag']=not check['original_stationary'];row['refit_flag']=check['material_refit_improvement']
                row['refit_improvement']=check['nll_improvement'];rows.append(row)
                if row['stationarity_flag'] or row['refit_flag']:flags.append(row)
                for h in result['history']:
                    learning.append(dict(id=task['id'],family=task['family'],rep=rep,policy=policy,round=h['round'],applications=h['applications'],
                        rmse=h['metrics']['heldout_rmse'],choi_distance=h['channel_metrics']['normalized_choi_trace_distance']))
    f=pd.DataFrame(rows);f.to_csv(OUT/'fresh_trajectories.csv',index=False)
    pd.DataFrame(flags,columns=f.columns).to_csv(OUT/'fresh_fitting_flags.csv',index=False)
    l=pd.DataFrame(learning);l.to_csv(OUT/'fresh_learning.csv',index=False)
    numeric=['rmse','choi_distance','characterization_shots','objective_shots','final_shots','total_shots']
    c=f.groupby(['id','family','policy'])[numeric].mean().reset_index()
    c.to_csv(OUT/'fresh_trajectory_channel_means.csv',index=False)
    means=[]
    for policy,g in c.groupby('policy'):
        for metric in numeric:means.append(dict(policy=policy,metric=metric,**interval(g,metric)))
    contrasts=[]
    for metric in ['rmse','choi_distance']:
        w=c.pivot(index=['id','family'],columns='policy',values=metric).reset_index()
        for a,b in [('selected','baseline'),('selected','greedy'),('selected','swap'),('baseline','swap'),('swap','greedy')]:
            w['difference']=w[a]-w[b]
            contrasts.append(dict(first=a,second=b,metric=metric,**interval(w,'difference')))
    pd.DataFrame(means).to_csv(OUT/'fresh_trajectory_means.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'fresh_trajectory_contrasts.csv',index=False)
    curves=[]
    for (policy,r),g in l.groupby(['policy','round']):
        cm=g.groupby(['id','family'])[['rmse','choi_distance']].mean().reset_index()
        for metric in ['rmse','choi_distance']:curves.append(dict(policy=policy,round=int(r),metric=metric,**interval(cm,metric)))
    pd.DataFrame(curves).to_csv(OUT/'fresh_curves.csv',index=False)
    within={}
    for policy,g in f.groupby('policy'):
        p=g.pivot(index='id',columns='rep',values='rmse')
        within[policy]=float(np.sqrt(np.mean((p[0]-p[1])**2)/2))
    return dict(trajectories=len(f),channels=len(tasks),means=means,contrasts=contrasts,
        within_channel_run_rmse_sd=within,replicate_variation='Acquisition, fitting initialization and solver streams vary; validation probes are common.',flagged_trajectories=len(flags),
        stationarity_flags=int(f.stationarity_flag.sum()),refit_flags=int(f.refit_flag.sum()),
        maximum_refit_gain=float(f.refit_improvement.max()))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--historical-only',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.historical_only:
        s=retrospective();print(json.dumps({k:v for k,v in s.items() if k!='paired_contrasts'},indent=2));return
    development=frozen('development');confirmation=frozen('confirmation');fresh=downstream()
    historical=read(OUT/'historical_summary.json') if (OUT/'historical_summary.json').exists() else retrospective()
    primary=next(x for x in confirmation['contrasts'] if x['metric']=='sampled_regret')
    # Stratified channel bootstrap is a sensitivity check, never substitutes primary t interval.
    f=pd.read_csv(OUT/'confirmation_channel_means.csv')
    w=f.pivot(index=['id','family'],columns='variant',values='sampled_regret').reset_index()
    selected=read(HERE/'selection.json')['selected_variant'];w['difference']=w[selected]-w.baseline
    rng=np.random.default_rng(28092311);boot=np.zeros(10000)
    for _,g in w.groupby('family'):
        x=g.difference.to_numpy();boot+=rng.choice(x,size=(10000,len(x))).mean(axis=1)/6
    bootstrap=dict(low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),replicates=10000)
    result=dict(completed_utc=now(),selection=read(HERE/'selection.json'),primary=primary,
        primary_bootstrap_sensitivity=bootstrap,development=development,confirmation=confirmation,
        downstream=fresh,historical=historical,validation=read(HERE/'validation.json'))
    write(OUT/'summary.json',result)
    print(json.dumps(dict(primary=primary,bootstrap=bootstrap,downstream=fresh['contrasts'],flags=fresh['flagged_trajectories']),indent=2))

if __name__=='__main__':main()
