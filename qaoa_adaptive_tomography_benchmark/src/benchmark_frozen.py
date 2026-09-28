from pathlib import Path
import numpy as np
from benchmark_common import ROOT,read,write,digest
from benchmark_solver import solve,random_search,swap_search,basis
from adaptive_kraus.design import BinaryDesign
from adaptive_kraus.study_protocol import stable_seed

def model_from(meta):
    with np.load(ROOT/meta['path']) as z:
        model=BinaryDesign(z['linear'],z['pairs'],meta['b'],z['experiment_ids'])
        energies,gains=z['energies'],z['gains']
    return model,energies,gains

def score(result,meta,model,energies,gains):
    span=float(energies.max()-energies.min());denom=span if span>1e-12 else 1.
    _,bits,lookup,_=basis(meta['n'],meta['b'])
    expectation=float(result['expected_energy'])
    result['expected_normalized_regret']=(expectation-float(energies.min()))/denom if span>1e-12 else 0.
    result['expected_true_gain']=float(np.asarray(result['probabilities'])@gains)
    for count,x in result['selected'].items():
        integer=sum(int(v)<<i for i,v in enumerate(x['bits']));idx=lookup[integer]
        x['normalized_regret']=(x['energy']-float(energies.min()))/denom if span>1e-12 else 0.
        x['optimal']=bool(x['energy']<=energies.min()+1e-9)
        x['true_gain']=float(gains[idx]);x['shortlist_true_gain_regret']=float(gains.max()-gains[idx])
    return result

def run_sequence(job):
    split,metas,depth,cap,seed=job
    name=f"{metas[0]['id']}_n{metas[0]['n']}_b{metas[0]['b']}_p{depth}_e{cap}_s{seed}"
    path=ROOT/'results/frozen'/split/(name+'.json')
    if path.exists():return name,'cached'
    records=[];warm=None
    for meta in metas:
        model,energies,gains=model_from(meta)
        pairedseed=stable_seed(seed,meta['id'],meta['round'],meta['n'],meta['b'],depth,cap,'qaoa')
        for init in ['cold','transfer']:
            result=solve(model,depth,cap,pairedseed,warm if init=='transfer' else None)
            if init=='transfer':warm=result['angles']
            result=score(result,meta,model,energies,gains)
            records.append(dict(result,**{k:meta[k] for k in ['id','family','round','n','b']},depth=depth,cap=cap,solver_seed=seed,initialization=init))
    write(path,{'configuration':name,'records':records});return name,'complete'

def run_classical(job):
    split,meta,seeds=job
    path=ROOT/'results/classical'/split/(meta['key']+'.json')
    if path.exists():return meta['key'],'cached'
    model,energies,gains=model_from(meta);_,bits,lookup,_=basis(meta['n'],meta['b']);span=float(energies.max()-energies.min());denom=span if span>1e-12 else 1.
    rows=[]
    for seed in seeds:
        for method,budgets,solver in [('random',[8,32,256],random_search),('swap',[20,80,256],swap_search)]:
            for budget in budgets:
                q,info=solver(model,stable_seed(seed,meta['id'],meta['round'],meta['n'],meta['b'],'classical',method),budget)
                i=lookup[sum(int(v)<<j for j,v in enumerate(q))]
                rows.append(dict(info,method=method,budget=budget,solver_seed=seed,normalized_regret=(float(energies[i])-float(energies.min()))/denom,
                      optimal=bool(energies[i]<=energies.min()+1e-9),true_gain=float(gains[i]),shortlist_true_gain_regret=float(gains.max()-gains[i]),bits=q.astype(int).tolist(),
                      **{k:meta[k] for k in ['id','family','round','n','b']}))
    write(path,{'problem':meta['key'],'records':rows});return meta['key'],'complete'
