"""Fresh paired adaptive trajectories using the unchanged archived fitter."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from benchmark_common import ROOT,read,write,save_model,digest
from benchmark_solver import solve,random_search,swap_search,basis
from adaptive_kraus.channels import gksl_channel,blocks
from adaptive_kraus.experiments import make_pool,pilot_ids,heldout_pool,AerOracle,Observation
from adaptive_kraus.manifold import random_isometry
from adaptive_kraus.multistart import fit_candidates
from adaptive_kraus.cost_budget import measure_selected,prospective_shots
from adaptive_kraus.study_execution import restore_oracle
from adaptive_kraus.design import build_design,make_binary_design,greedy_batch
from adaptive_kraus.study_protocol import stable_seed
from adaptive_kraus.study_diagnostics import stationarity,convergence_check,prediction_metrics

FIT={'iterations':2000,'learning_rate':1.,'tolerance':1e-6}
READOUT=[.03,.06]

def run_trajectory(job):
    task,policy=job
    folder=ROOT/'data/trajectories'/task['id']/policy
    if (folder/'complete.json').exists():
        if read(folder/'complete.json')['result_sha256']!=digest(folder/'result.json'):raise ValueError('Corrupted completed trajectory')
        return task['id'],policy,'cached'
    folder.mkdir(parents=True,exist_ok=True);write(folder/'task.json',dict(task,policy=policy))
    pool=make_pool(1,[1,2,4,6,8],READOUT);shots=prospective_shots(pool,{'applications_per_setting':408})
    reference=gksl_channel(**task['channel']);oracle=AerOracle(reference,READOUT,task['seeds']['acquisition'])
    initial=random_isometry(2,4,task['seeds']['initialization']);stack=initial
    obs=[];history=[];warm=None
    configuration=read(ROOT/'protocol/TRAJECTORY_CONFIGURATION.json') if policy.startswith('qaoa') else None
    for r in range(4):
        stage=folder/f'round_{r:03d}'
        if (stage/'complete.json').exists():
            mark=read(stage/'complete.json')
            for fn,checksum in mark.items():
                if digest(stage/fn)!=checksum:raise ValueError('Corrupted round '+str(stage))
            stack=np.load(stage/'model.npz')['stack']
            obs=[Observation(x['experiment_id'],tuple(x['counts']),x['shots']) for x in read(stage/'observations.json')]
            row=read(stage/'row.json');history.append(row)
            warm=row['selection'].get('angles',warm)
            restore_oracle(oracle,obs,pool)
            continue
        # The common pilot is physically sampled/fitted once and reused in paired branches.
        if r==0 and policy!='greedy':
            src=ROOT/'data/trajectories'/task['id']/'greedy'/'round_000'
            if not (src/'complete.json').exists():raise RuntimeError('Finish greedy common pilot first')
            import shutil
            shutil.copytree(src,stage,dirs_exist_ok=True)
            row=read(stage/'row.json');row['shared_pilot_from']='greedy'
            write(stage/'row.json',row)
            write(stage/'complete.json',{fn:digest(stage/fn) for fn in ['model.npz','observations.json','row.json','candidate_models.npz','fit_candidates.json']})
            stack=np.load(stage/'model.npz')['stack'];obs=[Observation(x['experiment_id'],tuple(x['counts']),x['shots']) for x in read(stage/'observations.json')]
            restore_oracle(oracle,obs,pool);history.append(row);continue
        started=perf_counter()
        selection={'solver':policy,'objective_shots':0,'sample_shots':0}
        if r==0:selected=pilot_ids(pool);newshots=32;selection['solver']='common_pilot'
        else:
            newshots=shots
            if policy=='fixed_prior':selected=read(ROOT/'protocol/FIXED_PRIOR_DESIGN.json')['schedules']['calibrated_readout'][r-1]
            else:
                design=build_design(blocks(stack),pool,obs,shots,1.,1e-6)
                if policy=='greedy':selected=greedy_batch(design,3)
                else:
                    model=make_binary_design(design,14,3)
                    seed=stable_seed(7101,task['id'],r,'solver')
                    if policy=='exact_qubo':
                        _,bits,_,_=basis(14,3);energies=model.energy(bits);q=bits[np.argmin(energies)]
                        selection['candidate_evaluations']=len(bits)
                    elif policy=='random_qubo':q,meta=random_search(model,seed,256);selection.update(meta)
                    elif policy=='swap':q,meta=swap_search(model,seed,256);selection.update(meta)
                    elif policy.startswith('qaoa'):
                        result=solve(model,configuration['depth'],configuration['cap'],seed,warm if policy=='qaoa_transfer' else None)
                        q=np.asarray(result['selected']['256']['bits']);warm=result['angles']
                        selection.update({k:result[k] for k in ['angles','start_angles','initial_bits','transferred','objective_evaluations','objective_shots','sample_shots','total_shots','seconds']})
                        write(stage/'solver.json',result)
                    else:raise ValueError(policy)
                    selected=model.experiment_ids[np.flatnonzero(q)].tolist()
                    save_model(stage/'design.npz',linear=model.linear,pairs=model.pairs,experiment_ids=model.experiment_ids)
        selection['selection_seconds']=perf_counter()-started
        observations=measure_selected(oracle,pool,selected,newshots);obs.extend(observations)
        fit,trials,candidates=fit_candidates(pool,obs,stack,FIT,3,task['seeds']['initialization'],r);stack=fit.stack
        check=stationarity(pool,obs,stack)
        row={'round':r,'selected':list(map(int,selected)),'selection':selection,'fit':fit.diagnostics,'stationarity':check,
             'characterization_shots':oracle.measurement_shots,'channel_applications':oracle.channel_applications}
        save_model(stage/'model.npz',stack=stack);save_model(stage/'candidate_models.npz',stacks=candidates)
        write(stage/'fit_candidates.json',trials);write(stage/'observations.json',[asdict(x) for x in obs]);write(stage/'row.json',row)
        write(stage/'complete.json',{fn:digest(stage/fn) for fn in ['model.npz','observations.json','row.json','candidate_models.npz','fit_candidates.json']})
        history.append(row)
    validation=heldout_pool(1,[3,5,9],8,task['seeds']['validation'],READOUT)
    for row in history:
        s=np.load(folder/f"round_{row['round']:03d}"/'model.npz')['stack']
        row['metrics']=prediction_metrics(s,reference,validation,validation)
    settings={'extra_iterations':400,'restarts':1,'gradient_tolerance':1e-4,'nll_gain_tolerance':1e-5}
    check,refitted=convergence_check(pool,obs,stack,settings,FIT,task['seeds']['refit'])
    save_model(folder/'offline_refit.npz',stack=refitted)
    result={'id':task['id'],'family':task['family'],'instance':task['instance'],'policy':policy,'history':history,'convergence':check,
            'rmse':history[-1]['metrics']['heldout_rmse'],'channel_applications':oracle.channel_applications,
            'characterization_shots':oracle.measurement_shots,'objective_shots':sum(x['selection']['objective_shots'] for x in history),
            'sample_shots':sum(x['selection']['sample_shots'] for x in history),
            'gradient_evaluations':sum(x['fit']['gradient_evaluations'] for x in history)}
    assert oracle.channel_applications==4056
    write(folder/'result.json',result);write(folder/'complete.json',{'result_sha256':digest(folder/'result.json')})
    return task['id'],policy,'complete'
