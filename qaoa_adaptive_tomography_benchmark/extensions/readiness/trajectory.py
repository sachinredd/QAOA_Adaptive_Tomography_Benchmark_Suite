"""Fresh paired adaptive branches with the archived Aer oracle and Kraus fitter."""
from common import *
from solver import run_solver
from benchmark_solver import swap_search, basis
from adaptive_kraus.channels import gksl_channel, blocks, channel_metrics
from adaptive_kraus.experiments import make_pool, heldout_pool, pilot_ids, AerOracle
from adaptive_kraus.manifold import random_isometry
from adaptive_kraus.multistart import fit_candidates
from adaptive_kraus.cost_budget import measure_selected, prospective_shots
from adaptive_kraus.study_execution import restore_oracle
from adaptive_kraus.design import build_design, make_binary_design, greedy_batch, batch_gain
from adaptive_kraus.study_diagnostics import stationarity, convergence_check, prediction_metrics
from dataclasses import asdict
from time import perf_counter

FIT = dict(iterations=2000, learning_rate=1., tolerance=1e-6)
READOUT = [.03,.06]

def run_trajectory(job):
    task, rep, policy = job
    protocol_guard()
    folder = HERE/'data'/'trajectories'/task['id']/f'rep_{rep}'/policy
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['sha256'] == digest(folder/'result.json')
        return task['id'],rep,policy,'cached'
    folder.mkdir(parents=True,exist_ok=True)
    seeds = {role:stable_seed(28092026,task['id'],rep,role) for role in ['acquisition','initialization','refit']}
    # Validation probes are common to repetitions, separating acquisition variance.
    seeds['validation'] = stable_seed(28092026,task['id'],'validation')
    write(folder/'task.json',dict(task,rep=rep,policy=policy,seeds=seeds))
    pool=make_pool(1,[1,2,4,6,8],READOUT)
    shots=prospective_shots(pool,{'applications_per_setting':408})
    reference=gksl_channel(**task['channel'])
    oracle=AerOracle(reference,READOUT,seeds['acquisition'])
    stack=random_isometry(2,4,seeds['initialization']); obs=[]; history=[]
    chosen=read(HERE/'selection.json')['selected_variant']
    for r in range(4):
        stage=folder/f'round_{r:03d}'
        if (stage/'complete.json').exists():
            for fn,h in read(stage/'complete.json').items():
                assert digest(stage/fn)==h
            stack=np.load(stage/'model.npz')['stack'];obs=observations(stage/'observations.json')
            history.append(read(stage/'row.json'));restore_oracle(oracle,obs,pool);continue
        if r==0 and policy!='greedy':
            import shutil
            src=folder.parent/'greedy'/'round_000'
            assert (src/'complete.json').exists()
            shutil.copytree(src,stage,dirs_exist_ok=True)
            stack=np.load(stage/'model.npz')['stack'];obs=observations(stage/'observations.json')
            history.append(read(stage/'row.json'));restore_oracle(oracle,obs,pool);continue
        started=perf_counter(); meta={'objective_shots':0,'final_shots':0,'policy':policy}
        if r==0:
            selected=pilot_ids(pool);newshots=32
        else:
            newshots=shots
            design=build_design(blocks(stack),pool,obs,shots,1.,1e-6)
            if policy=='greedy':
                selected=greedy_batch(design,3)
                meta['candidate_evaluations']=90+89+88
            else:
                model=make_binary_design(design,14,3)
                seed=stable_seed(28092101,task['id'],rep,r,'solver')
                if policy=='swap':
                    q,info=swap_search(model,seed,256);meta.update(info)
                elif policy in ('baseline','selected'):
                    variant='baseline' if policy=='baseline' else chosen
                    result=run_solver(model,seed,variant)
                    q=np.asarray(result['selected_bits'])
                    meta.update({k:result[k] for k in ['variant','objective_shots','final_shots','total_shots','objective_evaluations']})
                    write(stage/'solver.json',result)
                else: raise ValueError(policy)
                selected=model.experiment_ids[np.flatnonzero(q)].tolist()
                save_model(stage/'design.npz',linear=model.linear,pairs=model.pairs,experiment_ids=model.experiment_ids)
            # Own-model diagnostic; no truth or reference optimum drives selection.
            meta['local_gain']=batch_gain(design,selected)
        meta['selection_seconds']=perf_counter()-started
        obs.extend(measure_selected(oracle,pool,selected,newshots))
        fit,trials,candidates=fit_candidates(pool,obs,stack,FIT,3,seeds['initialization'],r)
        stack=fit.stack
        row=dict(round=r,selected=list(map(int,selected)),selection=meta,fit=fit.diagnostics,
                 stationarity=stationarity(pool,obs,stack),characterization_shots=oracle.measurement_shots,
                 applications=oracle.channel_applications)
        save_model(stage/'model.npz',stack=stack);save_model(stage/'candidate_models.npz',stacks=candidates)
        write(stage/'observations.json',[asdict(x) for x in obs]);write(stage/'fit_candidates.json',trials)
        write(stage/'row.json',row)
        write(stage/'complete.json',{fn:digest(stage/fn) for fn in ['model.npz','observations.json','fit_candidates.json','candidate_models.npz','row.json']})
        history.append(row)
    validation=heldout_pool(1,[3,5,9],8,seeds['validation'],READOUT)
    for row in history:
        saved=np.load(folder/f"round_{row['round']:03d}"/'model.npz')['stack']
        row['metrics']=prediction_metrics(saved,reference,validation,validation)
        row['channel_metrics']=channel_metrics(blocks(saved),reference)
    settings=dict(extra_iterations=400,restarts=1,gradient_tolerance=1e-4,nll_gain_tolerance=1e-5)
    check,refitted=convergence_check(pool,obs,stack,settings,FIT,seeds['refit'])
    save_model(folder/'offline_refit.npz',stack=refitted)
    result=dict(id=task['id'],family=task['family'],rep=rep,policy=policy,history=history,
        convergence=check,rmse=history[-1]['metrics']['heldout_rmse'],
        choi_distance=history[-1]['channel_metrics']['normalized_choi_trace_distance'],
        applications=oracle.channel_applications,characterization_shots=oracle.measurement_shots,
        objective_shots=sum(r['selection']['objective_shots'] for r in history),
        final_shots=sum(r['selection']['final_shots'] for r in history))
    assert result['applications']==4056
    write(folder/'result.json',result);write(folder/'complete.json',{'sha256':digest(folder/'result.json')})
    return task['id'],rep,policy,'complete'
