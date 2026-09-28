"""Rebuild primary outcomes, counts, fit choices, and channel scores from saved evidence."""
from common import *
from solver import make_circuit
from benchmark_solver import basis
from adaptive_kraus.design import BinaryDesign
from adaptive_kraus.experiments import make_pool, heldout_pool, predict_probabilities
from adaptive_kraus.channels import blocks, gksl_channel, channel_metrics
from adaptive_kraus.learning import CountObjective
from scipy.stats import t
import collections

def main():
    protocol_guard();chosen=read(HERE/'selection.json')['selected_variant']
    tasks=read(HERE/'data/channels.json');n=len(tasks)
    frozen_counts={};max_probability_error=0.;primary=collections.defaultdict(lambda:collections.defaultdict(list))
    for split,expected in [('development',1728),('confirmation',n*2*2*3*2)]:
        count=0
        for path in sorted((HERE/'results'/split).glob('*.json')):
            doc=read(path);assert doc['protocol_sha256']==digest(HERE/'protocol.json')
            for r in doc['records']:
                problem=(BASE if split=='development' else HERE)/r['problem']
                with np.load(problem) as z:m=BinaryDesign(z['linear'],z['pairs'],3,z['experiment_ids'])
                c=make_circuit(m,np.asarray(r['initial_bits']),r['variant'])
                p=c.probabilities(r['angles']);err=float(np.max(abs(p-r['probabilities'])))
                max_probability_error=max(max_probability_error,err);assert err<1e-11
                lookup={int(v):i for i,v in enumerate(c.integers)}
                indices=[lookup[v] for v in r['final_sample_integers']]
                winner=min(set(indices),key=lambda i:(c.energy[i],i))
                assert c.bits[winner].astype(int).tolist()==r['selected_bits']
                regret=float((c.energy[winner]-c.energy.min())/np.ptp(c.energy))
                assert abs(regret-r['sampled_regret'])<1e-12
                assert len(indices)==256 and r['objective_evaluations']==len(r['trace'])<=80
                assert r['objective_shots']==(0 if r['exact_training'] else 128*len(r['trace']))
                assert r['total_shots']==r['objective_shots']+256
                if split=='confirmation':
                    assert not r['exact_training'];primary[(r['id'],r['family'])][r['variant']].append(regret)
                count+=1
        assert count==expected,(split,count,expected);frozen_counts[split]=count
        print('Verified frozen',split,count,flush=True)
    differences={}
    for key,g in primary.items():
        assert len(g['baseline'])==len(g[chosen])==12
        differences[key]=float(np.mean(g[chosen])-np.mean(g['baseline']))
    family_values=collections.defaultdict(list)
    for (_,fam),v in differences.items():family_values[fam].append(v)
    means=np.array([np.mean(x) for x in family_values.values()])
    ns=np.array([len(x) for x in family_values.values()])
    v=np.array([np.var(x,ddof=1)/len(x) for x in family_values.values()])
    df=float(v.sum()**2/np.sum(v*v/(ns-1)));se=float(np.sqrt(v.sum())/6)
    mean=float(means.mean());half=float(t.ppf(.975,df)*se)
    saved=read(HERE/'analysis/summary.json')['primary']
    for name,value in [('mean',mean),('se',se),('low',mean-half),('high',mean+half)]:
        assert abs(saved[name]-value)<1e-12
    pool=make_pool(1,[1,2,4,6,8],[.03,.06]);candidate_count=0;round_count=0;trajectory_count=0
    max_loss=0.;max_tp=0.;max_rmse=0.;max_choi=0.
    for task in tasks:
        reference=gksl_channel(**task['channel'])
        for rep in [0,1]:
            pilots=[]
            for policy in ['greedy','swap','baseline','selected']:
                folder=HERE/'data/trajectories'/task['id']/f'rep_{rep}'/policy
                assert read(folder/'complete.json')['sha256']==digest(folder/'result.json')
                result=read(folder/'result.json');savedtask=read(folder/'task.json')
                for r in range(4):
                    stage=folder/f'round_{r:03d}'
                    for fn,h in read(stage/'complete.json').items():assert digest(stage/fn)==h
                    obs=observations(stage/'observations.json');row=read(stage/'row.json')
                    shots=sum(o.shots for o in obs);apps=sum(o.shots*pool[o.experiment_id].steps for o in obs)
                    assert shots==row['characterization_shots'] and apps==row['applications']==384+1224*r
                    if r==0:pilots.append((digest(stage/'observations.json'),digest(stage/'model.npz')))
                    candidates=np.load(stage/'candidate_models.npz')['stacks'];stack=np.load(stage/'model.npz')['stack']
                    obj=CountObjective(pool,obs);losses=[obj.value(s) for s in candidates]
                    trials=read(stage/'fit_candidates.json')
                    for s,loss,tr in zip(candidates,losses,trials):
                        err=abs(loss-tr['diagnostics']['nll']);max_loss=max(max_loss,err);assert err<1e-10
                        tp=float(np.linalg.norm(s.conj().T@s-np.eye(2)));max_tp=max(max_tp,tp);assert tp<1e-8
                    winner=int(np.argmin(losses))
                    assert row['fit']['multistart']['selected_candidate']==winner and np.array_equal(stack,candidates[winner])
                    candidate_count+=len(candidates);round_count+=1
                validation=heldout_pool(1,[3,5,9],8,savedtask['seeds']['validation'],[.03,.06])
                rmse=float(np.sqrt(np.mean((predict_probabilities(blocks(stack),validation)-predict_probabilities(reference,validation))**2)))
                choi=channel_metrics(blocks(stack),reference)['normalized_choi_trace_distance']
                max_rmse=max(max_rmse,abs(rmse-result['rmse']));max_choi=max(max_choi,abs(choi-result['choi_distance']))
                assert max_rmse<1e-12 and max_choi<1e-12
                assert result['applications']==4056
                trajectory_count+=1
            assert len(set(pilots))==1
        print('Verified paired channel',task['id'],flush=True)
    assert trajectory_count==n*2*4
    oldparams=[x['channel'] for x in read(BASE/'data/evaluation_channels.json')]
    assert all(x['channel'] not in oldparams for x in tasks)
    out=dict(status='passed',utc=now(),frozen_records=frozen_counts,
        maximum_circuit_probability_error=max_probability_error,primary_raw_recalculation='passed',
        primary=dict(mean=mean,low=mean-half,high=mean+half,channels=n),
        trajectories=trajectory_count,rounds=round_count,logical_candidates=candidate_count,
        maximum_likelihood_error=max_loss,maximum_TP_residual=max_tp,
        maximum_RMSE_error=max_rmse,maximum_Choi_distance_error=max_choi,
        paired_pilot_models_and_counts='identical',fresh_parameter_overlap='none',
        original_source_and_extension_protocol='matched')
    write(HERE/'verification.json',out)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
