from pathlib import Path
import sys,json
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,write,digest,source_digest
from benchmark_solver import basis,Circuit
from benchmark_frozen import model_from
from adaptive_kraus.experiments import make_pool,Observation,heldout_pool,predict_probabilities
from adaptive_kraus.channels import gksl_channel,blocks
from adaptive_kraus.learning import CountObjective
from adaptive_kraus.study_diagnostics import stationarity

def main():
 manifest=read(B/'protocol/EXECUTION_MANIFEST.json')
 assert source_digest()==manifest['source_sha256']
 assert digest(B/'protocol/PROTOCOL.json')==manifest['protocol_sha256']
 meta={split:{(m['id'],m['round'],m['n'],m['b']):m for m in read(B/f'data/{split}_problems.json')} for split in ['development','evaluation']}
 max_energy_error=0.;max_expected_error=0.;max_transfer_error=0.;qcount=0
 for split,expected in [('development',2160),('evaluation',10800)]:
  count=0
  for path in sorted((B/'results/frozen'/split).glob('*.json')):
   records=read(path)['records'];assert len(records)==6
   previous=None
   for r in records:
    m=meta[split][(r['id'],r['round'],r['n'],r['b'])];model,energies,gains=model_from(m)
    _,bits,lookup,_=basis(r['n'],r['b'])
    assert sum(r['initial_bits'])==r['b']
    if r['initialization']=='transfer':
     if r['round']>0:
      error=float(abs(np.asarray(r['start_angles'])-previous).max());max_transfer_error=max(max_transfer_error,error);assert error==0.
     previous=np.asarray(r['angles'])
    probability=np.asarray(r['probabilities']);assert abs(probability.sum()-1)<1e-12 and np.min(probability)>=0
    err=abs(float(probability@energies)-r['expected_energy']);max_expected_error=max(max_expected_error,err);assert err<1e-10
    span=float(energies.max()-energies.min())
    normalized=(float(probability@energies)-float(energies.min()))/span if span>1e-12 else 0.
    assert abs(normalized-r['expected_normalized_regret'])<1e-10
    assert r['objective_evaluations']==len(r['trace']) and r['objective_evaluations']<=r['cap']
    assert r['objective_shots']==128*r['objective_evaluations'] and r['total_shots']==r['objective_shots']+256
    draws=r['final_sample_integers'];assert len(draws)==256
    for number,selection in r['selected'].items():
     integer=sum(int(v)<<i for i,v in enumerate(selection['bits']));assert integer in draws[:int(number)]
     assert sum(selection['bits'])==r['b']
     best=min(energies[lookup[int(x)]] for x in draws[:int(number)])
     err=abs(best-selection['energy']);max_energy_error=max(max_energy_error,err);assert err<1e-10
    count+=1
   cold0,transfer0=records[:2]
   assert cold0['angles']==transfer0['angles'] and cold0['final_sample_integers']==transfer0['final_sample_integers']
  assert count==expected; qcount+=count
  print('Verified QAOA',split,count,flush=True)
 pool=make_pool(1,[1,2,4,6,8],[.03,.06]);max_loss_error=0.;max_tp=0.;candidate_count=0;round_count=0;swap_matches=0
 policies=read(B/'protocol/PROTOCOL.json')['trajectory_stage']['policies']
 for task in read(B/'data/evaluation_channels.json'):
  pilot_hash=None
  for policy in policies:
   folder=B/'data/trajectories'/task['id']/policy
   assert read(folder/'complete.json')['result_sha256']==digest(folder/'result.json')
   result=read(folder/'result.json');assert result['channel_applications']==4056
   applications=0;shots=0
   for r in range(4):
    stage=folder/f'round_{r:03d}'
    for fn,checksum in read(stage/'complete.json').items():assert digest(stage/fn)==checksum
    obs=[Observation(x['experiment_id'],tuple(x['counts']),x['shots']) for x in read(stage/'observations.json')]
    if r==0:
     h=digest(stage/'observations.json')
     if pilot_hash is None:pilot_hash=h
     assert h==pilot_hash
    objective=CountObjective(pool,obs);candidates=np.load(stage/'candidate_models.npz')['stacks'];stack=np.load(stage/'model.npz')['stack'];trials=read(stage/'fit_candidates.json');row=read(stage/'row.json')
    losses=[objective.value(x) for x in candidates]
    for loss,trial,model in zip(losses,trials,candidates):
     err=abs(loss-trial['diagnostics']['nll']);max_loss_error=max(max_loss_error,err);assert err<1e-10
     tp=float(np.linalg.norm(model.conj().T@model-np.eye(2)));max_tp=max(max_tp,tp);assert tp<1e-8
    winner=int(np.argmin(losses));assert row['fit']['multistart']['selected_candidate']==winner
    assert np.array_equal(stack,candidates[winner]);candidate_count+=4;round_count+=1
    shots=sum(o.shots for o in obs);applications=sum(o.shots*pool[o.experiment_id].steps for o in obs)
    assert shots==row['characterization_shots'] and applications==row['channel_applications']
   assert applications==4056
   validation=heldout_pool(1,[3,5,9],8,task['seeds']['validation'],[.03,.06])
   truth=predict_probabilities(gksl_channel(**task['channel']),validation)
   prediction=predict_probabilities(blocks(stack),validation)
   rmse=float(np.sqrt(np.mean((prediction-truth)**2)))
   assert abs(rmse-result['rmse'])<1e-12
   if policy=='swap':
    for r in [1,2,3]:
     exact=read(B/'data/trajectories'/task['id']/'exact_qubo'/f'round_{r:03d}'/'row.json');local=read(folder/f'round_{r:03d}'/'row.json')
     swap_matches+=int(sorted(exact['selected'])==sorted(local['selected']))
   if policy=='qaoa_transfer':
    cold=read(B/'data/trajectories'/task['id']/'qaoa_cold'/'round_001'/'solver.json')
    transferred=read(folder/'round_001'/'solver.json')
    assert cold['angles']==transferred['angles'] and cold['final_sample_integers']==transferred['final_sample_integers']
    for r in [2,3]:
     now=read(folder/f'round_{r:03d}'/'solver.json');before=read(folder/f'round_{r-1:03d}'/'solver.json')
     assert now['start_angles']==before['angles']
  print('Verified paired trajectories',task['id'],flush=True)
 # Independent raw-record reconstruction of the primary paired estimate.
 from scipy.stats import t
 paired={}
 for path in sorted((B/'results/frozen/evaluation').glob('*_n14_b3_p1_e20_*.json')):
  for r in read(path)['records']:
   if r['round']>0:
    key=(r['family'],r['id'],r['round'],r['solver_seed']);paired.setdefault(key,{})[r['initialization']]=r['expected_normalized_regret']
 assert len(paired)==180
 channel={}
 for (family,identifier,_,_),item in paired.items():channel.setdefault((family,identifier),[]).append(item['transfer']-item['cold'])
 assert len(channel)==30 and all(len(x)==6 for x in channel.values())
 families=sorted(set(k[0] for k in channel));array=np.array([[np.mean(v) for k,v in sorted(channel.items()) if k[0]==fam] for fam in families]);assert array.shape==(6,5)
 mean=float(array.mean());variances=np.var(array,axis=1,ddof=1)/5;se=float(np.sqrt(variances.sum()/36));df=float(variances.sum()**2/np.sum(variances**2/4));half=float(t.ppf(.975,df)*se)
 expected=read(B/'analysis/summary.json')['primary']
 for key,value in [('mean',mean),('se',se),('df',df),('ci_low',mean-half),('ci_high',mean+half)]:assert abs(expected[key]-value)<1e-10
 record={'qaoa_records':qcount,'shared_round_zero_streams':'identical cold and transfer','maximum_transfer_angle_error':max_transfer_error,'maximum_sampled_energy_error':max_energy_error,'maximum_expected_energy_error':max_expected_error,
  'trajectory_round_records':round_count,'logical_candidates_recomputed':candidate_count,'distinct_candidates':2640,'maximum_candidate_loss_error':max_loss_error,'maximum_trace_preservation_error':max_tp,'paired_pilot_counts':'identical for every channel','source_and_protocol_hashes':'matched','raw_primary_recalculation':'passed','primary_independent_channels':len(channel),'swap_matches_exact_adaptive_batches':swap_matches,'status':'passed'}
 write(B/'results/verification.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':main()
