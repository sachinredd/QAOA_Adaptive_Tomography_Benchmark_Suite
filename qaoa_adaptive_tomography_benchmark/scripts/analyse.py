from pathlib import Path
import sys,json,math
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import t
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,write,digest,source_digest
O=B/'analysis';O.mkdir(exist_ok=True)

def interval(frame,value):
 g=frame.groupby('family')[value];means=g.mean();n=g.count();v=g.var(ddof=1)/n;f=len(means)
 estimate=float(means.mean());variance=float(v.sum()/f**2)
 df=float(v.sum()**2/((v*v/(n-1)).sum())) if v.sum()>0 else float(len(frame)-1)
 half=float(t.ppf(.975,df)*np.sqrt(variance))
 return dict(mean=estimate,se=float(np.sqrt(variance)),df=df,ci_low=estimate-half,ci_high=estimate+half,channels=len(frame))

def main(frozen_only=False):
 rows=[];sampled=[]
 for split,expected in [('development',2160),('evaluation',10800)]:
  start=len(rows)
  for path in sorted((B/'results/frozen'/split).glob('*.json')):
   for r in read(path)['records']:
    base={k:r[k] for k in ['id','family','round','n','b','depth','cap','solver_seed','initialization','expected_energy','expected_normalized_regret','expected_true_gain','objective_evaluations','objective_shots','sample_shots','total_shots','seconds']}
    base['split']=split;rows.append(base)
    for k,s in r['selected'].items():sampled.append(dict(base,prefix=int(k),**{x:s[x] for x in ['energy','normalized_regret','optimal','true_gain','shortlist_true_gain_regret']}))
  assert len(rows)-start==expected,(split,len(rows)-start)
 q=pd.DataFrame(rows);sm=pd.DataFrame(sampled);q.to_csv(O/'qaoa_records.csv',index=False);sm.to_csv(O/'sampled_records.csv',index=False)
 cs=[]
 for split,expected in [('development',1620),('evaluation',8100)]:
  before=len(cs)
  for path in sorted((B/'results/classical'/split).glob('*.json')):
   cs.extend(dict(r,split=split) for r in read(path)['records'])
  assert len(cs)-before==expected
 c=pd.DataFrame(cs);c.drop(columns='bits').to_csv(O/'classical_records.csv',index=False)
 e=q.query("split == 'evaluation' and round > 0")
 ch=e.groupby(['id','family','n','b','depth','cap','initialization']).expected_normalized_regret.mean().unstack('initialization').reset_index()
 ch['difference']=ch.transfer-ch.cold;ch.to_csv(O/'transfer_channel_differences.csv',index=False)
 contrasts=[]
 for key,group in ch.groupby(['n','b','depth','cap']):
  row=dict(zip(['n','b','depth','cap'],map(int,key)));row.update(interval(group,'difference'));row['primary']=key==(14,3,1,20);contrasts.append(row)
 pd.DataFrame(contrasts).to_csv(O/'transfer_contrasts.csv',index=False)
 means=[]
 for key,group in ch.groupby(['n','b','depth','cap']):
  for init in ['cold','transfer']:
   means.append(dict(zip(['n','b','depth','cap'],map(int,key)))|dict(initialization=init,**interval(group,init)))
 pd.DataFrame(means).to_csv(O/'qaoa_expected_summary.csv',index=False)
 sampled_summary=[]
 for key,group in sm.query("split == 'evaluation' and round > 0").groupby(['n','b','depth','cap','initialization','prefix']):
  cm=group.groupby(['id','family'])[['normalized_regret','optimal','shortlist_true_gain_regret']].mean().reset_index()
  row=dict(zip(['n','b','depth','cap','initialization','prefix'],key));row.update(interval(cm,'normalized_regret'));row['optimal_fraction']=float(cm.optimal.mean());row['mean_local_gain_regret']=float(cm.shortlist_true_gain_regret.mean());sampled_summary.append(row)
 pd.DataFrame(sampled_summary).to_csv(O/'sampled_summary.csv',index=False)
 classical_summary=[]
 for key,group in c.query("split == 'evaluation' and round > 0").groupby(['n','b','method','budget']):
  cm=group.groupby(['id','family'])[['normalized_regret','optimal','shortlist_true_gain_regret']].mean().reset_index()
  row=dict(zip(['n','b','method','budget'],key));row.update(interval(cm,'normalized_regret'));row['optimal_fraction']=float(cm.optimal.mean());row['mean_local_gain_regret']=float(cm.shortlist_true_gain_regret.mean());classical_summary.append(row)
 pd.DataFrame(classical_summary).to_csv(O/'classical_summary.csv',index=False)
 metadata=pd.DataFrame(read(B/'data/evaluation_problems.json'));metadata.to_csv(O/'problem_audits.csv',index=False)
 if frozen_only:
  write(O/'frozen_summary.json',{'primary':next(x for x in contrasts if x['primary']),'qaoa_optimizations':len(q),'classical_heuristic_runs':len(c),'status':'Frozen-state stage complete; adaptive trajectory stage still running'})
  print(json.dumps(read(O/'frozen_summary.json'),indent=2));return
 final=[];learning=[];flags=[]
 for task in read(B/'data/evaluation_channels.json'):
  for policy in read(B/'protocol/PROTOCOL.json')['trajectory_stage']['policies']:
   folder=B/'data/trajectories'/task['id']/policy;r=read(folder/'result.json')
   assert r['channel_applications']==4056
   row={k:r[k] for k in ['id','family','instance','policy','rmse','channel_applications','characterization_shots','objective_shots','sample_shots','gradient_evaluations']}
   row['total_shots']=row['characterization_shots']+row['objective_shots']+row['sample_shots']
   check=r['convergence'];row['stationary']=check['original_stationary'];row['material_refit']=check['material_refit_improvement'];row['refit_gain']=check['nll_improvement'];row['gradient_norm']=check['original']['gradient_norm'];final.append(row)
   for h in r['history']:learning.append(dict(id=r['id'],family=r['family'],policy=policy,round=h['round'],rmse=h['metrics']['heldout_rmse'],applications=h['channel_applications']))
   if not row['stationary'] or row['material_refit']:flags.append(row)
 final=pd.DataFrame(final);assert len(final)==210;final.to_csv(O/'trajectory_results.csv',index=False);pd.DataFrame(learning).to_csv(O/'trajectory_learning.csv',index=False);pd.DataFrame(flags,columns=final.columns).to_csv(O/'fitting_flags.csv',index=False)
 accuracy=[]
 for policy,group in final.groupby('policy'):
  accuracy.append(dict(policy=policy,**interval(group,'rmse')))
 pd.DataFrame(accuracy).to_csv(O/'trajectory_accuracy.csv',index=False)
 wide=final.pivot(index=['id','family'],columns='policy',values='rmse').reset_index();tc=[]
 for a,b in [('qaoa_transfer','qaoa_cold'),('qaoa_transfer','greedy'),('qaoa_cold','greedy'),('qaoa_transfer','exact_qubo'),('qaoa_transfer','swap'),('qaoa_transfer','random_qubo'),('qaoa_transfer','fixed_prior'),('greedy','fixed_prior'),('exact_qubo','greedy')]:
  d=wide[['id','family']].copy();d['difference']=wide[a]-wide[b];tc.append(dict(first=a,second=b,**interval(d,'difference')))
 pd.DataFrame(tc).to_csv(O/'trajectory_contrasts.csv',index=False)
 resources=final.groupby('policy')[['channel_applications','characterization_shots','objective_shots','sample_shots','total_shots','gradient_evaluations']].mean();resources.to_csv(O/'trajectory_resources.csv')
 summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'counts':{'development_channels':6,'fresh_channels':30,'frozen_checkpoints':108,'selection_problems':540,'qaoa_optimizations':len(q),'adaptive_qaoa_optimizations':180,'total_qaoa_optimizations':len(q)+180,'classical_heuristic_runs':len(c),'fresh_policy_trajectories':len(final),'logical_fit_candidates':3360,'distinct_acquisition_fit_candidates':2640,'offline_refit_branches':420},
          'primary':next(x for x in contrasts if x['primary']),'trajectory_configuration':read(B/'protocol/TRAJECTORY_CONFIGURATION.json'),
          'trajectory_accuracy':accuracy,'trajectory_contrasts':tc,'trajectory_resources':resources.reset_index().to_dict('records'),
          'fitting':{'stationarity_flags':int((~final.stationary).sum()),'material_refit_flags':int(final.material_refit.sum()),'distinct_flagged_trajectories':len(flags),'maximum_refit_gain':float(final.refit_gain.max())},
          'solver_validation':read(B/'results/solver_validation.json'),'protocol_sha256':digest(B/'protocol/PROTOCOL.json'),'source_sha256':source_digest()}
 write(O/'summary.json',summary)
 write(O/'frozen_summary.json',{'primary':summary['primary'],'qaoa_optimizations':len(q),'classical_heuristic_runs':len(c),'status':'Frozen-state and adaptive trajectory stages complete'})
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--frozen-only',action='store_true');args=parser.parse_args();main(args.frozen_only)
