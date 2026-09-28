from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];A=B/'analysis';O=B/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'svg.fonttype':'none'})
COL={'cold':'#ce642c','transfer':'#147d87','random':'#9a8128','swap':'#786299'}
LABEL={'greedy':'Full-pool greedy','fixed_prior':'Fixed prior','exact_qubo':'Exact QUBO','random_qubo':'Random 256','swap':'Swap search','qaoa_cold':'QAOA cold','qaoa_transfer':'QAOA transfer'}
def save(name,fig):
 fig.tight_layout();fig.savefig(O/(name+'.png'),bbox_inches='tight');fig.savefig(O/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def ci(ax,x,r,color,label):
 ax.errorbar(x,r['mean'],yerr=[r['mean']-r['ci_low'],r['ci_high']-r['mean']],marker='o',capsize=3,color=color,label=label)
def forest(ax,data,labels,xlabel):
 for i,r in enumerate(data):ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='o',capsize=3,color='#147d87')
 ax.axvline(0,color='#777',lw=1);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.2)

s={'trajectory_configuration':json.loads((B/'protocol/TRAJECTORY_CONFIGURATION.json').read_text())};depth=s['trajectory_configuration']['depth'];cap=s['trajectory_configuration']['cap']
q=pd.read_csv(A/'qaoa_expected_summary.csv');q=q.query('b==3 and depth==1 and cap==20')
fig,ax=plt.subplots(figsize=(7.4,3.6))
for init in ['cold','transfer']:
 r=q[q.initialization==init].sort_values('n');ci(ax,r.n,r,COL[init],init.capitalize())
ax.set_xticks([8,10,12,14]);ax.set_xlabel('Shortlisted experiments and QAOA register qubits');ax.set_ylabel('Expected normalized energy regret');ax.set_ylim(bottom=0);ax.legend();ax.grid(alpha=.15);save('fig01_transfer_scaling',fig)

c=pd.read_csv(A/'transfer_contrasts.csv').query('n==14 and b==3').sort_values(['depth','cap'])
fig,ax=plt.subplots(figsize=(7.4,3.1));forest(ax,c.to_dict('records'),[f"Depth {r.depth}, cap {r.cap}"+('  [primary]' if r.primary else '') for r in c.itertuples()],'Transferred minus cold expected normalized regret');save('fig02_transfer_contrasts',fig)

sm=pd.read_csv(A/'sampled_summary.csv');cl=pd.read_csv(A/'classical_summary.csv');pairs=[(8,3),(10,3),(12,3),(14,3),(14,5)]
fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),sharey=True)
for ax,prefix in zip(axs,[32,256]):
 for init in ['cold','transfer']:
  r=sm.query('depth==@depth and cap==@cap and initialization==@init and prefix==@prefix').set_index(['n','b'])
  ax.plot(range(5),[r.loc[k].optimal_fraction for k in pairs],'-o',color=COL[init],label='QAOA '+init)
 for method,budget in [('random',prefix),('swap',256)]:
  r=cl.query('method==@method and budget==@budget').set_index(['n','b'])
  ax.plot(range(5),[r.loc[k].optimal_fraction for k in pairs],'--s',color=COL[method],label=f'{method.capitalize()} {budget}')
 ax.set_xticks(range(5),['8/3','10/3','12/3','14/3','14/5']);ax.set_xlabel('Candidates / selected');ax.set_title(f'Best of {prefix} QAOA samples',fontsize=10);ax.grid(alpha=.15);ax.set_ylim(-.03,1.05);ax.axvline(3.5,color='#999',ls=':',lw=.8)
axs[0].set_ylabel('Fraction attaining quadratic optimum');handles,_=axs[1].get_legend_handles_labels();fig.legend(handles,['QAOA cold','QAOA transfer','Random search','Swap search 256'],loc='upper center',bbox_to_anchor=(.5,-.04),ncol=4,fontsize=8,frameon=False);save('fig03_sampling_success',fig)

aud=pd.read_csv(A/'problem_audits.csv').query('b==3 and round>0');selection=pd.read_csv(A/'sampled_records.csv').query("split=='evaluation' and b==3 and round>0 and depth==@depth and cap==@cap and initialization=='transfer' and prefix==256")
mix=selection.merge(aud[['id','round','n','b','full_pool_optimum_gain','shortlist_optimum_gain','surrogate_optimum_gain']],on=['id','round','n','b'])
mix['Shortlist']=mix.full_pool_optimum_gain-mix.shortlist_optimum_gain
mix['Quadratic approximation']=mix.shortlist_optimum_gain-mix.surrogate_optimum_gain
mix['QAOA selection']=mix.surrogate_optimum_gain-mix.true_gain
fig,ax=plt.subplots(figsize=(7.4,3.4))
for metric,color in [('Shortlist','#626d7a'),('Quadratic approximation','#9a8128'),('QAOA selection','#ce642c')]:
 r=mix.groupby('n')[metric].mean();ax.plot(r.index,r.values,'-o',label=metric,color=color)
ax.axhline(0,color='#777',lw=.7);ax.set_xticks([8,10,12,14]);ax.set_xlabel('Shortlisted experiments');ax.set_ylabel('Mean loss in local information gain');ax.legend(fontsize=9);ax.grid(alpha=.15);save('fig04_approximation_losses',fig)

if '--frozen-only' in sys.argv:
 print('Rendered four frozen-state figures');raise SystemExit(0)

accuracy=pd.read_csv(A/'trajectory_accuracy.csv');order=list(LABEL);accuracy=accuracy.set_index('policy').loc[order].reset_index()
fig,ax=plt.subplots(figsize=(7.4,3.6));x=np.arange(len(accuracy));ax.bar(x,accuracy['mean'],color=['#616b78','#88599e','#4575b4','#b59132','#78a5c8','#ce642c','#147d87']);ax.errorbar(x,accuracy['mean'],yerr=[accuracy['mean']-accuracy.ci_low,accuracy.ci_high-accuracy['mean']],fmt='none',color='#333',capsize=3)
ax.set_xticks(x,[LABEL[p] for p in accuracy.policy],rotation=25,ha='right');ax.set_ylabel('Mean final held-out probability RMSE');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15);save('fig05_trajectory_accuracy',fig)

tc=pd.read_csv(A/'trajectory_contrasts.csv');rows=tc[tc['first']=='qaoa_transfer']
fig,ax=plt.subplots(figsize=(7.4,3.8));forest(ax,rows.to_dict('records'),['Transfer minus '+LABEL[x] for x in rows.second],'Paired final probability RMSE difference');save('fig06_trajectory_contrasts',fig)

r=pd.read_csv(A/'trajectory_resources.csv').set_index('policy').loc[order]
fig,axs=plt.subplots(1,2,figsize=(7.4,3.6));x=np.arange(7)
axs[0].bar(x,r.characterization_shots,color='#147d87');axs[0].set_ylabel('Mean characterization shots')
axs[1].bar(x,r.characterization_shots,color='#147d87',label='Characterization');axs[1].bar(x,r.objective_shots+r.sample_shots,bottom=r.characterization_shots,color='#ce642c',label='QAOA solver');axs[1].set_ylabel('Mean attempted shots');axs[1].legend(fontsize=8)
for ax in axs:ax.set_xticks(x,[LABEL[p].replace('Full-pool ','') for p in order],rotation=50,ha='right',fontsize=8);ax.grid(axis='y',alpha=.15)
save('fig07_resources',fig)

learning=pd.read_csv(A/'trajectory_learning.csv');fig,ax=plt.subplots(figsize=(7.4,3.5))
for policy in ['greedy','fixed_prior','qaoa_cold','qaoa_transfer','swap']:
 r=learning[learning.policy==policy].groupby('applications').rmse.mean();ax.plot(r.index,r.values,'-o',label=LABEL[policy],markersize=4)
ax.set_xticks([384,1608,2832,4056]);ax.set_xlabel('Unknown-channel applications');ax.set_ylabel('Mean held-out probability RMSE');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);save('fig08_learning',fig)
print('Rendered eight scientific figures')
