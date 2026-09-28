"""Post hoc mechanism diagnostics; not the prespecified primary analysis."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read
from benchmark_frozen import model_from
metas={(m['id'],m['round'],m['n'],m['b']):m for m in read(B/'data/evaluation_problems.json')};rows=[]
for path in sorted((B/'results/frozen/evaluation').glob('*.json')):
 for r in read(path)['records']:
  if r['round']==0:continue
  m=metas[(r['id'],r['round'],r['n'],r['b'])];model,energies,gains=model_from(m)
  p=np.asarray(r['probabilities']);optimal=energies<=energies.min()+1e-9;mass=float(p[optimal].sum())
  positive=p[p>0];entropy=float(-np.sum(positive*np.log(positive)));span=float(energies.max()-energies.min())
  rows.append({k:r[k] for k in ['id','family','round','n','b','depth','cap','initialization','solver_seed']}|dict(
    optimum_mass=mass,predicted_best256_success=float(-np.expm1(256*np.log1p(-mass))) if mass<1 else 1.,
    uniform_optimum_mass=float(optimal.mean()),effective_support=float(np.exp(entropy)),feasible_batches=len(p),
    support_fraction=float(np.mean(p>1e-14)),uniform_expected_normalized_regret=float((energies.mean()-energies.min())/span) if span>1e-12 else 0.,
    qaoa_expected_normalized_regret=r['expected_normalized_regret']))
frame=pd.DataFrame(rows);frame.to_csv(B/'analysis/posthoc_distribution_records.csv',index=False)
frame.groupby(['n','b','depth','cap','initialization']).mean(numeric_only=True).drop(columns=['round','solver_seed']).to_csv(B/'analysis/posthoc_distribution_summary.csv')
print('Post hoc distribution diagnostics complete',len(frame))
