from pathlib import Path
import sys,argparse
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,write,save_model,digest
from benchmark_solver import basis
from adaptive_kraus.experiments import make_pool,Observation
from adaptive_kraus.channels import blocks
from adaptive_kraus.design import build_design,make_binary_design,greedy_batch,batch_gain
from adaptive_kraus.study_diagnostics import exact_gain_optimum

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','evaluation'],required=True);a=p.parse_args()
 protocol=read(B/'protocol/PROTOCOL.json')
 if a.split=='development':sources=read(B/'data/development_index.json')
 else:sources=[dict(x,folder=f"data/trajectories/{x['id']}/greedy") for x in read(B/'data/evaluation_channels.json')]
 index=[];pool=make_pool(1,[1,2,4,6,8],[.03,.06]);shots=np.array([408//e.steps for e in pool])
 for source in sources:
  for r in range(3):
   folder=B/source['folder']/f'round_{r:03d}'
   stack=np.load(folder/'model.npz')['stack'];obs=[Observation(x['experiment_id'],tuple(x['counts']),x['shots']) for x in read(folder/'observations.json')]
   design=build_design(blocks(stack),pool,obs,shots,1.,1e-6)
   full=exact_gain_optimum(design,range(90),3,150000)
   for n,b in protocol['selection_problems']:
    model=make_binary_design(design,n,b);_,bits,_,_=basis(n,b)
    energies=model.energy(bits);subsets=np.array([model.experiment_ids[np.flatnonzero(q)] for q in bits])
    matrices=np.eye(12)+design.whitened[subsets].sum(axis=1)
    lower=np.linalg.cholesky((matrices+matrices.swapaxes(-1,-2))/2)
    gains=2*np.log(np.diagonal(lower,axis1=-2,axis2=-1)).sum(axis=1)
    key=f"{source['id']}_r{r}_n{n}_b{b}";path=B/'data/problems'/a.split/(key+'.npz')
    save_model(path,linear=model.linear,pairs=model.pairs,experiment_ids=model.experiment_ids,batch_size=np.array(b),energies=energies,gains=gains)
    eq=int(np.argmin(energies));ig=int(np.argmax(gains));greedy=greedy_batch(design,b)
    metadata={'id':source['id'],'family':source['family'],'round':r,'n':n,'b':b,'key':key,'path':str(path.relative_to(B)),
      'minimum_energy':float(energies.min()),'maximum_energy':float(energies.max()),'shortlist_optimum_gain':float(gains[ig]),
      'surrogate_optimum_gain':float(gains[eq]),'full_pool_greedy_gain':batch_gain(design,greedy),
      'full_pool_optimum_gain':full['gain'] if b==3 else None,'feasible_batches':len(bits),
      'source_model_sha256':digest(folder/'model.npz'),'source_observations_sha256':digest(folder/'observations.json'),'problem_sha256':digest(path)}
    index.append(metadata)
  print('Prepared',a.split,source['id'],flush=True)
 write(B/f'data/{a.split}_problems.json',index)
