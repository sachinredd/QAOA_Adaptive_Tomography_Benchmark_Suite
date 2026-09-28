from pathlib import Path
import sys,argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,source_digest,digest
from benchmark_frozen import run_sequence,run_classical
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','evaluation'],required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 protocol=read(B/'protocol/PROTOCOL.json');metas=read(B/f'data/{a.split}_problems.json');groups=defaultdict(list)
 manifest=read(B/'protocol/EXECUTION_MANIFEST.json')
 assert manifest['source_sha256']==source_digest(), 'Source changed; create a new experiment'
 assert manifest['protocol_sha256']==digest(B/'protocol/PROTOCOL.json'), 'Protocol changed'
 for m in metas:groups[(m['id'],m['n'],m['b'])].append(m)
 jobs=[(a.split,sorted(ms,key=lambda x:x['round']),d,c,s) for ms in groups.values() for d in [1,2] for c in [20,80] for s in protocol['evaluation']['solver_seeds']]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(run_sequence,j) for j in jobs]
  for i,f in enumerate(as_completed(fs),1):
   result=f.result()
   if i%20==0 or i==len(fs):print('QAOA sequences',i,'/',len(fs),result,flush=True)
  fs=[ex.submit(run_classical,(a.split,m,protocol['evaluation']['solver_seeds'])) for m in metas]
  for i,f in enumerate(as_completed(fs),1):
   result=f.result()
   if i%50==0 or i==len(fs):print('Classical problems',i,'/',len(fs),result,flush=True)
