from pathlib import Path
import sys,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,source_digest,digest
from benchmark_trajectory import run_trajectory
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['scaffold','remaining'],required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 channels=read(B/'data/evaluation_channels.json')
 manifest=read(B/'protocol/EXECUTION_MANIFEST.json')
 assert manifest['source_sha256']==source_digest(), 'Source changed; create a new experiment'
 assert manifest['protocol_sha256']==digest(B/'protocol/PROTOCOL.json'), 'Protocol changed'
 policies=['greedy'] if a.stage=='scaffold' else ['fixed_prior','exact_qubo','random_qubo','swap','qaoa_cold','qaoa_transfer']
 jobs=[(c,policy) for c in channels for policy in policies]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  futures=[ex.submit(run_trajectory,j) for j in jobs]
  for i,f in enumerate(as_completed(futures),1):print(i,'/',len(jobs),f.result(),flush=True)
