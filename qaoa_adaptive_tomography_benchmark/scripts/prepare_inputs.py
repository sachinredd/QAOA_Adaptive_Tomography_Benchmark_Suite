from pathlib import Path
import sys,json,zipfile,shutil,hashlib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import write,read,digest
from adaptive_kraus.study_sampling import broad_parameters
from adaptive_kraus.study_protocol import stable_seed
protocol=read(B/'protocol/PROTOCOL.json')
archive=B.parent/'thesis_sources/Adaptive_Kraus_Matched_Cost_Suite.zip'
prefix='adaptive_kraus_matched_cost/examples/study_matched_cost/'
development=[]
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read(prefix+'manifest.json'))
 for task in manifest['tasks']:
  if task['policy']!='greedy' or task['replicate']!=4001 or task['case']['instance']!=0 or task['case']['noise']!='calibrated_readout':continue
  key=task['case']['family'];folder=B/'data/development'/key
  write(folder/'task.json',task)
  for r in range(3):
   for name in ['model.npz','observations.json','row.json','complete.json']:
    path=folder/f'round_{r:03d}'/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(z.read(prefix+'tasks/'+task['id']+f'/round_{r:03d}/'+name))
  development.append({'id':key,'family':key,'folder':str(folder.relative_to(B)),'source_task':task['id']})
assert len(development)==6
write(B/'data/development_index.json',development)
prior=B.parent/'thesis_work/package/data/matched_cost/FIXED_PRIOR_DESIGN.json'
shutil.copy2(prior,B/'protocol/FIXED_PRIOR_DESIGN.json')
evaluation=[]
for family in protocol['evaluation']['families']:
 for instance in range(5):
  key=f'{family}_{instance:02d}'
  task={'id':key,'family':family,'instance':instance,'channel':broad_parameters(family,instance,1.,protocol['evaluation']['channel_seed']),
        'seeds':{name:stable_seed(protocol['evaluation']['acquisition_seed'],family,instance,name) for name in ['acquisition','initialization','validation','refit']}}
  evaluation.append(task)
write(B/'data/evaluation_channels.json',evaluation)
write(B/'protocol/INPUT_PROVENANCE.json',{'archive':archive.name,'archive_sha256':digest(archive),'source_manifest':manifest['fingerprint'],'fixed_prior_sha256':digest(prior),'development_checkpoints':18,'fresh_channels':30,'protocol_sha256':digest(B/'protocol/PROTOCOL.json')})
print('Prepared 18 development checkpoints and 30 fresh channel specifications')
