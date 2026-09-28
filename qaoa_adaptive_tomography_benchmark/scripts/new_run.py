"""Create a clean replication directory without deleting existing results."""
from pathlib import Path
import argparse,shutil,json
from datetime import datetime,timezone
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
for name in ['src','scripts','tests','protocol']:
 shutil.copytree(B/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','TRAJECTORY_CONFIGURATION.json'))
for name in ['data','results','analysis','figures','docs']:(out/name).mkdir(exist_ok=True)
shutil.copy2(B/'docs/report_template.md',out/'docs/report_template.md')
shutil.copytree(B/'data/development',out/'data/development')
for name in ['development_index.json','evaluation_channels.json']:shutil.copy2(B/'data'/name,out/'data'/name)
for name in ['README.md','requirements.txt']:
 if (B/name).exists():shutil.copy2(B/name,out/name)
(out/'protocol/REPLICATION.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Clean replication of the frozen benchmark; original results are retained in the source package.'},indent=2)+'\n')
print(out)
