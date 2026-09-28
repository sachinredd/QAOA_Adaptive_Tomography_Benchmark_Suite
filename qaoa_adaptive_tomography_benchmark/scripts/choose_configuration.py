from pathlib import Path
import sys,json
from datetime import datetime,timezone
import pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from benchmark_common import read,write,digest
records=[r for p in sorted((B/'results/frozen/development').glob('*.json')) for r in read(p)['records']]
assert len(records)==2160
df=pd.DataFrame(records);subset=df.query('n == 14 and b == 3 and round > 0')
scores=subset.groupby(['depth','cap']).expected_normalized_regret.mean().reset_index();minimum=scores.expected_normalized_regret.min()
selected=scores[scores.expected_normalized_regret<=minimum+1e-12].sort_values(['cap','depth']).iloc[0]
record={'depth':int(selected.depth),'cap':int(selected.cap),'shortlist_size':14,'batch_size':3,'locked_utc':datetime.now(timezone.utc).isoformat(),
        'development_scores':scores.to_dict('records'),'rule':'Lowest pooled cold/transfer development mean expected normalized energy regret; ties favor lower cap then depth.','evaluation_outcomes_used':False,
        'development_records':len(df),'protocol_sha256':digest(B/'protocol/PROTOCOL.json')}
path=B/'protocol/TRAJECTORY_CONFIGURATION.json'
if path.exists():raise SystemExit('Configuration already locked')
write(path,record);print(json.dumps(record,indent=2))
