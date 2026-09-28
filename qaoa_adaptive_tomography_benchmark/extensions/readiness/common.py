"""Utilities for the additive, separately frozen write-up readiness study."""
from pathlib import Path
import sys, json, hashlib, os
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'src'))
from benchmark_common import read, write, digest, save_model, source_digest
from adaptive_kraus.study_protocol import stable_seed, FAMILIES

def now():
    return datetime.now(timezone.utc).isoformat()

def code_hash():
    h = hashlib.sha256()
    for p in sorted(HERE.glob('*.py')):
        if p.name in ('report.py', 'analyse.py', 'verify.py'):
            continue  # analysis cannot alter acquisition; separately hashed at release
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def protocol_guard():
    p = read(HERE / 'protocol.json')
    assert p['archived_source_sha256'] == source_digest(), 'Archived scientific source changed'
    assert p['extension_code_sha256'] == code_hash(), 'Extension execution code changed'
    return p

def observations(path):
    from adaptive_kraus.experiments import Observation
    return [Observation(x['experiment_id'], tuple(x['counts']), x['shots']) for x in read(path)]

def interval(frame, value):
    """Equal-family stratified interval; input has one row per physical channel."""
    from scipy.stats import t
    groups = frame.groupby('family')[value]
    means, n = groups.mean(), groups.count()
    if (n < 2).any():
        raise ValueError('At least two independent channels per stratum required')
    v = groups.var(ddof=1) / n
    se = float(np.sqrt(v.sum()) / len(means))
    df = float(v.sum()**2 / (v*v/(n-1)).sum()) if v.sum() else len(frame)-len(means)
    mean = float(means.mean()); half = float(t.ppf(.975, df)*se)
    return dict(mean=mean, se=se, df=df, low=mean-half, high=mean+half, channels=len(frame))
