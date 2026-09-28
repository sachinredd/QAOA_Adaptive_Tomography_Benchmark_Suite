from pathlib import Path
import json,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_digest():
    h=hashlib.sha256()
    for p in sorted((ROOT/'src').rglob('*.py')):
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
def save_model(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
