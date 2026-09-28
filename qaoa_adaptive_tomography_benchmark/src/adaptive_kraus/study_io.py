"""Atomic study outputs, per-task locks and verified round commits."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from .study_protocol import digest, source_digest, environment, expand_spec, plan_estimate


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_plan(spec, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Plan output must be empty; run/resume an existing plan instead')
    tasks = expand_spec(spec)
    manifest = {'schema_version': 1, 'spec': spec, 'tasks': tasks,
                'source_digest': source_digest(), 'environment': environment(),
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'estimate': plan_estimate(spec, tasks)}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(output/'manifest.json', manifest)
    return manifest


def load_plan(output, executing=False):
    manifest = read_json(Path(output)/'manifest.json')
    payload = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    if digest(payload) != manifest['fingerprint']:
        raise ValueError('Manifest was modified or damaged; create a new plan')
    if executing:
        if source_digest() != manifest['source_digest']:
            raise ValueError('Source code differs from the frozen plan; create a new plan')
        if environment() != manifest['environment']:
            raise ValueError('Runtime package versions differ from the frozen plan; create a new plan in this environment')
    return manifest


@contextmanager
def task_lock(directory, recover=False):
    """Never steal a live lock. Only recover a dead process on this same host."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'.lock'
    if recover and lock.exists():
        owner = read_json(lock)
        if owner['host'] != socket.gethostname():
            raise RuntimeError('Lock belongs to another host; verify its job has ended before manually removing .lock')
        try:
            os.kill(owner['pid'], 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError('Lock owner is still running; cannot recover')
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Task is locked: {lock}') from exc
    owner = {'pid': os.getpid(), 'host': socket.gethostname(), 'created_utc': datetime.now(timezone.utc).isoformat()}
    with os.fdopen(fd, 'w') as stream: json.dump(owner, stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def commit_round(staging, destination):
    staging, destination = Path(staging), Path(destination)
    checksums = {p.name: file_digest(p) for p in staging.iterdir() if p.is_file()}
    atomic_json(staging/'complete.json', checksums)
    if destination.exists(): raise FileExistsError('Round is already committed')
    os.replace(staging, destination)


def verify_round(directory):
    directory = Path(directory)
    checksums = read_json(directory/'complete.json')
    required = {'model.npz', 'observations.json', 'row.json', 'fit_history.json'}
    if not required <= set(checksums): raise ValueError('Incomplete round commit')
    for name, expected in checksums.items():
        if Path(name).name != name or file_digest(directory/name) != expected:
            raise ValueError(f'Round checksum mismatch: {directory/name}')
    return read_json(directory/'row.json')
