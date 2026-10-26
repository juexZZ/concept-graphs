"""GT-free immutable input and provenance helpers for the 2026-10-24 CG run."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/root/dejaview_experiments/baselines/conceptgraphs')
RUN = Path('/root/dejaview_experiments/runs/conceptgraphs-v2-20261024-r01')
SHARED = Path('/root/dejaview_experiments/runs/full-v2-20261023-r01')
DA3_REPO = BASE / 'deps/Depth-Anything-3'
DA3_REV = '01accac4c0b73c8d6d35226d86a1c43fc32ea9aa'

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def frames_for(scene, chapter, limit=None):
    path = SHARED / 'inputs' / scene / 'frames.json'
    records = json.loads(path.read_text())['frames']
    prefix_dir = SHARED / 'inputs' / scene / f'prefix_{chapter}' / 'images'
    # The cumulative prefix symlinks are the contract; never sort legacy names.
    allowed = {p.resolve() for p in prefix_dir.iterdir() if p.suffix.lower() == '.jpg'}
    frames = [dict(f, index=i) for i, f in enumerate(records) if Path(f['path']).resolve() in allowed]
    assert len(frames) == len(allowed) and frames
    assert all(f['run'] <= chapter for f in frames)
    assert [f['index'] for f in frames] == list(range(len(frames)))
    if limit is not None:
        frames = frames[:limit]
    for frame in frames:
        if sha(frame['path']) != frame['sha256']:
            raise ValueError(f"Source hash mismatch: {frame['file']}")
    return frames

def source_provenance():
    return {
        'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'branch': subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip(),
        'files': {str(p.relative_to(REPO)): sha(p) for p in sorted(REPO.rglob('*.py')) if '.git' not in p.parts},
    }
