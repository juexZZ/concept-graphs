import datetime
import json
import os
from pathlib import Path

from conceptgraph.v2_unified_cache import SHARED, sha

REPO = Path(os.environ.get('CG_UNIFIED_SOURCE_ROOT','/root/dejaview-baselines/concept-graphs'))
BASE = Path('/root/dejaview_experiments/baselines/conceptgraphs_unified')
RUN = Path('/root/dejaview_experiments/runs/conceptgraphs-unified-v2-20261026-r01')
GEOMETRY = Path('/root/dejaview_data/shared_geometry/da3-full-sequence-20261026-r01')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def backend_hashes():
    paths = [REPO/'conceptgraph/slam/utils.py', REPO/'conceptgraph/slam/mapping.py',
             REPO/'conceptgraph/slam/slam_classes.py', REPO/'conceptgraph/utils/ious.py',
             REPO/'conceptgraph/dataset/datasets_common.py', REPO/'conceptgraph/configs/slam_pipeline/base.yaml']
    return {str(path.relative_to(REPO)): sha(path) for path in paths}
