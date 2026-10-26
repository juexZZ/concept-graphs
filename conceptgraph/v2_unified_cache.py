"""Read-only shared SAM2/SigLIP adapter. Never imports evaluation ground truth."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SHARED = Path('/root/dejaview_experiments/runs/full-v2-20261023-r01')
FEATURE_DIM = 1152


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class NumpyCacheUnpickler(pickle.Unpickler):
    """NumPy2 cache compatibility without modifying the pinned NumPy1 runtime."""
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_objects(path):
    with Path(path).open('rb') as stream:
        objects = NumpyCacheUnpickler(stream).load()
    if not isinstance(objects, list):
        raise ValueError('Expected original list of per-object masks and features')
    return objects


def frame_run(frame):
    value = int(frame.get('run_id', frame.get('run', -1)))
    if value not in (1, 2, 3):
        raise ValueError(f'Invalid published run_id: {value}')
    return value


def validate_frames(frames, require_three=True):
    previous = None
    for i, frame in enumerate(frames):
        key = (frame_run(frame), int(frame['timestamp_ns']))
        if previous is not None and key < previous:
            raise ValueError('Manifest must be ordered by run_id, integer timestamp_ns')
        if int(frame.get('index', i)) != i:
            raise ValueError('Frame IDs must be full-sequence global indices')
        previous = key
    if require_three and {frame_run(f) for f in frames} != {1, 2, 3}:
        raise ValueError('Full experiment must cover all three runs')
    return {frame_run(f): i + 1 for i, f in enumerate(frames)}


def cache_paths(scene, frame, shared=SHARED):
    root = shared / 'features' / scene / Path(frame['file']).stem
    return root / 'siglip_objects.pkl', root / 'complete.json'


def to_gobs(objects, item_text):
    """Retain overlap and original masks; native CG resizes and filters them."""
    item_text = np.asarray(item_text, dtype=np.float32).reshape(-1)
    if item_text.shape != (FEATURE_DIM,) or not np.isfinite(item_text).all():
        raise ValueError('Wrong SigLIP item text feature')
    masks, boxes, features, confidence = [], [], [], []
    for obj in objects:
        mask = np.asarray(obj['mask'])
        feature = np.asarray(obj['embedding'])
        if mask.shape != (800, 800) or mask.dtype != np.bool_:
            raise ValueError('Expected original published 800x800 boolean mask')
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            raise ValueError('Expected original 1152-D SigLIP feature')
        if not np.isclose(np.linalg.norm(feature), 1., atol=2e-4):
            raise ValueError('Cached object features must be normalized')
        if int(mask.sum()) != int(obj['area']):
            raise ValueError('Original mask and area disagree')
        x, y, w, h = obj['original_bbox']
        # original_bbox is mask xywh, NOT the padded semantic crop.
        boxes.append((x, y, x + w, y + h))
        masks.append(mask)
        features.append(feature)
        # Cache has no predicted_iou. Explicitly use saved SAM2 stability_score.
        score = float(obj['stability_score'])
        if not 0 <= score <= 1:
            raise ValueError('Invalid SAM2 stability score')
        confidence.append(score)
    n = len(objects)
    return dict(xyxy=np.asarray(boxes, dtype=np.float32).reshape(n, 4),
                mask=np.stack(masks) if n else np.empty((0, 800, 800), dtype=bool),
                confidence=np.asarray(confidence, dtype=np.float32),
                class_id=np.zeros(n, dtype=np.int64), classes=['item'],
                image_feats=np.stack(features) if n else np.empty((0, FEATURE_DIM), np.float32),
                text_feats=np.repeat(item_text[None], n, axis=0))


def read_gobs(scene, frame, item_text, shared=SHARED):
    objects_path, marker_path = cache_paths(scene, frame, shared)
    marker = json.loads(marker_path.read_text())
    if marker['image_sha256'] != frame['sha256'] or marker['image'] != frame['file']:
        raise ValueError(f'Feature source identity mismatch: {objects_path}')
    objects = load_objects(objects_path)
    if marker['objects'] != len(objects):
        raise ValueError('Incomplete per-object cache')
    return to_gobs(objects, item_text)


def query_vectors(shared=SHARED):
    metadata = json.loads((shared / 'inputs/query_embeddings.json').read_text())
    model = next(m for m in json.loads((shared / 'inputs/perception_models.json').read_text())
                 if m['repo'] == 'google/siglip-so400m-patch14-384')
    if metadata['status'] != 'COMPLETE' or metadata['graph_prompt'] != 'a photo of a {query}':
        raise ValueError('Unexpected shared graph text protocol')
    if metadata['model'] != model['local_dir']:
        raise ValueError('Image and text SigLIP model identity mismatch')
    with np.load(shared / 'inputs/query_embeddings.npz', allow_pickle=False) as cache:
        vectors = cache['graph'].copy()
    if vectors.shape != (len(metadata['labels']), FEATURE_DIM):
        raise ValueError('Wrong query embedding shape')
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=2e-5)
    return dict(zip(metadata['labels'], vectors)), metadata
