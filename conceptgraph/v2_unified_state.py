"""Exact stream checkpointing; terminal evaluation never mutates the live map."""
import copy
import gzip
import hashlib
import pickle
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from conceptgraph.slam.slam_classes import MapObjectList


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def exact_serializable(objects):
    data = objects.to_serializable()
    for obj, saved in zip(objects, data):
        box = obj['bbox']
        if isinstance(box, o3d.geometry.OrientedBoundingBox):
            saved['_exact_bbox'] = dict(kind='oriented', center=np.asarray(box.center).copy(),
                                       R=np.asarray(box.R).copy(), extent=np.asarray(box.extent).copy(),
                                       color=np.asarray(box.color).copy())
        else:
            saved['_exact_bbox'] = dict(kind='axis', low=box.get_min_bound().copy(),
                                       high=box.get_max_bound().copy(), color=np.asarray(box.color).copy())
    return data


def exact_load(data):
    objects = MapObjectList()
    for saved in data:
        obj = copy.deepcopy(saved)
        box = obj.pop('_exact_bbox')
        obj['pcd'] = o3d.geometry.PointCloud()
        obj['pcd'].points = o3d.utility.Vector3dVector(obj.pop('pcd_np'))
        obj['pcd'].colors = o3d.utility.Vector3dVector(obj.pop('pcd_color_np'))
        obj.pop('bbox_np')
        if box['kind'] == 'oriented':
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(box['center'], box['R'], box['extent'])
        else:
            obj['bbox'] = o3d.geometry.AxisAlignedBoundingBox(box['low'], box['high'])
        obj['bbox'].color = box['color']
        obj['clip_ft'] = torch.from_numpy(np.asarray(obj['clip_ft']).copy())
        obj['text_ft'] = torch.from_numpy(np.asarray(obj['text_ft']).copy())
        objects.append(obj)
    return objects


def _digest(value, h):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        h.update(str((value.dtype.str, value.shape)).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            h.update(str(key).encode()); _digest(value[key], h)
    elif isinstance(value, (tuple, list)):
        h.update(str(len(value)).encode())
        for item in value:
            _digest(item, h)
    else:
        h.update(repr(value).encode())


def fingerprint(objects):
    h = hashlib.sha256()
    _digest(exact_serializable(objects), h)
    return h.hexdigest()


def write_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wb', compresslevel=1) as stream:
        pickle.dump(data, stream, protocol=4)
    temporary.replace(path)


def checkpoint(path, objects, next_index, config, provenance):
    write_pickle(path, dict(format_version=1, objects=exact_serializable(objects),
                           next_global_index=next_index, rng=rng_state(), config=config,
                           provenance=provenance, phase='pre-terminal-stream'))


def load_checkpoint(path):
    with gzip.open(path, 'rb') as stream:
        data = pickle.load(stream)
    objects = exact_load(data['objects'])
    restore_rng(data['rng'])
    return objects, data


def evaluation_copy(objects, cfg):
    from conceptgraph.slam.utils import denoise_objects, filter_objects, merge_objects
    before = fingerprint(objects)
    rng = rng_state()
    try:
        result = denoise_objects(cfg, copy.deepcopy(objects))
        result = filter_objects(cfg, result)
        result = merge_objects(cfg, result)
    finally:
        restore_rng(rng)
    after = fingerprint(objects)
    if before != after:
        raise AssertionError('Evaluation snapshot mutated the ongoing map')
    return result, before
