"""Single-pass cumulative ConceptGraphs over sealed full-sequence DA3 geometry."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.v2_unified_cache import (SHARED, cache_paths, frame_run, read_gobs, sha, validate_frames)
from conceptgraph.v2_unified_mapping import step
from conceptgraph.v2_unified_state import (checkpoint, evaluation_copy, load_checkpoint, write_pickle, fingerprint)
from v2_unified_common import REPO, BASE, now, read, write, backend_hashes
from v2_unified_retrieve import retrieve


def configuration(geometry):
    cfg = OmegaConf.load(REPO/'conceptgraph/configs/slam_pipeline/base.yaml')
    overrides = dict(dataset_root=str(geometry), dataset_config=str(geometry/'dataset.yaml'),
                     scene_id='native', start=0, end=-1, stride=1, spatial_sim_type='overlap',
                     mask_conf_threshold=.95, match_method='sim_sum', sim_threshold=1.2,
                     dbscan_eps=.1, gsa_variant='none', class_agnostic=True, skip_bg=True,
                     max_bbox_area_ratio=.5, obj_min_detections=2, merge_interval=20,
                     filter_interval=10, merge_overlap_thresh=.7, merge_visual_sim_thresh=.8,
                     merge_text_sim_thresh=.8, image_height=504, image_width=504)
    cfg.update(overrides)
    return cfg


def dataset_for(geometry):
    return get_dataset(dataconfig=str(geometry/'dataset.yaml'), start=0, end=-1, stride=1,
                       basedir=str(geometry), sequence='native', desired_height=504,
                       desired_width=504, device='cpu', dtype=torch.float)


def validate_geometry(geometry, scene):
    ready = read(geometry/'READY.json')
    if ready['status'] != 'COMPLETE' or ready['scene'] != scene or ready['subsampled']:
        raise ValueError('Unsealed/incompatible geometry')
    if ready['causal_prefix_geometry'] or ready['dimensions'] != [504, 504]:
        raise ValueError('This variant requires full-sequence native geometry')
    if sha(geometry/'frames.json') != ready['source_frames_sha256']:
        raise ValueError('Geometry frame manifest changed')
    if sha(geometry/'file_hashes.json') != ready['file_hashes_sha256']:
        raise ValueError('Geometry checksum manifest changed')
    frames = read(geometry/'frames.json')['frames']
    source = read(SHARED/'inputs'/scene/'frames.json')['frames']
    if len(frames) != ready['frames'] or len(frames) != len(source):
        raise ValueError('Not the complete T3 sequence')
    boundaries = validate_frames(frames)
    for frame, original in zip(frames, source):
        for key in ('file','sha256','timestamp_ns','run','pair_index'):
            if frame[key] != original[key]:
                raise ValueError(f'Shared input identity mismatch {key}')
    return frames, boundaries, ready


def verify_perception(scene, frames):
    records = []
    for i, frame in enumerate(frames):
        path, marker_path = cache_paths(scene, frame)
        marker = read(marker_path)
        if marker['image_sha256'] != frame['sha256'] or marker['image'] != frame['file']:
            raise ValueError(f'Perception hash mismatch at {scene}/{i}')
        records.append(dict(index=i, original=frame['file'], objects=marker['objects'],
                            image_sha256=marker['image_sha256'], object_cache_sha256=sha(path)))
        if (i+1)%100==0:
            print('VERIFY_EXISTING_CACHE',scene,i+1,len(frames),flush=True)
    return records


def process_frame(cfg, objects, index, dataset, frames, scene, item):
    frame = frames[index]
    color_tensor, depth_tensor, intrinsics, *_ = dataset[index]
    image = color_tensor.cpu().numpy().astype(np.uint8)
    depth = depth_tensor[..., 0].cpu().numpy()
    if image.shape != (504,504,3) or depth.shape != (504,504):
        raise ValueError('Geometry resolution mismatch')
    if int(Path(dataset.color_paths[index]).stem) != index:
        raise ValueError('Dataset reordered numeric global IDs')
    gobs = read_gobs(scene, frame, item)
    # Use dataset.poses (untransformed native c2w), as in native pipeline.
    return step(cfg, objects, index, dataset.color_paths[index], image, depth,
                intrinsics.cpu().numpy()[:3,:3], dataset.poses[index].cpu().numpy(), gobs)


def snapshot(objects, cfg, scene, chapter, end, geometry, output, provenance):
    target = output/'snapshots'/f't{chapter}_post.pkl.gz'
    prediction = output/'predictions'/f'{scene}-t{chapter}.json'
    receipt = output/'snapshots'/f't{chapter}.json'
    if target.exists() and receipt.exists():
        data = read(receipt)
        if data['next_global_index'] != end or data['stream_fingerprint'] != fingerprint(objects):
            raise ValueError('Cannot silently replace a completed snapshot')
    else:
        evaluated, before = evaluation_copy(objects, cfg)
        write_pickle(target, dict(objects=evaluated.to_serializable(), bg_objects=None, cfg=cfg,
                                 class_names=['item'], class_colors={'0':[0.5,0.5,0.5]},
                                 run_id=chapter, next_global_index=end, provenance=provenance))
        write(receipt, dict(status='COMPLETE', chapter=chapter, next_global_index=end,
                            stream_fingerprint=before, snapshot_does_not_mutate_stream=True,
                            stream_objects=len(objects), snapshot_objects=len(evaluated),
                            map_sha256=sha(target), finished=now()))
    if not prediction.exists():
        retrieve(scene, chapter, target, geometry, prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    frames, boundaries, ready = validate_geometry(args.geometry, args.scene)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'COMPLETE.json').exists():
        raise ValueError('Completed scene is immutable; scheduler should skip it')
    cfg = configuration(args.geometry)
    item = np.load(BASE/'item_text.npz', allow_pickle=False)['item']
    item_receipt = read(BASE/'item_text.json')
    if item_receipt['status'] != 'PASS':
        raise ValueError('Text model cross-check not passed')
    provenance = dict(scene=args.scene, geometry_ready_sha256=sha(args.geometry/'READY.json'),
                      geometry=ready, backend_sha256=backend_hashes(),
                      query_sha256=sha(SHARED/'inputs/query_embeddings.npz'),
                      item_sha256=sha(BASE/'item_text.npz'), feature_dimension=1152,
                      confidence='saved SAM2 stability_score (not native SAM predicted_iou)',
                      background_id_excluded=0, fullframe=True, gt_used=False)
    source_receipt = args.output/'perception.json'
    if not source_receipt.exists():
        write(args.output/'PROGRESS.json',dict(status='RUNNING',phase='verify_existing_cache',
                                             scene=args.scene,completed_frames=0,
                                             total_frames=len(frames),updated=now()))
        write(source_receipt, dict(status='COMPLETE', frames=verify_perception(args.scene, frames)))
    provenance['perception_receipt_sha256'] = sha(source_receipt)
    for path, value in [(args.output/'config.json',OmegaConf.to_container(cfg,resolve=True)),
                        (args.output/'provenance.json',provenance)]:
        if path.exists():
            if read(path)!=value:
                raise ValueError(f'Existing provenance/config changed: {path}')
        else:
            write(path,value)
    dataset = dataset_for(args.geometry)
    if len(dataset) != len(frames):
        raise ValueError('Native dataset coverage mismatch')
    checkpoint_path = args.output/'stream.pkl.gz'
    objects = MapObjectList(); start = 0
    if checkpoint_path.exists():
        objects, saved = load_checkpoint(checkpoint_path)
        if saved['provenance'] != provenance:
            raise ValueError('Resume provenance mismatch')
        if saved['config'] != OmegaConf.to_container(cfg, resolve=True):
            raise ValueError('Resume configuration mismatch')
        start = int(saved['next_global_index'])
        print('RESUME_EXACT_STREAM', start, len(objects), flush=True)
        for chapter, end in boundaries.items():
            if end == start:
                snapshot(objects, cfg, args.scene, chapter, end, args.geometry, args.output, provenance)
    processed = []
    for index in range(start, len(frames)):
        before = time.monotonic()
        write(args.output/'PROGRESS.json', dict(status='RUNNING', scene=args.scene, updated=now(),
                                              phase='native_frame_update', next_global_index=index,
                                              completed_frames=index, total_frames=len(frames),
                                              run_id=frame_run(frames[index]), objects=len(objects)))
        objects, n = process_frame(cfg, objects, index, dataset, frames, args.scene, item)
        processed.append(index)
        print('FRAME', index, 'RUN', frame_run(frames[index]), 'DETECTIONS', n, 'OBJECTS', len(objects),
              'SECONDS', round(time.monotonic()-before,3), flush=True)
        # Journal contains actual updates, preserving global source identities.
        with (args.output/'processed.jsonl').open('a') as journal:
            journal.write(json.dumps(dict(global_index=index, run=frame_run(frames[index]),
                                          frame=frames[index]['file'], detections=n, objects=len(objects)))+'\n')
        end = index+1
        at_boundary = end in boundaries.values()
        if at_boundary or end % args.checkpoint_interval == 0:
            checkpoint(checkpoint_path, objects, end, OmegaConf.to_container(cfg, resolve=True), provenance)
        if at_boundary:
            chapter = frame_run(frames[index])
            write(args.output/'PROGRESS.json', dict(status='RUNNING', phase='evaluation_copy',
                                                  scene=args.scene, completed_frames=end,
                                                  total_frames=len(frames), run_id=chapter, updated=now()))
            snapshot(objects, cfg, args.scene, chapter, end, args.geometry, args.output, provenance)
    write(args.output/'COMPLETE.json', dict(status='COMPLETE', scene=args.scene, finished=now(),
                                          frames=len(frames), snapshots=3, resumed_from=start,
                                          processed_this_invocation=len(processed),
                                          no_prefix_replay=True, gt_used=False, geometry_protocol=ready['geometry_protocol']))


if __name__ == '__main__':
    main()
