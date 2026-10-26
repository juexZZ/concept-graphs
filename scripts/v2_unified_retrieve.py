"""Shared SigLIP graph vectors; unchanged native top1-object/latest readout.

NO labels/GT frame lists are used. Query names are normal inference inputs.
"""
import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np

from conceptgraph.v2_unified_cache import SHARED, frame_run, query_vectors, sha
from v2_unified_common import read, write, now


def retrieve(scene, chapter, map_path, geometry, output, smoke=False):
    frames = read(geometry/'frames.json')['frames']
    mapping_data = read(geometry/'native/filename_mapping.json')
    mapping = {int(row['index']): row.get('original', row.get('original_filename')) for row in mapping_data}
    with gzip.open(map_path, 'rb') as stream:
        data = pickle.load(stream)
    objects = data['objects']
    for obj in objects:
        indices = [int(Path(path).stem) for path in obj['color_path']]
        if indices != list(obj['image_idx']):
            raise ValueError('Contributor IDs differ from actual source image IDs')
        if any(i >= len(frames) or frame_run(frames[i]) > chapter or mapping[i] != frames[i]['file'] for i in indices):
            raise ValueError('Snapshot contains future/unknown contributors')
    features = np.stack([np.asarray(obj['clip_ft'], dtype=np.float32) for obj in objects]) if objects else np.empty((0,1152), np.float32)
    if features.shape != (len(objects), 1152):
        raise ValueError('Mixed frontend feature spaces')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms == 0, 1., norms)
    vectors, metadata = query_vectors()
    queries = [q for q in read(SHARED/'inputs/queries.json')['queries'] if q['scene'] == scene and q['chapter'] == chapter]
    rows = []
    for query in queries:
        sims = features @ vectors[query['query']].reshape(-1, 1)
        sims = sims.reshape(-1)
        order = np.argsort(-sims)[:1]  # identical native argsort/tie behavior
        selected = int(order[0]) if len(order) else None
        indices = sorted({int(Path(p).stem) for p in objects[selected]['color_path']}, reverse=True) if selected is not None else []
        index = indices[0] if indices else None
        name = mapping[index] if index is not None else None
        current = index is not None and frame_run(frames[index]) == chapter
        rows.append(dict(query, method='conceptgraphs_unified', config='sam2_siglip_fullseq_allframes',
                         frame_name=name, frame_index=index, contributor_frame_name=name,
                         contributor_frame_index=index, in_current_traversal=bool(current),
                         candidate_frames=[name] if current else [], instance_id=selected,
                         similarity=float(sims[selected]) if selected is not None else None))
    result = dict(status='SMOKE' if smoke else 'COMPLETE', finished=now(), scene=scene, chapter=chapter,
                  map_sha256=sha(map_path), n_objects=len(objects), gt_used=False,
                  template=metadata['graph_prompt'], query_feature_source='shared graph',
                  query_sha256=sha(SHARED/'inputs/query_embeddings.npz'), predictions=rows,
                  geometry_protocol='offline full-sequence native DA3 64/8, possible future-view geometry influence')
    write(output, result)
    print('RETRIEVAL_COMPLETE', scene, chapter, len(rows), len(objects), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True); p.add_argument('--chapter', type=int, required=True)
    p.add_argument('--map', type=Path, required=True); p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    retrieve(args.scene, args.chapter, args.map, args.geometry, args.output)


if __name__ == '__main__':
    main()
