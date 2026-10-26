"""Real 2metro full-T3 geometry/cache smoke, not scored experiment results."""
import argparse
import copy
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.v2_unified_state import (checkpoint, evaluation_copy, fingerprint,
                                          load_checkpoint, rng_state, restore_rng, write_pickle)
from conceptgraph.v2_unified_cache import sha
from v2_unified_map import configuration, dataset_for, process_frame, validate_geometry
from v2_unified_common import BASE, GEOMETRY, REPO, RUN, backend_hashes, now, read, write
from v2_unified_retrieve import retrieve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-gpu', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous smoke; use a new attempt directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    if args.require_gpu:
        assert torch.cuda.is_available(), 'Allocated GPU2 must be accessible'
        assert torch.cuda.device_count()==1
        torch.cuda.init()
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(REPO/'tests'),
                    '-p','test_v2_unified_*.py','-v'],check=True,cwd=REPO)
    scene='2metro8floor'; geometry=GEOMETRY/scene
    frames, boundaries, ready = validate_geometry(geometry, scene)
    cfg = configuration(geometry); dataset=dataset_for(geometry)
    item=np.load(BASE/'item_text.npz')['item']
    assert read(BASE/'item_text.json')['status']=='PASS'
    random.seed(121); np.random.seed(121); torch.manual_seed(121)
    objects=MapObjectList()
    for index in range(4):
        objects,_=process_frame(cfg,objects,index,dataset,frames,scene,item)
    assert objects
    prefix_fingerprint=fingerprint(objects)
    checkpoint(args.output/'stream4.pkl.gz', objects, 4, {}, {})
    evaluated,before=evaluation_copy(objects,cfg)
    assert prefix_fingerprint==before==fingerprint(objects)
    expected_rng=rng_state()
    # Continue directly after a non-mutating evaluation snapshot.
    direct=copy.deepcopy(objects)
    for index in (4,5):
        direct,_=process_frame(cfg,direct,index,dataset,frames,scene,item)
    direct_fingerprint=fingerprint(direct)
    # Resume exact stored stream and process only new global frame IDs 4 and 5.
    resumed,saved=load_checkpoint(args.output/'stream4.pkl.gz')
    assert saved['next_global_index']==4 and fingerprint(resumed)==prefix_fingerprint
    for index in range(saved['next_global_index'],6):
        resumed,_=process_frame(cfg,resumed,index,dataset,frames,scene,item)
    assert fingerprint(resumed)==direct_fingerprint, 'Snapshot/resume changed continuation'
    contributors=[i for obj in resumed for i in obj['image_idx']]
    assert 4 in contributors and 5 in contributors and min(contributors)>=0 and max(contributors)==5
    for obj in resumed:
        assert np.asarray(obj['clip_ft']).size==1152 and np.asarray(obj['text_ft']).size==1152
        assert not obj['is_background']
    endpoint,_=evaluation_copy(resumed,cfg)
    map_path=args.output/'smoke_post.pkl.gz'
    write_pickle(map_path,dict(objects=endpoint.to_serializable(),bg_objects=None,cfg=cfg))
    result=retrieve(scene,1,map_path,geometry,args.output/'smoke_predictions.json',smoke=True)
    assert result['status']=='SMOKE' and result['gt_used'] is False
    for row in result['predictions']:
        assert row['frame_index'] is None or row['frame_index']<6
    # An empty run-ending frame must not prevent snapshot creation.
    empty=MapObjectList(); empty_after,_=evaluation_copy(empty,cfg)
    assert not empty and not empty_after
    receipt=dict(status='PASS',finished=now(),scene=scene,real_frames_processed=6,
                 full_geometry_frames=len(frames),published_run_boundaries=boundaries,
                 synthetic_snapshot_after=4,synthetic_boundary_not_for_scoring=True,
                 resumed_global_indices=[4,5],prefix_replay=False,
                 exact_snapshot_no_mutation=True,exact_resume_continuation=True,
                 original_masks_overlap_preserved=True,embedding_dimension=1152,
                 query_source='same shared graph NPZ',gt_used=False,
                 gpu_available=torch.cuda.is_available(),
                 map_objects=len(endpoint),backend_sha256=backend_hashes(),
                 item_text_check=read(BASE/'item_text.json'),
                 smoke_root=str(args.output))
    write(args.output/'PASS.json',receipt)
    write(RUN/'smoke/PASS.json',receipt)
    print('UNIFIED_SMOKE_PASS',receipt,flush=True)


if __name__=='__main__':
    main()
