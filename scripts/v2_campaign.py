"""Serial native CG campaign with real smokes, durable logs and a full-run gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from v2_common import BASE, REPO, RUN, SHARED, source_provenance, utc, write_new

sys.path.insert(0, str(BASE))
from update_status import update

CG_PY = '/root/dejaview_envs/conceptgraphs/bin/python'
DA3_PY = '/root/dejaview_envs/conceptgraphs-support/da3/bin/python'

def run_step(command, log, phase, scene, chapter, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {'argv': command, 'started': utc(), 'log': str(log), 'scene': scene, 'chapter': chapter}
    write_new(log.with_suffix('.command.json'), record)
    update(phase=phase, status='in_progress', current_scene=scene, current_chapter=chapter,
           current_command=command, current_log=str(log), last_error=None)
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                update(phase=phase, status='in_progress', current_pid=process.pid)
    write_new(log.with_suffix('.exit.json'), dict(record, finished=utc(), returncode=code))
    if code:
        raise RuntimeError(f'{phase} exited {code}; see {log}')

def run_prefix(scene, chapter, smoke_id=None):
    smoke = smoke_id is not None
    scope = RUN / 'smoke' / smoke_id if smoke else RUN
    name = f'{scene}-t{chapter}'
    geometry = scope / 'geometry' / name
    logdir = scope / 'logs'
    command = [DA3_PY, str(REPO / 'scripts/v2_metric_geometry.py'), '--scene', scene,
               '--chapter', str(chapter), '--output', str(geometry)]
    if smoke:
        command += ['--limit', '8']
    run_step(command, logdir / f'{name}-metric.log', 'smoke_geometry' if smoke else 'metric_geometry', scene, chapter)
    env = dict(os.environ, PYTHON=CG_PY, DATASET_ROOT=str(geometry),
               DATASET_CONFIG=str(geometry / 'dataset.yaml'), GPU_ID='4',
               RUN_ID=f'cg-v2-{smoke_id or "full"}', STRIDE='1')
    # Native SAM segment-all and OpenCLIP features; no DejaView feature cache.
    run_step(['bash', str(REPO / 'run_cg_detect.sh'), 'native'], logdir / f'{name}-detect.log',
             'smoke_detection' if smoke else 'native_detection', scene, chapter, env)
    count = json.loads((geometry / 'complete.json').read_text())['frames']
    files = list((geometry / 'native/gsa_detections_none').glob('*.pkl.gz'))
    assert len(files) == count
    run_step(['bash', str(REPO / 'run_cg_mapping.sh'), 'native'], logdir / f'{name}-map.log',
             'smoke_mapping' if smoke else 'native_mapping', scene, chapter, env)
    maps = list((geometry / 'native/pcd_saves').glob('*_post.pkl.gz'))
    assert len(maps) == 1, maps
    result = scope / 'predictions' / f'{name}.json'
    command = [CG_PY, str(REPO / 'scripts/v2_retrieve.py'), '--scene', scene,
               '--chapter', str(chapter), '--map', str(maps[0]), '--geometry', str(geometry), '--output', str(result)]
    if smoke:
        command += ['--smoke']
    run_step(command, logdir / f'{name}-retrieve.log', 'smoke_retrieval' if smoke else 'native_retrieval', scene, chapter)
    data = json.loads(result.read_text())
    if smoke:
        assert data['n_objects'] > 0, 'A real nonempty native map is required to pass the smoke gate'
        run_step([CG_PY, str(REPO / 'scripts/v2_validate_smoke.py'), '--geometry', str(geometry),
                  '--map', str(maps[0]), '--output', str(scope / 'interface_checks.json')],
                 logdir / 'interface-validation.log', 'smoke_validation', scene, chapter)
        write_new(scope / 'complete.json', {'status': 'COMPLETE', 'finished': utc(), 'frames': count,
                  'n_objects': data['n_objects'], 'ranking_queries': len(data['predictions']), 'scoring_gt_read': False,
                  'source_provenance': source_provenance()})
    return data

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke', metavar='ID')
    group.add_argument('--full', action='store_true')
    parser.add_argument('--approved-smoke', default='s01')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if args.smoke:
            run_prefix('2metro8floor', 1, args.smoke)
            update(phase='smoke_complete', status='ready_for_full_run',
                   smoke_receipt=str(RUN / 'smoke' / args.smoke / 'complete.json'), last_error=None)
            return
        gate = RUN / 'smoke' / args.approved_smoke / 'complete.json'
        smoke = json.loads(gate.read_text())
        assert smoke['status'] == 'COMPLETE'
        snapshot = source_provenance()
        assert snapshot == smoke['source_provenance'], 'Checkout changed after smoke; repeat smoke'
        queries = json.loads((SHARED / 'inputs/queries.json').read_text())['queries']
        assert len(queries) == 198
        scenes = sorted({q['scene'] for q in queries})
        assert len(scenes) == 9
        write_new(RUN / 'protocol.json', {'started': utc(), 'method': 'conceptgraphs',
                  'expected_scenes': 9, 'expected_prefixes': 27, 'expected_queries': 198,
                  'expected_positive_queries_by_traversal': [66, 67, 65], 'scenes': scenes,
                  'depth_pose': 'native metric exporter DA3 01accac4, 64/8, no streaming',
                  'features': 'native SAM ViT-H + OpenCLIP ViT-H-14 laion2b_s32b_b79k',
                  'template': 'a photo of {query}', 'readout': 'top instance / latest actual color_path contributor',
                  'causality': 'independent cumulative prefix depth, poses, detections, and mapping',
                  'scoring': 'exact current-traversal camera-frame; GT is accessed offline only after all predictions',
                  'source_provenance': snapshot, 'smoke_gate': str(gate)})
        completed = 0
        nqueries = 0
        for scene in scenes:
            for chapter in (1, 2, 3):
                assert source_provenance() == snapshot, 'Active checkout changed'
                result = run_prefix(scene, chapter)
                completed += 1
                nqueries += len(result['predictions'])
                update(progress={'scenes_completed': completed // 3, 'prefixes_completed': completed,
                                 'queries_predicted': nqueries, 'queries_scored': 0, 'expected_queries': 198})
        run_step([CG_PY, str(REPO / 'scripts/v2_score.py')], RUN / 'logs/offline-score.log', 'offline_scoring', 'all', 3)
        summary = json.loads((RUN / 'results/summary.json').read_text())
        update(phase='complete', status='complete', results=summary,
               progress={'scenes_completed': 9, 'prefixes_completed': 27, 'queries_predicted': 198,
                         'queries_scored': 198, 'expected_queries': 198}, last_error=None)
    except Exception as error:
        update(status='failed', last_error=repr(error))
        raise

if __name__ == '__main__':
    main()
