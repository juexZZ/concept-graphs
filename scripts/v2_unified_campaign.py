"""Durable single-worker nine-scene scheduler with per-scene geometry gates."""
import fcntl
import os
import subprocess
import sys
import time

import torch

from v2_unified_common import BASE, GEOMETRY, REPO, RUN, SHARED, backend_hashes, now, read, write


def status(phase, state, progress, error=None):
    write(BASE/'STATUS.json',dict(method='conceptgraphs_unified',phase=phase,status=state,
                                run_root=str(RUN),updated=now(),progress=progress,last_error=error,
                                gpu=2,cpu_threads=4,supervisor_pid=os.getpid(),
                                environment='/root/dejaview_envs/conceptgraphs',
                                protocol='sam2_siglip_fullseq_allframes',
                                native_conceptgraphs_untouched=True))


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    smoke=read(RUN/'smoke/PASS.json')
    assert smoke['status']=='PASS' and smoke['gpu_available']
    assert smoke['backend_sha256']==backend_hashes(), 'Backend changed since smoke'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4)
    scenes=sorted({q['scene'] for q in read(SHARED/'inputs/queries.json')['queries']})
    assert len(scenes)==9
    write(RUN/'campaign.json',dict(status='RUNNING',started=now(),scenes=scenes,gpu=2,
                                 python=sys.executable,source_root=str(REPO),
                                 frames=9692,main_pid=os.getpid(),smoke=smoke))
    (RUN/'logs').mkdir(exist_ok=True)
    failed={}
    last_prediction_count=-1
    while True:
        complete=[s for s in scenes if (RUN/'scenes'/s/'COMPLETE.json').exists()]
        predictions=list((RUN/'scenes').glob('*/predictions/*.json'))
        if len(predictions)!=last_prediction_count:
            subprocess.run([sys.executable,str(REPO/'scripts/v2_unified_score.py')],check=True)
            last_prediction_count=len(predictions)
        if len(complete)==9:
            status('complete','COMPLETE',dict(completed_scenes=9,scenes=complete,snapshots=27))
            write(RUN/'COMPLETE.json',dict(status='COMPLETE',finished=now(),scenes=9,frames=9692))
            return
        ready=[s for s in scenes if s not in complete and s not in failed and (GEOMETRY/s/'READY.json').exists()]
        if not ready:
            pending=[s for s in scenes if s not in complete and s not in failed]
            if not pending:
                status('failed','FAILED',dict(completed_scenes=len(complete),failed_scenes=failed),failed)
                return
            status('waiting_shared_geometry','WAITING',dict(completed_scenes=len(complete),
                    waiting_scenes=pending,failed_scenes=failed))
            time.sleep(15)
            continue
        scene=ready[0]; output=RUN/'scenes'/scene
        argv=[sys.executable,str(REPO/'scripts/v2_unified_map.py'),'--scene',scene,
              '--geometry',str(GEOMETRY/scene),'--output',str(output)]
        stamp=now().replace(':','-')
        log_path=RUN/'logs'/f'{scene}-{stamp}.log'
        write(RUN/'logs'/f'{scene}-{stamp}.command.json',dict(argv=argv,started=now(),source_root=str(REPO)))
        with log_path.open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
            while child.poll() is None:
                progress=read(output/'PROGRESS.json') if (output/'PROGRESS.json').exists() else {}
                status('mapping','RUNNING',dict(completed_scenes=len(complete),active_scene=scene,
                    child_pid=child.pid,log=str(log_path),scene_progress=progress,
                    failed_scenes=failed,total_scenes=9))
                count=len(list((RUN/'scenes').glob('*/predictions/*.json')))
                if count!=last_prediction_count:
                    subprocess.run([sys.executable,str(REPO/'scripts/v2_unified_score.py')],check=True)
                    last_prediction_count=count
                time.sleep(15)
        write(RUN/'logs'/f'{scene}-{stamp}.exit.json',dict(returncode=child.returncode,finished=now()))
        if child.returncode!=0:
            failed[scene]=dict(returncode=child.returncode,log=str(log_path))
            status('scene_failed','RUNNING',dict(completed_scenes=len(complete),failed_scenes=failed),failed[scene])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('controller_failed','FAILED',{},repr(exc))
        raise
