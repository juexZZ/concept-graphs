"""Separate offline scorer. Never imported by map construction or retrieval."""
import json
from pathlib import Path

from v2_unified_common import RUN, SHARED, now, read, write
from conceptgraph.v2_unified_cache import sha


def main():
    expected=read(SHARED/'inputs/queries.json')['queries']
    key=lambda q:(q['scene'],q['chapter'],q['query'])
    expected_keys={key(q) for q in expected}
    rows=[]; sources={}
    for path in sorted((RUN/'scenes').glob('*/predictions/*.json')):
        data=read(path)
        if data['status']!='COMPLETE':
            raise ValueError(f'Not a completed prediction artifact: {path}')
        rows.extend(data['predictions']); sources[str(path)]=sha(path)
    keys=[key(q) for q in rows]
    if len(set(keys))!=len(keys) or not set(keys)<=expected_keys:
        raise ValueError('Duplicate or unexpected queries')
    # Only this standalone offline scorer opens ground-truth frame membership.
    gt_path=SHARED/'evaluation/scoring_gt.json'
    gt={key(q):set(q['gt_frames']) for q in read(gt_path)['queries']}
    for row in rows:
        row['hit']=bool(row['in_current_traversal'] and row['frame_name'] in gt[key(row)])
    by_traversal={str(c):dict(hits=sum(q['hit'] for q in rows if q['chapter']==c),
                            total=sum(q['chapter']==c for q in rows)) for c in (1,2,3)}
    complete=set(keys)==expected_keys and len(rows)==198
    if complete:
        assert [by_traversal[str(c)]['total'] for c in (1,2,3)]==[66,67,65]
    summary=dict(status='COMPLETE' if complete else 'PARTIAL',method='conceptgraphs_unified',
                 config='sam2_siglip_fullseq_allframes',by_traversal=by_traversal,
                 hits=sum(q['hit'] for q in rows),total=len(rows),expected_total=198,
                 evaluated_queries=len(rows),expected_queries=198,updated=now(),
                 protocol='full-sequence native DA3 64/8; shared SAM2/SigLIP graph vectors; allframes single-pass cumulative CG; native top1-object/latest actual contributor; copy-only run-end postprocessing',
                 causal_prefix_geometry=False,query_prompt='a photo of a {query}',
                 confidence='cached SAM2 stability_score, NOT predicted_iou',
                 scoring='exact current-run camera-frame GT membership',
                 prediction_sha256=sources,gt_sha256=sha(gt_path))
    write(RUN/'results/per_query_scored.json',dict(status=summary['status'],predictions=rows))
    write(RUN/'results/summary.json',summary)
    print('OFFLINE_SCORE',summary['status'],summary['hits'],summary['total'],flush=True)


if __name__=='__main__':
    main()
