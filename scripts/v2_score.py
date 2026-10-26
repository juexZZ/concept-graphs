"""Offline-only scoring. This module is not imported by mapping or ranking."""
import json
from pathlib import Path
from v2_common import RUN, SHARED, utc, write_new

def main():
    expected = json.loads((SHARED / 'inputs/queries.json').read_text())['queries']
    key = lambda q: (q['scene'], q['chapter'], q['query'])
    expected_keys = {key(q) for q in expected}
    predictions = []
    for file in sorted((RUN / 'predictions').glob('*.json')):
        data = json.loads(file.read_text())
        if data['status'] != 'COMPLETE':
            raise ValueError(f'Incomplete prediction file: {file}')
        predictions.extend(data['predictions'])
    if len(predictions) != 198 or {key(q) for q in predictions} != expected_keys:
        raise ValueError(f'Not final coverage: {len(predictions)}/198')
    # First and only access to scoring GT, after all predictions have been sealed.
    source = SHARED / 'evaluation/scoring_gt.json'
    gt = {key(q): set(q['gt_frames']) for q in json.loads(source.read_text())['queries']}
    for row in predictions:
        row['hit'] = bool(row['in_current_traversal'] and row['frame_name'] in gt[key(row)])
    by_chapter = {str(c): {'hits': sum(p['hit'] for p in predictions if p['chapter'] == c),
                          'total': sum(p['chapter'] == c for p in predictions)} for c in (1, 2, 3)}
    assert [by_chapter[str(c)]['total'] for c in (1, 2, 3)] == [66, 67, 65]
    write_new(RUN / 'results/per_query_scored.json', {'predictions': predictions})
    write_new(RUN / 'results/summary.json', {'status': 'COMPLETE', 'method': 'conceptgraphs',
        'config': 'native_allframes_latest_actual', 'finished': utc(), 'by_traversal': by_chapter,
        'hits': sum(p['hit'] for p in predictions), 'evaluated_queries': len(predictions), 'expected_queries': 198,
        'scoring': 'exact returned actual-contributor camera-frame membership in current traversal'})

if __name__ == '__main__':
    main()
