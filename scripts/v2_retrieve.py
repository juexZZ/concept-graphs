"""GT-free native CG ranking and latest *actual* contributor readout for v2."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eval_object_retrieval import ConceptGraphsRetriever
from v2_common import SHARED, sha, utc, write_new

def validate_contributors(retriever, frames):
    by_index = {f['index']: f for f in frames}
    for obj in retriever.objects:
        for path in obj.get('color_path', []):
            index = int(Path(path).stem)
            if index not in by_index:
                raise ValueError(f'Future/unknown contributor {index}')
            if retriever.idx_to_orig.get(index) != by_index[index]['file']:
                raise ValueError(f'Incorrect contributor mapping {index}')
    return by_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--chapter', required=True, type=int)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    frames = json.loads((args.geometry / 'frames.json').read_text())['frames']
    assert all(f['run'] <= args.chapter for f in frames)
    mapping = args.geometry / 'native/filename_mapping.json'
    retriever = ConceptGraphsRetriever(str(args.map), str(mapping), device='cuda')
    by_index = validate_contributors(retriever, frames)
    queries_path = SHARED / 'inputs/queries.json'
    queries = [q for q in json.loads(queries_path.read_text())['queries']
               if q['scene'] == args.scene and q['chapter'] == args.chapter]
    predictions = []
    for query in queries:
        embedding = retriever.get_text_embedding(query['query'])
        order, sims = retriever.query_top_instance(embedding, top_k=1)
        top = retriever.get_top_t_latest_frames(int(order[0]), 1) if len(order) else []
        name, index = top[0] if top else (None, None)
        current = bool(index is not None and by_index[index]['run'] == args.chapter)
        predictions.append(dict(query, method='conceptgraphs', config='native_allframes_latest_actual',
            frame_name=name, frame_index=index, contributor_frame_name=name, contributor_frame_index=index,
            in_current_traversal=current, candidate_frames=[name] if name and current else [],
            instance_id=int(order[0]) if len(order) else None, similarity=float(sims[0]) if len(sims) else None))
    write_new(args.output, {'status': 'SMOKE' if args.smoke else 'COMPLETE', 'finished': utc(),
        'scene': args.scene, 'chapter': args.chapter, 'n_objects': len(retriever.objects),
        'map_sha256': sha(args.map), 'mapping_sha256': sha(mapping), 'queries_sha256': sha(queries_path),
        'gt_used': False, 'template': 'a photo of {query}', 'predictions': predictions})
    print('RETRIEVAL_COMPLETE', args.scene, args.chapter, len(predictions), len(retriever.objects), flush=True)

if __name__ == '__main__':
    main()
