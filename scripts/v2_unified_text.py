"""One-shot SigLIP class text export in the existing shared encoder runtime.

CG's pinned transformers4.31 predates SigLIP; do not upgrade the running CG env.
Query retrieval itself reads the exact existing shared graph vectors, not these
new encodings. One shared query is independently checked against that cache.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipModel, SiglipProcessor

from conceptgraph.v2_unified_cache import SHARED, query_vectors, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    vectors, metadata = query_vectors()
    model_path = metadata['model']
    model = SiglipModel.from_pretrained(model_path, local_files_only=True).eval().cuda()
    processor = SiglipProcessor.from_pretrained(model_path, local_files_only=True)
    def encode(text):
        inputs = processor(text=[text], return_tensors='pt', padding='max_length').to('cuda')
        with torch.inference_mode():
            ft = model.get_text_features(**inputs)
            return (ft / ft.norm(dim=-1, keepdim=True)).float().cpu().numpy()[0]
    label = metadata['labels'][0]
    verified = encode(metadata['graph_prompt'].format(query=label))
    maximum_error = float(np.max(np.abs(verified - vectors[label])))
    np.testing.assert_allclose(verified, vectors[label], atol=2e-5, rtol=2e-4)
    item = encode('item')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, item=item)
    receipt = dict(status='PASS', model=model_path, model_sha256=sha(Path(model_path)/'model.safetensors'),
                   item_text='item', query_source=str(SHARED/'inputs/query_embeddings.npz'),
                   query_sha256=sha(SHARED/'inputs/query_embeddings.npz'),
                   shared_graph_cache_check_label=label, maximum_absolute_error=maximum_error,
                   image_and_text_dimension=1152, gpu=torch.cuda.get_device_name(),
                   mapping_environment_unchanged=True)
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
