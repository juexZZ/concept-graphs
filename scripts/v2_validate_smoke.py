"""Real native artifact validation, no annotations or scoring labels."""
import argparse
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from conceptgraph.dataset.datasets_common import get_dataset
from v2_common import utc, write_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    metadata = json.loads((args.geometry / 'complete.json').read_text())
    height, width = metadata['dimensions']
    dataset = get_dataset(dataconfig=str(args.geometry / 'dataset.yaml'), basedir=str(args.geometry),
              sequence='native', start=0, end=-1, stride=1, desired_height=height, desired_width=width,
              device='cpu', dtype=torch.float)
    assert len(dataset) == metadata['frames']
    color, depth, intrinsics, pose = dataset[0]
    assert tuple(color.shape) == (height, width, 3)
    assert tuple(depth.shape) == (height, width, 1)
    from PIL import Image
    with Image.open(args.geometry / 'native/depth/0.png') as image:
        source = np.asarray(image).astype(np.float32) / 1000.
    np.testing.assert_allclose(depth[..., 0].numpy(), source, atol=1e-6)
    np.testing.assert_allclose(intrinsics.numpy(), np.loadtxt(args.geometry / 'native/intrinsic/intrinsic_color.txt'), atol=1e-4)
    np.testing.assert_allclose(dataset.poses[0].numpy(), np.loadtxt(args.geometry / 'native/pose/0.txt'), atol=1e-5)
    masks = 0
    for path in sorted((args.geometry / 'native/gsa_detections_none').glob('*.pkl.gz')):
        with gzip.open(path, 'rb') as stream:
            data = pickle.load(stream)
        if len(data['image_feats']):
            assert data['image_feats'].shape[1] == 1024
            assert data['text_feats'].shape[1] == 1024
            assert np.isfinite(data['image_feats']).all()
            assert data['mask'].shape[1:] == (height, width)
        masks += len(data['mask'])
    with gzip.open(args.map, 'rb') as stream:
        objects = pickle.load(stream)['objects']
    assert objects and masks
    contributor_counts = []
    for obj in objects:
        assert np.asarray(obj['clip_ft']).size == 1024
        actual = {int(Path(p).stem) for p in obj['color_path']}
        assert actual and actual <= set(range(len(dataset)))
        contributor_counts.append(len(actual))
    assert max(contributor_counts) >= 2, 'Need at least one genuinely multi-frame fused object'
    write_new(args.output, {'status': 'COMPLETE', 'finished': utc(), 'frames': len(dataset),
        'mask_count': masks, 'objects': len(objects), 'max_actual_contributors': max(contributor_counts),
        'clip_dimension': 1024, 'dimensions': [height, width], 'native_depth_mm_to_m_verified': True,
        'native_c2w_verified': True, 'native_scene_intrinsics_verified': True,
        'contributor_ids_in_prefix_verified': True, 'scoring_gt_read': False})

if __name__ == '__main__':
    main()
