"""Run the branch-pinned metric exporter, with manifest order and source audits.

Run in the dedicated ConceptGraphs DA3 environment. This never reads scoring GT
and never substitutes streaming geometry. Native inference/stitch/export functions
are called unchanged; the adapter only supplies ordered paths and validates output.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import torch

from v2_common import DA3_REPO, DA3_REV, SHARED, frames_for, sha, utc, write_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--chapter', required=True, type=int, choices=[1, 2, 3])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    assert not args.output.exists(), f'Output exists: {args.output}'
    frames = frames_for(args.scene, args.chapter, args.limit)
    args.output.mkdir(parents=True)
    write_new(args.output / 'frames.json', {'scene': args.scene, 'chapter': args.chapter, 'frames': frames})
    exporter = DA3_REPO / 'tools/run_metric_depth.py'
    pinned = subprocess.check_output(['git', '-C', str(DA3_REPO), 'show', f'{DA3_REV}:tools/run_metric_depth.py'])
    assert exporter.read_bytes() == pinned
    spec = importlib.util.spec_from_file_location('cg_native_metric_export', exporter)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    model_record = json.loads((SHARED / 'inputs/da3_model.json').read_text())
    # Model weights are shared read-only, and already pinned by exact file hashes.
    for item in model_record['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f"DA3 model hash mismatch: {item['path']}")
    write_new(args.output / 'provenance.json', {
        'started': utc(), 'exporter_commit': DA3_REV, 'exporter_sha256': sha(exporter),
        'model': model_record, 'chunk_size': 64, 'overlap': 8,
        'axis_scale': .1, 'min_depth': .05, 'max_depth': 100., 'conf_quantile': .5,
        'input_order': 'immutable frames.json filtered by exact cumulative prefix membership',
        'gt_used': False, 'streaming_geometry_used': False,
    })
    model = native.DepthAnything3.from_pretrained(model_record['local_dir']).to(device='cuda').eval()
    paths = [f['path'] for f in frames]
    chunks = native.chunked_inference(model, paths, 64, 8)
    del model
    torch.cuda.empty_cache()
    stitched = native.stitch_chunks(chunks, len(frames))
    out = args.output / 'native'
    native.export_scene(stitched, str(out), paths, chunks, depth_scale=1000.)
    checks = []
    intrinsics = np.loadtxt(out / 'intrinsic/intrinsic_color.txt')
    np.testing.assert_allclose(intrinsics[:3, :3], stitched['intrinsics'][0], atol=1e-7)
    mapping = json.loads((out / 'filename_mapping.json').read_text())
    assert [row['original'] for row in mapping] == [f['file'] for f in frames]
    sizes = set()
    for i, frame in enumerate(frames):
        with Image.open(out / 'color' / f'{i}.jpg') as image:
            sizes.add(image.size)
            rgb_shape = (image.height, image.width)
        with Image.open(out / 'depth' / f'{i}.png') as image:
            depth = np.asarray(image)
        assert rgb_shape == depth.shape
        expected = np.clip(np.nan_to_num(stitched['depths'][i].astype(np.float32)*1000., nan=0., posinf=0., neginf=0.), 0, 65535).astype(np.uint16)
        np.testing.assert_array_equal(depth, expected)
        pose = np.loadtxt(out / 'pose' / f'{i}.txt')
        assert np.isfinite(pose).all()
        np.testing.assert_allclose(pose, stitched['c2ws'][i], atol=1e-7)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
        assert abs(np.linalg.det(pose[:3, :3])-1.) < 1e-3
        checks.append({'index': i, 'file': frame['file'], 'source_sha256': frame['sha256'],
                       'rgb_sha256': sha(out / 'color' / f'{i}.jpg'), 'valid_depth_pixels': int((depth > 0).sum()),
                       'max_depth_mm': int(depth.max()), 'source_chunk': int(stitched['sources'][i])})
    assert len(sizes) == 1, sizes
    width, height = sizes.pop()
    # Dataset config must describe exported pixel dimensions, not assume square v2 RGB.
    config = {'dataset_name': 'scannet', 'camera_params': {'image_height': height, 'image_width': width,
              'fx': 0., 'fy': 0., 'cx': 0., 'cy': 0., 'png_depth_scale': 1000.}}
    import yaml
    with (args.output / 'dataset.yaml').open('x') as stream:
        yaml.safe_dump(config, stream)
    write_new(args.output / 'complete.json', {'status': 'COMPLETE', 'finished': utc(), 'frames': len(frames),
        'dimensions': [height, width], 'depth_units': 'millimeters / 1000 on load', 'pose_convention': 'camera_to_world',
        'intrinsics_source': 'native exporter first-frame scene K (unchanged)', 'checks': checks})
    print('METRIC_GEOMETRY_COMPLETE', args.scene, args.chapter, len(frames), flush=True)

if __name__ == '__main__':
    main()
