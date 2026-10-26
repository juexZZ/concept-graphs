import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from conceptgraph.v2_unified_cache import (SHARED, frame_run, load_objects, query_vectors,
                                          read_gobs, to_gobs, validate_frames)
from conceptgraph.v2_unified_state import (fingerprint, checkpoint, load_checkpoint,
                                          evaluation_copy, rng_state)
from conceptgraph.slam.utils import resize_gobs
from conceptgraph.slam.slam_classes import MapObjectList


class CacheContract(unittest.TestCase):
    def setUp(self):
        self.item = np.ones(1152, np.float32) / np.sqrt(1152)

    def test_real_numpy2_cache_and_overlap(self):
        frame = json.loads((SHARED/'inputs/2metro8floor/frames.json').read_text())['frames'][0]
        original = load_objects(SHARED/'features/2metro8floor'/Path(frame['file']).stem/'siglip_objects.pkl')
        gobs = read_gobs('2metro8floor', frame, self.item)
        self.assertEqual(len(original), 14)
        self.assertTrue((gobs['mask'].sum(0)>1).any())
        for i, obj in enumerate(original):
            np.testing.assert_array_equal(gobs['image_feats'][i], obj['embedding'])
            np.testing.assert_array_equal(gobs['mask'][i], obj['mask'])
        self.assertEqual(gobs['classes'], ['item'])

    def test_nearest_resize_and_box_scaling_preserve_features(self):
        frame = json.loads((SHARED/'inputs/2metro8floor/frames.json').read_text())['frames'][0]
        gobs = read_gobs('2metro8floor', frame, self.item)
        before = copy.deepcopy(gobs)
        resized = resize_gobs(gobs, np.zeros((504,504,3), np.uint8))
        for old, new in zip(before['mask'], resized['mask']):
            np.testing.assert_array_equal(new, cv2.resize(old.astype(np.uint8),(504,504), interpolation=cv2.INTER_NEAREST).astype(bool))
        np.testing.assert_array_equal(before['image_feats'], resized['image_feats'])
        np.testing.assert_array_equal(resized['xyxy'], np.round(before['xyxy']*(504/800)))

    def test_empty_cache_creates_no_background(self):
        gobs = to_gobs([], self.item)
        self.assertEqual(gobs['mask'].shape, (0,800,800))
        self.assertEqual(gobs['image_feats'].shape, (0,1152))

    def test_run_id_not_new_traversal(self):
        frames = [dict(run=r,timestamp_ns=t,new_traversal=True) for r,t in [(1,9),(1,10),(2,1),(3,0)]]
        self.assertEqual(validate_frames(frames), {1:2,2:3,3:4})
        self.assertEqual(frame_run(frames[0]),1)
        with self.assertRaises(ValueError):
            validate_frames(list(reversed(frames)))

    def test_existing_graph_vectors_no_transformers(self):
        vectors, metadata = query_vectors()
        self.assertEqual(metadata['graph_prompt'],'a photo of a {query}')
        self.assertEqual(len(vectors),15)
        self.assertEqual(next(iter(vectors.values())).shape,(1152,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
