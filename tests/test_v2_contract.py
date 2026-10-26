import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from v2_common import frames_for, write_new
from v2_retrieve import validate_contributors
from eval_object_retrieval import ConceptGraphsRetriever

class V2ContractTests(unittest.TestCase):
    def test_actual_contributors_not_surrogate_object_last_seen(self):
        retriever = ConceptGraphsRetriever.__new__(ConceptGraphsRetriever)
        retriever.objects = [{'color_path': ['2.jpg', '10.jpg', '2.jpg'], 'image_idx': [999]}]
        retriever.idx_to_orig = {2: 'previous_left.jpg', 10: 'current_right.jpg'}
        frames = [{'index': 2, 'file': 'previous_left.jpg', 'run': 1},
                  {'index': 10, 'file': 'current_right.jpg', 'run': 2}]
        validate_contributors(retriever, frames)
        self.assertEqual(retriever.get_top_t_latest_frames(0, 1), [('current_right.jpg', 10)])

    def test_future_or_wrong_identity_is_rejected(self):
        retriever = ConceptGraphsRetriever.__new__(ConceptGraphsRetriever)
        retriever.objects = [{'color_path': ['10.jpg']}]
        retriever.idx_to_orig = {10: 'wrong_right.jpg'}
        with self.assertRaises(ValueError):
            validate_contributors(retriever, [{'index': 10, 'file': 'true_left.jpg'}])
        with self.assertRaises(ValueError):
            validate_contributors(retriever, [{'index': 9, 'file': 'wrong_right.jpg'}])

    def test_manifest_order_is_not_filename_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'inputs/s'
            images = scene / 'prefix_1/images'
            images.mkdir(parents=True)
            rows = []
            for name in ('z.jpg', 'a.jpg'):
                source = root / name
                source.write_bytes(name.encode())
                (images / name).symlink_to(source)
                rows.append({'file': name, 'path': str(source), 'run': 1,
                             'sha256': hashlib.sha256(name.encode()).hexdigest()})
            (scene / 'frames.json').write_text(json.dumps({'frames': rows}))
            with patch('v2_common.SHARED', root):
                frames = frames_for('s', 1)
            self.assertEqual([f['file'] for f in frames], ['z.jpg', 'a.jpg'])
            self.assertEqual([f['index'] for f in frames], [0, 1])

    def test_completed_artifacts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'complete.json'
            write_new(path, {'ok': True})
            with self.assertRaises(FileExistsError):
                write_new(path, {'ok': False})
            self.assertEqual(json.loads(path.read_text()), {'ok': True})

if __name__ == '__main__':
    unittest.main()
