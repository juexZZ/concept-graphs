import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import gzip
import pickle
import numpy as np
from baseline_io import save_result
from eval_object_retrieval import ConceptGraphsRetriever, ObjectEvaluator


class BaselineTests(unittest.TestCase):
    def test_empty_map_initialization_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with gzip.open(root / "map.pkl.gz", "wb") as f:
                pickle.dump({"objects": []}, f)
            (root / "mapping.json").write_text("{}")
            (root / "stream_annotations.json").write_text(json.dumps({
                "labels": ["chair"], "images": [
                    {"file": "left_0.png", "new_traversal": False, "annotations": [{"category": "chair"}]},
                    {"file": "right_0.png", "new_traversal": False, "annotations": []}]}))
            with patch("eval_object_retrieval.open_clip.create_model_and_transforms", return_value=(MagicMock(), None, None)), patch("eval_object_retrieval.open_clip.get_tokenizer"):
                retriever = ConceptGraphsRetriever(str(root / "map.pkl.gz"), str(root / "mapping.json"), device="cpu")
            self.assertEqual(len(retriever.query_top_instance(np.ones((1, 4)))[0]), 0)
            evaluator = ObjectEvaluator("test", str(root), retriever)
            result = evaluator.evaluate(1, ["chair", "absent"])
            self.assertEqual(result["retrieval_results"]["chair"]["status"], "miss")
            self.assertEqual(result["retrieval_results"]["absent"]["status"], "True_Negative")
            with self.assertRaises(ValueError):
                evaluator.evaluate(99, ["chair"])

    def test_latest_contributors_are_unique_numeric_frames(self):
        retriever = ConceptGraphsRetriever.__new__(ConceptGraphsRetriever)
        retriever.objects = [{"color_path": ["10.jpg", "2.jpg", "10.jpg", "100.jpg"]}]
        retriever.idx_to_orig = {2: "left_2.png", 10: "left_10.png", 100: "left_100.png"}
        self.assertEqual(retriever.get_top_t_latest_frames(0, 2), [("left_100.png", 100), ("left_10.png", 10)])

    def test_plain_output_path_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                save_result("result.json", {"v": 1})
                with self.assertRaises(FileExistsError):
                    save_result("result.json", {"v": 2})
                self.assertEqual(json.loads(Path("result.json").read_text()), {"v": 1})
            finally:
                os.chdir(old)

    def test_stage_runners_share_settings_and_route_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "python"
            log = root / "args.jsonl"
            fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['ARG_LOG'], 'a') as f: print(json.dumps(sys.argv[1:]), file=f)\n")
            fake.chmod(0o755)
            env = {**os.environ, "PYTHON": str(fake), "ARG_LOG": str(log), "DATASET_ROOT": str(root / "data"), "RUN_ID": "smoke", "ANNOTATION_ROOT": str(root / "annotations"), "OBJECT_LIST_ROOT": str(root / "queries")}
            for script in ("run_cg_mapping.sh", "run_cg_evaluator.sh"):
                subprocess.run(["bash", str(ROOT / script), "pantry1"], env=env, cwd=tmp, check=True)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            mapping, evaluation = commands
            suffix = next(v.split("=", 1)[1] for v in mapping if v.startswith("save_suffix="))
            self.assertIn("mask_conf_threshold=0.95", mapping)
            self.assertIn("scene_id=pantry1", mapping)
            self.assertIn(suffix, evaluation[evaluation.index("--map_pkl") + 1])
            self.assertIn(suffix, evaluation[evaluation.index("--output_path") + 1])
            self.assertIn("pantry1", evaluation[evaluation.index("--map_pkl") + 1])
            env.update(ONLY="pantry1", STAGE="eval")
            subprocess.run(["bash", str(ROOT / "run_cg_all.sh")], env=env, cwd=tmp, check=True)
            command = json.loads(log.read_text().splitlines()[-1])
            self.assertIn("pantry1", command[command.index("--map_pkl") + 1])


    def test_run_manifest_records_failure_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "run.json"
            command = [sys.executable, str(ROOT / "scripts/record_run.py"), "--manifest", str(manifest), "--", sys.executable, "-c", "raise SystemExit(3)"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            first = manifest.read_bytes()
            self.assertEqual(json.loads(first)["status"], "failed")
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(manifest.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
