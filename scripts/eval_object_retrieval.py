"""
Object retrieval evaluation on a ConceptGraphs map (Variant A: SAM segment-all + CLIP).

Mirrors HOV-SG/application/eval_object_retrieval.py and DejaView/semantic_voxels/evaluator.py:
metrics, chapter parsing, hit/miss/ghost/TN logic, top-t contributors. The only
ConceptGraphs-specific part is the retriever:
  - Loads <map_pkl> (gzipped pickle) -> dict with 'objects' = list of dicts
  - Each object stores 'clip_ft' (np array), 'image_idx' (list[int]),
    'color_path' (list[str]) accumulated through merges.
  - Top-1 instance picked by cosine similarity vs text embedding (OpenCLIP ViT-H-14).
  - Top-t latest contributor frames = unique color_path stems sorted descending.
  - Stem -> original filename via filename_mapping.json (same as HOV-SG eval).
"""
from __future__ import annotations
import os
import json
import gzip
import pickle
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import open_clip
from tqdm import tqdm


def _split_left_right(images: Iterable[dict]):
    left, right = [], []
    for e in images:
        fn = e.get("file")
        if not isinstance(fn, str):
            continue
        if fn.startswith("left_"):
            left.append(e)
        elif fn.startswith("right_"):
            right.append(e)
    return left, right


class ChapterAnnotation:
    def __init__(self, annotation_file: str, dataset_dir: str):
        with open(annotation_file) as f:
            self.annotation = json.load(f)
        self.left_right = "realsense" not in dataset_dir
        self.present_objects = self.annotation["labels"]
        self.annotation_by_chapter: dict[int, dict[str, list[str]]] = {}
        self._parse()

    def _parse(self):
        chapter_id = 1
        self.annotation_by_chapter[chapter_id] = {o: [] for o in self.present_objects}
        if self.left_right:
            left_images, right_images = _split_left_right(self.annotation["images"])
            for i, left in enumerate(left_images):
                if left["new_traversal"]:
                    chapter_id += 1
                    self.annotation_by_chapter[chapter_id] = {o: [] for o in self.present_objects}
                left_anns = set(a["category"] for a in left.get("annotations", []))
                right = right_images[i]
                right_anns = set(a["category"] for a in right.get("annotations", []))
                for o in left_anns:
                    self.annotation_by_chapter[chapter_id][o].append(left["file"])
                for o in right_anns:
                    self.annotation_by_chapter[chapter_id][o].append(right["file"])
        else:
            for image in self.annotation["images"]:
                if image.get("new_traversal"):
                    chapter_id += 1
                    self.annotation_by_chapter[chapter_id] = {o: [] for o in self.present_objects}
                anns = set(a["category"] for a in image.get("annotations", []))
                for o in anns:
                    self.annotation_by_chapter[chapter_id][o].append(image["file"])
        self.total_chapters = chapter_id


def load_filename_mapping(path: str) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        out = {}
        for e in raw:
            idx = int(e["index"])
            orig = e.get("original") or e.get("original_filename")
            out[idx] = orig
        return out
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise ValueError("unsupported filename_mapping format")


def _stem_int(p: str) -> int:
    return int(Path(p).stem)


class ConceptGraphsRetriever:
    def __init__(self, map_pkl: str, mapping_path: str,
                 clip_type: str = "ViT-H-14",
                 clip_pretrained: str = "laion2b_s32b_b79k",
                 device: str = "cuda"):
        self.map_pkl = map_pkl
        self.device = device
        self.clip_type = clip_type
        self.clip_pretrained = clip_pretrained

        with gzip.open(map_pkl, "rb") as f:
            results = pickle.load(f)
        # results: {'objects': [...], 'bg_objects': [...] | None, 'cfg': ..., 'class_names': [...]}
        self.objects = results["objects"]
        # Stack clip features (D,) -> (N, D)
        feats = np.stack([np.asarray(o["clip_ft"], dtype=np.float32).reshape(-1) for o in self.objects])
        # L2-normalize defensively
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.clip_feats = feats / norms

        self.idx_to_orig = load_filename_mapping(mapping_path)

        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_type, pretrained=clip_pretrained, device=device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_type)

    def get_text_embedding(self, text: str) -> np.ndarray:
        prompt = f"a photo of {text}"
        with torch.no_grad():
            tokens = self.tokenizer([prompt]).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()  # (1, D)

    def query_top_instance(self, text_feats: np.ndarray, top_k: int = 1):
        sims = (self.clip_feats @ text_feats.T).reshape(-1)
        order = np.argsort(-sims)[:top_k]
        return order, sims[order]

    def get_top_t_latest_frames(self, instance_id: int, top_t: int = 1):
        """Return list of (orig_filename, local_stem). Latest first by descending stem."""
        obj = self.objects[int(instance_id)]
        color_paths = obj.get("color_path", [])
        seen = set()
        unique_stems = []
        for p in color_paths:
            try:
                s = _stem_int(p)
            except ValueError:
                continue
            if s in seen:
                continue
            seen.add(s)
            unique_stems.append(s)
        unique_stems.sort(reverse=True)
        out = []
        for s in unique_stems[:top_t]:
            out.append((self.idx_to_orig.get(s), s))
        return out


class ObjectEvaluator:
    def __init__(self, dataset_name: str, dataset_dir: str,
                 retriever: ConceptGraphsRetriever,
                 similarity_threshold: float = 0.2,
                 contributor_top_t: int = 1,
                 ignore_similarity_threshold: bool = False):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.contributor_top_t = max(1, contributor_top_t)
        self.ignore_similarity_threshold = ignore_similarity_threshold
        self.chapters = ChapterAnnotation(
            os.path.join(dataset_dir, "stream_annotations.json"), dataset_dir
        )

    def _evaluate_single(self, query: str, chapter_ann: dict, query_present: bool):
        text_feats = self.retriever.get_text_embedding(query)
        order, sims = self.retriever.query_top_instance(text_feats, top_k=1)
        if len(order) == 0:
            return {"hit_any_contributor": False, "hit_primary": False, "hit@1": False,
                    "detected": False, "results": []}
        instance_id = int(order[0])
        similarity = float(sims[0])
        contributors = self.retriever.get_top_t_latest_frames(instance_id, top_t=self.contributor_top_t)
        detected = similarity > self.similarity_threshold
        hit_primary = False
        hit_any = False
        results = []
        if not contributors:
            results.append({
                "instance_id": instance_id, "similarity": similarity, "score": similarity,
                "score_type": "cosine_similarity",
                "frame_name": None, "frame_index": None, "contributor_frames": [],
            })
        else:
            primary_fn, primary_idx = contributors[0]
            contributor_frames = [
                {"frame_name": fn, "frame_index": int(fi)} for fn, fi in contributors
            ]
            results.append({
                "instance_id": instance_id, "similarity": similarity, "score": similarity,
                "score_type": "cosine_similarity",
                "frame_name": primary_fn, "frame_index": int(primary_idx),
                "contributor_frames": contributor_frames,
            })
            if query_present and (self.ignore_similarity_threshold or similarity > self.similarity_threshold):
                gt = set(chapter_ann.get(query, []))
                if primary_fn is not None and primary_fn in gt:
                    hit_primary = True
                if any(fn is not None and fn in gt for fn, _ in contributors):
                    hit_any = True
        return {"hit_any_contributor": hit_any, "hit_primary": hit_primary,
                "hit@1": hit_primary, "detected": detected, "results": results}

    def evaluate(self, chapter_id: int, object_list: list[str]):
        retrieval_results = {}
        hit_any_count = miss_any_count = 0
        hit_primary_count = miss_primary_count = 0
        ghost_count = 0
        chapter_ann = self.chapters.annotation_by_chapter.get(chapter_id, {})
        n_gt_in_chapter = sum(1 for files in chapter_ann.values() if files)

        for query in tqdm(object_list, desc=f"chapter {chapter_id}"):
            query_present = bool(chapter_ann.get(query, ()))
            qr = self._evaluate_single(query, chapter_ann, query_present)
            detected = qr["detected"]
            hit_any = qr["hit_any_contributor"]
            hit_primary = qr["hit_primary"]
            if query_present:
                if hit_any:
                    status = "hit"; hit_any_count += 1
                else:
                    status = "miss"; miss_any_count += 1
                if hit_primary: hit_primary_count += 1
                else: miss_primary_count += 1
            else:
                if detected:
                    status = "ghost"; ghost_count += 1
                else:
                    status = "True_Negative"
            qr["status"] = status
            qr["present_in_chapter"] = query_present
            retrieval_results[query] = qr

        total = max(len(object_list), 1)
        tp, fn, fp = hit_any_count, miss_any_count, ghost_count
        tn = len(object_list) - tp - fn - fp
        chapter_positive = hit_any_count + miss_any_count
        denom = max(chapter_positive, 1)
        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "dataset_name": self.dataset_name,
            "chapter_id": chapter_id,
            "object_list": object_list,
            "retrieval_results": retrieval_results,
            "hit_rate_any_contributor": hit_any_count / total,
            "hit_rate(TP)": hit_any_count / total,
            "miss_rate(FN)": miss_any_count / total,
            "hit_rate_primary": hit_primary_count / total,
            "miss_rate_primary(FN)": miss_primary_count / total,
            "hit@1": hit_primary_count / total,
            "ghost_rate(FP)": ghost_count / total,
            "True_Negative(TN)": tn / total,
            "In chapter hit (any contributor)": hit_any_count / denom,
            "In chapter miss (any contributor)": miss_any_count / denom,
            "In chapter hit (primary only)": hit_primary_count / denom,
            "In chapter miss (primary only)": miss_primary_count / denom,
            "recall": recall, "precision": precision, "f1_score": f1,
            "meta": {
                "similarity_threshold": self.similarity_threshold,
                "ignore_similarity_threshold_for_hits": self.ignore_similarity_threshold,
                "contributor_top_t": self.contributor_top_t,
                "tp_fn_criterion": "any_contributor_in_gt",
                "primary_metric_alias_hit@1": "hit_rate_primary",
                "n_gt_objects_in_chapter_annotation": n_gt_in_chapter,
                "n_evaluated_positive_queries": chapter_positive,
                "in_chapter_rates_denominator": "object_list queries with GT in this chapter",
                "score_type": "cosine_similarity",
                "map_pkl": self.retriever.map_pkl,
                "clip_type": self.retriever.clip_type,
                "clip_pretrained": self.retriever.clip_pretrained,
                "model": "conceptgraphs_variant_a",
            },
        }


def _load_object_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map_pkl", required=True, help="path to ConceptGraphs output .pkl.gz (use the _post one)")
    p.add_argument("--mapping", required=True, help="filename_mapping.json for this scene")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_dir", required=True, help="dir with stream_annotations.json")
    p.add_argument("--object_list_txt", required=True)
    p.add_argument("--chapter_id", type=int, required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--clip_type", default="ViT-H-14")
    p.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    p.add_argument("--device", default="cuda")
    p.add_argument("--similarity_threshold", type=float, default=0.2)
    p.add_argument("--ignore_similarity_threshold", action="store_true")
    p.add_argument("--contributor_top_t", type=int, default=1)
    args = p.parse_args()

    retriever = ConceptGraphsRetriever(
        map_pkl=args.map_pkl, mapping_path=args.mapping,
        clip_type=args.clip_type, clip_pretrained=args.clip_pretrained, device=args.device,
    )
    object_list = _load_object_list(args.object_list_txt)
    evaluator = ObjectEvaluator(
        dataset_name=args.dataset_name, dataset_dir=args.dataset_dir, retriever=retriever,
        similarity_threshold=args.similarity_threshold,
        contributor_top_t=args.contributor_top_t,
        ignore_similarity_threshold=args.ignore_similarity_threshold,
    )
    results = evaluator.evaluate(args.chapter_id, object_list)

    print("***************** evaluation results *****************")
    for k, v in results.items():
        if k == "retrieval_results":
            for obj, r in v.items():
                print(f"Object: {obj}")
                for kk, vv in r.items():
                    print(f"  {kk}: {vv}")
                print("--------------------------------")
        print(f"{k}: {v}")
    print("*******************************************************")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
