"""
Interactive viser visualizer for a ConceptGraphs map.

- Renders every instance pcd with its own per-instance color.
- Text query box: encodes via OpenCLIP, finds top-1 instance, highlights it red,
  dims everything else, prints + displays the latest contributor frame.

Usage:
    python scripts/visualize_query.py \
        --map_pkl /local_data/jz4725/Depth-Anything-3/metric_depth_results/pantry1/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_post.pkl.gz \
        --mapping /local_data/jz4725/Depth-Anything-3/metric_depth_results/pantry1/filename_mapping.json \
        --port 8080
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import viser
from PIL import Image


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
    return {int(k): v for k, v in raw.items()}


def stem_int(p: str) -> int:
    return int(Path(p).stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map_pkl", required=True, help="post-processed map (queryable instances)")
    p.add_argument("--mapping", required=True)
    p.add_argument("--full_pkl", default=None,
                   help="optional: pre-post pkl.gz; its concatenated pcds shown as dim scene background")
    p.add_argument("--color_dir", default=None, help="Relocate stored contributor image paths to this color directory")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--clip_type", default="ViT-H-14")
    p.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    p.add_argument("--device", default="cuda")
    p.add_argument("--point_size", type=float, default=0.01)
    p.add_argument("--bg_point_size", type=float, default=0.005)
    p.add_argument("--bg_dim", type=float, default=0.55, help="rgb multiplier for background cloud")
    p.add_argument("--initial_query", default="remote")
    args = p.parse_args()

    with gzip.open(args.map_pkl, "rb") as f:
        results = pickle.load(f)
    objects = results["objects"]
    print(f"loaded {len(objects)} objects from {args.map_pkl}")

    if not objects:
        raise SystemExit("Map contains no queryable objects")

    # Stack + L2-normalize CLIP feats
    feats = np.stack(
        [np.asarray(o["clip_ft"], dtype=np.float32).reshape(-1) for o in objects]
    )
    norms = np.linalg.norm(feats, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = feats / norms

    idx_to_orig = load_filename_mapping(args.mapping)

    # Per-instance base color (from inst_color or fallback)
    base_colors = []
    for o in objects:
        c = np.asarray(o.get("inst_color", np.random.rand(3))).reshape(-1)[:3]
        base_colors.append(np.clip(c, 0.0, 1.0))

    # CLIP
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_type, pretrained=args.clip_pretrained, device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.clip_type)

    def encode_text(text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = tokenizer([f"a photo of {text}"]).to(args.device)
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        return tf.cpu().numpy()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.world_axes.visible = True

    # Optional: render the larger pre-filter scene cloud as a dim background
    bg_handle = None
    if args.full_pkl is not None and os.path.exists(args.full_pkl):
        with gzip.open(args.full_pkl, "rb") as f:
            full_results = pickle.load(f)
        bg_pts_list, bg_col_list = [], []
        for o in full_results["objects"]:
            pts = np.asarray(o["pcd_np"], dtype=np.float32)
            if len(pts) == 0:
                continue
            cols = o.get("pcd_color_np")
            if cols is None or len(cols) != len(pts):
                cols = np.tile(np.asarray(o.get("inst_color", [0.5, 0.5, 0.5]))[None, :], (len(pts), 1))
            bg_pts_list.append(pts)
            bg_col_list.append(np.clip(np.asarray(cols, dtype=np.float32) * args.bg_dim, 0.0, 1.0))
        if bg_pts_list:
            bg_pts = np.concatenate(bg_pts_list, axis=0)
            bg_cols = (np.concatenate(bg_col_list, axis=0) * 255).astype(np.uint8)
            print(f"loaded full-scene background: {len(bg_pts)} points from {len(bg_pts_list)} pre-filter objects")
            bg_handle = server.scene.add_point_cloud(
                "/scene/full",
                points=bg_pts,
                colors=bg_cols,
                point_size=args.bg_point_size,
            )

    pc_handles = []
    for i, obj in enumerate(objects):
        pts = np.asarray(obj["pcd_np"], dtype=np.float32)
        col = base_colors[i]
        cols = np.tile((col * 255).astype(np.uint8)[None, :], (len(pts), 1))
        h = server.scene.add_point_cloud(
            f"/objects/inst_{i}",
            points=pts,
            colors=cols,
            point_size=args.point_size,
        )
        pc_handles.append(h)

    # GUI
    server.gui.add_markdown(f"### ConceptGraphs viewer\nLoaded **{len(objects)} instances**")
    query_input = server.gui.add_text("query", initial_value=args.initial_query)
    search_btn = server.gui.add_button("Search")
    reset_btn = server.gui.add_button("Reset colors")
    point_size_slider = server.gui.add_slider(
        "point size (instances)", min=0.002, max=0.05, step=0.001, initial_value=args.point_size
    )
    if bg_handle is not None:
        bg_visible = server.gui.add_checkbox("show full scene", initial_value=True)
        bg_point_size_slider = server.gui.add_slider(
            "point size (scene)", min=0.001, max=0.03, step=0.001, initial_value=args.bg_point_size
        )

        @bg_visible.on_update
        def _(_):
            bg_handle.visible = bg_visible.value

        @bg_point_size_slider.on_update
        def _(_):
            bg_handle.point_size = bg_point_size_slider.value

    status = server.gui.add_markdown("_no query yet_")
    image_panel = {"handle": None}

    @point_size_slider.on_update
    def _(_):
        for h in pc_handles:
            h.point_size = point_size_slider.value

    def reset_colors():
        for j, h in enumerate(pc_handles):
            pts_n = len(np.asarray(objects[j]["pcd_np"]))
            cols = np.tile((base_colors[j] * 255).astype(np.uint8)[None, :], (pts_n, 1))
            h.colors = cols

    @reset_btn.on_click
    def _(_):
        reset_colors()
        status.content = "_colors reset_"

    @search_btn.on_click
    def _(_):
        q = query_input.value.strip()
        if not q:
            status.content = "_empty query_"
            return
        tf = encode_text(q)
        sims = (feats @ tf.T).reshape(-1)
        top = int(np.argmax(sims))
        sim = float(sims[top])
        obj = objects[top]

        # latest contributor frame from color_path stems
        cps = [p for p in obj.get("color_path", []) if p]
        latest_path = None
        latest_stem = None
        latest_orig = None
        if cps:
            latest_path = max(cps, key=stem_int)
            latest_stem = stem_int(latest_path)
            latest_orig = idx_to_orig.get(latest_stem)
            if args.color_dir:
                latest_path = str(Path(args.color_dir) / Path(latest_path).name)

        # recolor: dim everything, top-1 in red
        for j, h in enumerate(pc_handles):
            pts_n = len(np.asarray(objects[j]["pcd_np"]))
            if j == top:
                rgb = np.array([1.0, 0.05, 0.05])
            else:
                rgb = base_colors[j] * 0.4
            cols = np.tile((rgb * 255).astype(np.uint8)[None, :], (pts_n, 1))
            h.colors = cols

        msg = (
            f"**query**: `{q}`  \n"
            f"**top-1**: instance **{top}**, sim **{sim:.3f}**  \n"
            f"**latest local frame**: `{latest_stem}.jpg`  \n"
            f"**original filename**: `{latest_orig}`  \n"
            f"**path**: `{latest_path}`"
        )
        status.content = msg

        if latest_path and os.path.exists(latest_path):
            img = np.array(Image.open(latest_path).convert("RGB"))
            if image_panel["handle"] is None:
                image_panel["handle"] = server.gui.add_image(
                    img, label="latest observing frame"
                )
            else:
                image_panel["handle"].image = img

        print(
            f"[query={q!r}] top-1 inst {top}  sim={sim:.3f}  "
            f"latest_orig={latest_orig}  path={latest_path}"
        )

    print(f"\nviser server running on port {args.port}")
    print(f"open in browser: http://<this-host>:{args.port}")
    print("if you're SSH-tunneled, forward this port. Ctrl+C to exit.\n")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
