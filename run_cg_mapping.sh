#!/bin/bash
# ConceptGraphs Variant A 3D mapping pipeline (after run_cg_detect.sh).
# Reads gsa_detections_none/ and produces an object map at
#   <DATASET_ROOT>/<scene>/pcd_saves/full_pcd_none_<SAVE_SUFFIX>.pkl.gz
# along with the post-processed companion (...post.pkl.gz). Use the _post one.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/dejaview_env.sh"
FULL_PKL="$DATASET_ROOT/$DATASET_NAME/pcd_saves/full_pcd_none_${SAVE_SUFFIX}.pkl.gz"
if [[ "$MAP_PKL" != "${FULL_PKL%.pkl.gz}_post.pkl.gz" ]]; then
  echo "For mapping, set SAVE_SUFFIX; MAP_PKL must match the generated filename" >&2
  exit 2
fi
if [[ -e "$MAP_PKL" || -e "$FULL_PKL" ]]; then
  echo "Map exists; choose a new RUN_ID or SAVE_SUFFIX" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/record_run.py \
  --manifest "${MAP_PKL}.run.json" -- \
  "$PYTHON" conceptgraph/slam/cfslam_pipeline_batch.py \
  dataset_root="$DATASET_ROOT" \
  dataset_config="$DATASET_CONFIG" \
  scene_id="$DATASET_NAME" \
  start=0 end=-1 stride="$STRIDE" \
  spatial_sim_type=overlap \
  mask_conf_threshold="$MASK_CONF_THRESHOLD" \
  match_method=sim_sum \
  sim_threshold="$THRESHOLD" \
  dbscan_eps=0.1 \
  gsa_variant=none \
  class_agnostic=True \
  skip_bg=True \
  max_bbox_area_ratio=0.5 \
  obj_min_detections="$OBJ_MIN_DETECTIONS" \
  save_suffix="$SAVE_SUFFIX" \
  merge_interval=20 \
  filter_interval="$FILTER_INTERVAL" \
  merge_overlap_thresh="$MERGE_OVERLAP_THRESH" \
  merge_visual_sim_thresh="$MERGE_VISUAL_SIM_THRESH" \
  merge_text_sim_thresh="$MERGE_TEXT_SIM_THRESH" \
  hydra.run.dir="$ROOT/outputs/$RUN_ID/$DATASET_NAME"
