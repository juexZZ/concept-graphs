#!/bin/bash
# ConceptGraphs Variant A 3D mapping pipeline (after run_cg_detect.sh).
# Reads gsa_detections_none/ and produces an object map at
#   <DATASET_ROOT>/<scene>/pcd_saves/full_pcd_none_<SAVE_SUFFIX>.pkl.gz
# along with the post-processed companion (...post.pkl.gz). Use the _post one.

set -e

DATASET_NAME="${1:-pantry2}"
DATASET_ROOT="/local_data/jz4725/Depth-Anything-3/metric_depth_results"
DATASET_CONFIG="${DATASET_CONFIG:-/local_data/jz4725/concept-graphs/conceptgraph/dataset/dataconfigs/scannet/metacam_504.yaml}"
STRIDE="${STRIDE:-1}"
THRESHOLD="${THRESHOLD:-1.2}"
MASK_CONF_THRESHOLD="${MASK_CONF_THRESHOLD:-0.2}"
OBJ_MIN_DETECTIONS="${OBJ_MIN_DETECTIONS:-2}"
FILTER_INTERVAL="${FILTER_INTERVAL:-10}"   # prune singletons every N frames; -1 to disable
MERGE_OVERLAP_THRESH="${MERGE_OVERLAP_THRESH:-0.7}"        # 3D overlap to merge two existing objects
MERGE_VISUAL_SIM_THRESH="${MERGE_VISUAL_SIM_THRESH:-0.8}"  # CLIP-image cosine gate for above
MERGE_TEXT_SIM_THRESH="${MERGE_TEXT_SIM_THRESH:-0.8}"      # CLIP-text cosine gate for above
GPU_ID="${GPU_ID:-7}"
# Stride suffix: empty for stride=1 (backward compat), e.g. "_stride5" otherwise.
if [ "$STRIDE" != "1" ]; then STRIDE_SFX="_stride${STRIDE}"; else STRIDE_SFX=""; fi
SAVE_SUFFIX="${SAVE_SUFFIX:-overlap_maskconf${MASK_CONF_THRESHOLD}_simsum${THRESHOLD}_dbscan.1_merge20${STRIDE_SFX}}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python conceptgraph/slam/cfslam_pipeline_batch.py \
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
  merge_text_sim_thresh="$MERGE_TEXT_SIM_THRESH"
