#!/bin/bash
# ConceptGraphs Variant A: SAM "segment all" + per-mask CLIP features.
# class_set=none triggers dense SAM mode without any open-vocab detector.
#
# Outputs (under <DATASET_ROOT>/<scene>/):
#   gsa_detections_none/<frame>.pkl.gz
#   gsa_classes_none.json
#   gsa_vis_none/<frame>.jpg
#
# Requires GSA_PATH env var set to the Grounded-Segment-Anything checkout
# (only sam_vit_h_4b8939.pth is needed for class_set=none).

set -e

DATASET_NAME="${1:-pantry2}"
DATASET_ROOT="/local_data/jz4725/Depth-Anything-3/metric_depth_results"
DATASET_CONFIG="${DATASET_CONFIG:-/local_data/jz4725/concept-graphs/conceptgraph/dataset/dataconfigs/scannet/metacam_504.yaml}"
STRIDE="${STRIDE:-1}"
GPU_ID="${GPU_ID:-7}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python conceptgraph/scripts/generate_gsa_results.py \
  --dataset_root "$DATASET_ROOT" \
  --dataset_config "$DATASET_CONFIG" \
  --scene_id "$DATASET_NAME" \
  --class_set none \
  --stride "$STRIDE"
