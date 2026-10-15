#!/bin/bash
# SAM segment-all + per-mask CLIP; set GSA_PATH per upstream installation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/dejaview_env.sh"
DETECTIONS="$DATASET_ROOT/$DATASET_NAME/gsa_detections_none"
if [[ -e "$DETECTIONS" ]]; then
  if [[ "${REUSE_DETECTIONS:-0}" == 1 ]]; then
    echo "Explicitly reusing detection cache: $DETECTIONS"
    exit 0
  fi
  echo "Detection cache exists: $DETECTIONS. Set REUSE_DETECTIONS=1 after checking its provenance." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/record_run.py   --manifest "$DATASET_ROOT/$DATASET_NAME/gsa_detections_none.run.json" --   "$PYTHON" conceptgraph/scripts/generate_gsa_results.py   --dataset_root "$DATASET_ROOT" --dataset_config "$DATASET_CONFIG"   --scene_id "$DATASET_NAME" --class_set none --stride "$STRIDE"
