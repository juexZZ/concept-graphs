#!/bin/bash
# Run ConceptGraphs Variant A end-to-end on every dataset:
#   1) generate_gsa_results.py  (SAM segment-all + per-mask CLIP)
#   2) cfslam_pipeline_batch.py (3D object map -> pcd_saves/*.pkl.gz)
#   3) eval_object_retrieval.py (metrics vs stream_annotations.json)
#
# Usage:
#   GPU_ID=7 ./run_cg_all.sh                 # detect + map + eval all
#   GPU_ID=7 STAGE=detect ./run_cg_all.sh    # only run stage 1
#   GPU_ID=7 STAGE=map    ./run_cg_all.sh    # only stage 2
#   GPU_ID=7 STAGE=eval   ./run_cg_all.sh    # only stage 3
#   GPU_ID=7 ONLY="pantry1 pantry2" ./run_cg_all.sh   # subset of datasets

set -e

GPU_ID="${GPU_ID:-0}"
STRIDE="${STRIDE:-1}"
THRESHOLD="${THRESHOLD:-1.2}"
MASK_CONF_THRESHOLD="${MASK_CONF_THRESHOLD:-0.95}"
FILTER_INTERVAL="${FILTER_INTERVAL:-10}"
OBJ_MIN_DETECTIONS="${OBJ_MIN_DETECTIONS:-2}"
if [ "$STRIDE" != "1" ]; then STRIDE_SFX="_stride${STRIDE}"; else STRIDE_SFX=""; fi
SAVE_SUFFIX="${SAVE_SUFFIX:-overlap_maskconf${MASK_CONF_THRESHOLD}_simsum${THRESHOLD}_dbscan.1_merge20${STRIDE_SFX}}"
STAGE="${STAGE:-all}"   # all | detect | map | eval

# Datasets and the chapter id their annotation maps to.
# Format: "dataset_name chapter_id"
DATASETS=(
  "12thfloor 1"
  "12thfloor_first150 1"
  "12thfloor_trav_1 1"
  "28_metfloor 3"
  "28_metfloor_first150 1"
  "28_metfloor_trav_1 1"
  "28_metfloor_trav_12 2"
  "2ndfloor 1"
  "2ndfloor_first150 1"
  "2ndfloor_trav_1 1"
  "pantry1 1"
  "pantry2 1"
  "pantry_2traverse 2"
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# allow ONLY="a b c" to subset
declare -A KEEP
if [[ -n "${ONLY:-}" ]]; then
  for d in $ONLY; do KEEP["$d"]=1; done
fi

run_one_detect() {
  local NAME="$1"
  echo "===== detect: $NAME ====="
  GPU_ID="$GPU_ID" STRIDE="$STRIDE" ./run_cg_detect.sh "$NAME"
}

run_one_map() {
  local NAME="$1"
  echo "===== map: $NAME ====="
  GPU_ID="$GPU_ID" STRIDE="$STRIDE" THRESHOLD="$THRESHOLD" \
    MASK_CONF_THRESHOLD="$MASK_CONF_THRESHOLD" \
    FILTER_INTERVAL="$FILTER_INTERVAL" \
    OBJ_MIN_DETECTIONS="$OBJ_MIN_DETECTIONS" \
    SAVE_SUFFIX="$SAVE_SUFFIX" \
    ./run_cg_mapping.sh "$NAME"
}

run_one_eval() {
  local NAME="$1"
  local CH="$2"
  echo "===== eval: $NAME ch=$CH ====="
  GPU_ID="$GPU_ID" CHAPTER_ID="$CH" STRIDE="$STRIDE" THRESHOLD="$THRESHOLD" \
    MASK_CONF_THRESHOLD="$MASK_CONF_THRESHOLD" SAVE_SUFFIX="$SAVE_SUFFIX" \
    ./run_cg_evaluator.sh "$NAME"
}

for entry in "${DATASETS[@]}"; do
  read -r NAME CH <<< "$entry"
  if [[ -n "${ONLY:-}" && -z "${KEEP[$NAME]:-}" ]]; then
    continue
  fi
  case "$STAGE" in
    detect) run_one_detect "$NAME" ;;
    map)    run_one_map "$NAME" ;;
    eval)   run_one_eval "$NAME" "$CH" ;;
    all)
      run_one_detect "$NAME"
      run_one_map    "$NAME"
      run_one_eval   "$NAME" "$CH"
      ;;
    *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
  esac
done

echo "all done."
