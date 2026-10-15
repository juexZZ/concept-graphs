#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/dejaview_env.sh"
: "${ANNOTATION_ROOT:?Set ANNOTATION_ROOT to the parent of annotation scene directories}"
: "${OBJECT_LIST_ROOT:?Set OBJECT_LIST_ROOT to DejaView/eval/object_test_lists}"
CHAPTER_ID="${CHAPTER_ID:-1}"
DATASET_DIR="${DATASET_DIR:-$ANNOTATION_ROOT/$DATASET_NAME}"
case "$DATASET_NAME" in
  pantry*) OBJECT_LIST_TXT="${OBJECT_LIST_TXT:-$OBJECT_LIST_ROOT/pantry.txt}" ;;
  *) OBJECT_LIST_TXT="${OBJECT_LIST_TXT:-$OBJECT_LIST_ROOT/$DATASET_NAME.txt}" ;;
esac
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT/eval_results/runs/$SAVE_SUFFIX/cg_${DATASET_NAME}_ch${CHAPTER_ID}.json}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/eval_object_retrieval.py   --map_pkl "$MAP_PKL" --mapping "$MAPPING"   --dataset_name "$DATASET_NAME" --dataset_dir "$DATASET_DIR"   --object_list_txt "$OBJECT_LIST_TXT" --chapter_id "$CHAPTER_ID"   --output_path "$OUTPUT_PATH" --contributor_top_t "${CONTRIBUTOR_TOP_T:-1}"   --similarity_threshold -1.0 --ignore_similarity_threshold
