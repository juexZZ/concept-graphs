#!/bin/bash
# Evaluate object retrieval against a ConceptGraphs Variant-A map.

set -e

DATASET_NAME="${1:-pantry2}"
CHAPTER_ID="${CHAPTER_ID:-1}"
GPU_ID="${GPU_ID:-7}"
STRIDE="${STRIDE:-1}"
THRESHOLD="${THRESHOLD:-1.2}"
MASK_CONF_THRESHOLD="${MASK_CONF_THRESHOLD:-0.2}"
if [ "$STRIDE" != "1" ]; then STRIDE_SFX="_stride${STRIDE}"; else STRIDE_SFX=""; fi
SAVE_SUFFIX="${SAVE_SUFFIX:-overlap_maskconf${MASK_CONF_THRESHOLD}_simsum${THRESHOLD}_dbscan.1_merge20${STRIDE_SFX}}"
# Output also tagged with stride so different runs don't overwrite each other.
OUT_STRIDE_SFX=""; [ "$STRIDE" != "1" ] && OUT_STRIDE_SFX="_stride${STRIDE}"

DATASET_ROOT="/local_data/jz4725/Depth-Anything-3/metric_depth_results"
MAP_PKL="${DATASET_ROOT}/${DATASET_NAME}/pcd_saves/full_pcd_none_${SAVE_SUFFIX}_post.pkl.gz"
MAPPING="${DATASET_ROOT}/${DATASET_NAME}/filename_mapping.json"

# pantry datasets use the rebuttal annotation tree + a shared object list
case "$DATASET_NAME" in
  pantry*)
    DATASET_DIR="/local_data/jz4725/metacam/rebuttal/${DATASET_NAME}"
    OBJECT_LIST_TXT="/local_data/jz4725/DejaView/eval/object_test_lists/pantry.txt"
    ;;
  *)
    DATASET_DIR="/local_data/jz4725/metacam/data_v3/${DATASET_NAME}"
    OBJECT_LIST_TXT="/local_data/jz4725/DejaView/eval/object_test_lists/${DATASET_NAME}.txt"
    ;;
esac

OUTPUT_PATH="/local_data/jz4725/concept-graphs/eval_results/cg_${DATASET_NAME}_ch${CHAPTER_ID}${OUT_STRIDE_SFX}.json"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/eval_object_retrieval.py \
  --map_pkl "$MAP_PKL" \
  --mapping "$MAPPING" \
  --dataset_name "$DATASET_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --object_list_txt "$OBJECT_LIST_TXT" \
  --chapter_id "$CHAPTER_ID" \
  --output_path "$OUTPUT_PATH" \
  --similarity_threshold -1.0 \
  --ignore_similarity_threshold
