#!/bin/bash
# Launch viser-based interactive viewer for a ConceptGraphs map.
# Renders the post-filtered queryable instances + (optional) full pre-filter
# scene cloud as a dim background. Type a text query and the top-1 instance
# gets highlighted plus its latest contributor frame is shown.
#
# Usage:
#   GPU_ID=5 ./run_cg_visualize.sh pantry2
#   PORT=8081 GPU_ID=5 ./run_cg_visualize.sh 12thfloor "yellow trash bin"
#
# Open http://<host>:$PORT in a browser (forward via SSH if remote).

set -e

DATASET_NAME="${1:-pantry2}"
QUERY="${2:-}"
DATASET_ROOT="/local_data/jz4725/Depth-Anything-3/metric_depth_results"
THRESHOLD="${THRESHOLD:-1.2}"
MASK_CONF_THRESHOLD="${MASK_CONF_THRESHOLD:-0.2}"
STRIDE="${STRIDE:-1}"
if [ "$STRIDE" != "1" ]; then STRIDE_SFX="_stride${STRIDE}"; else STRIDE_SFX=""; fi
SAVE_SUFFIX="${SAVE_SUFFIX:-overlap_maskconf${MASK_CONF_THRESHOLD}_simsum${THRESHOLD}_dbscan.1_merge20${STRIDE_SFX}}"
PORT="${PORT:-8080}"
GPU_ID="${GPU_ID:-7}"

MAP_PKL="${DATASET_ROOT}/${DATASET_NAME}/pcd_saves/full_pcd_none_${SAVE_SUFFIX}_post.pkl.gz"
FULL_PKL="${DATASET_ROOT}/${DATASET_NAME}/pcd_saves/full_pcd_none_${SAVE_SUFFIX}.pkl.gz"
MAPPING="${DATASET_ROOT}/${DATASET_NAME}/filename_mapping.json"

if [[ ! -f "$MAP_PKL" ]]; then
  echo "missing post-filter map: $MAP_PKL" >&2
  echo "run ./run_cg_mapping.sh $DATASET_NAME first." >&2
  exit 1
fi

ARGS=(--map_pkl "$MAP_PKL" --mapping "$MAPPING" --port "$PORT")
[[ -f "$FULL_PKL" ]] && ARGS+=(--full_pkl "$FULL_PKL")
[[ -n "$QUERY" ]] && ARGS+=(--initial_query "$QUERY")

echo "viser viewer for $DATASET_NAME on port $PORT"
echo "  map_pkl : $MAP_PKL"
echo "  full_pkl: ${FULL_PKL:-(none)}"
echo "  open in browser: http://<host>:${PORT}"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/visualize_query.py "${ARGS[@]}"
