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

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/dejaview_env.sh"
QUERY="${2:-}"
PORT="${PORT:-8080}"
FULL_PKL="${FULL_PKL:-$DATASET_ROOT/$DATASET_NAME/pcd_saves/full_pcd_none_${SAVE_SUFFIX}.pkl.gz}"

if [[ ! -f "$MAP_PKL" ]]; then
  echo "missing post-filter map: $MAP_PKL" >&2
  echo "run ./run_cg_mapping.sh $DATASET_NAME first." >&2
  exit 1
fi

ARGS=(--map_pkl "$MAP_PKL" --mapping "$MAPPING" --port "$PORT" --color_dir "$DATASET_ROOT/$DATASET_NAME/color")
[[ -f "$FULL_PKL" ]] && ARGS+=(--full_pkl "$FULL_PKL")
[[ -n "$QUERY" ]] && ARGS+=(--initial_query "$QUERY")

echo "viser viewer for $DATASET_NAME on port $PORT"
echo "  map_pkl : $MAP_PKL"
echo "  full_pkl: ${FULL_PKL:-(none)}"
echo "  open in browser: http://<host>:${PORT}"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/visualize_query.py "${ARGS[@]}"
