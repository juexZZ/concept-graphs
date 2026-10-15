# Shared defaults for all DejaView ConceptGraphs stages. Source from a runner.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DATASET_ROOT:?Set DATASET_ROOT to the ScanNet-style export parent}"
PYTHON="${PYTHON:-python}"
DATASET_NAME="${1:-pantry2}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/conceptgraph/dataset/dataconfigs/scannet/metacam_504.yaml}"
GPU_ID="${GPU_ID:-0}"
STRIDE="${STRIDE:-1}"
THRESHOLD="${THRESHOLD:-1.2}"
MASK_CONF_THRESHOLD="${MASK_CONF_THRESHOLD:-0.95}"
OBJ_MIN_DETECTIONS="${OBJ_MIN_DETECTIONS:-2}"
FILTER_INTERVAL="${FILTER_INTERVAL:-10}"
MERGE_OVERLAP_THRESH="${MERGE_OVERLAP_THRESH:-0.7}"
MERGE_VISUAL_SIM_THRESH="${MERGE_VISUAL_SIM_THRESH:-0.8}"
MERGE_TEXT_SIM_THRESH="${MERGE_TEXT_SIM_THRESH:-0.8}"
RUN_ID="${RUN_ID:-manual}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
SAVE_SUFFIX="${SAVE_SUFFIX:-${RUN_ID}_overlap_maskconf${MASK_CONF_THRESHOLD}_simsum${THRESHOLD}_dbscan.1_merge20_stride${STRIDE}_mindet${OBJ_MIN_DETECTIONS}_filter${FILTER_INTERVAL}_merge${MERGE_OVERLAP_THRESH}-${MERGE_VISUAL_SIM_THRESH}-${MERGE_TEXT_SIM_THRESH}}"
MAP_PKL="${MAP_PKL:-$DATASET_ROOT/$DATASET_NAME/pcd_saves/full_pcd_none_${SAVE_SUFFIX}_post.pkl.gz}"
MAPPING="${MAPPING:-$DATASET_ROOT/$DATASET_NAME/filename_mapping.json}"
