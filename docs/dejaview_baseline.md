# DejaView ConceptGraphs baseline

Upstream base: `93277a02bd89171f8121e84203121cf7af9ebb5d`.

This adaptation selects upstream SAM segment-all + per-mask OpenCLIP ViT-H-14
(`laion2b_s32b_b79k`), class-agnostic spatial/visual object fusion. It does not
modify the upstream mapping algorithm or perform relation-graph retrieval.
`run_cg_detect.sh`, `run_cg_mapping.sh`, `run_cg_evaluator.sh` are the three
stages; `run_cg_all.sh` selects scenes/stages. `run_cg_visualize.sh` launches
the optional Viser viewer (install `viser` separately).

The retriever uses a single text template, `a photo of {query}`. Contributor
frames are actual merged `color_path` records, deduplicated and ordered by
numeric filename. The 504x504 config loads intrinsics from each scene.

## Dependencies

Follow the upstream README and use the separate ConceptGraphs environment.
Nested dependency checkouts are deliberately not vendored or gitlinks:

| Repository | Revision |
| --- | --- |
| IDEA-Research/Grounded-Segment-Anything | a4d76a2b55e348943cba4cd57d7553c354296223 |
| krrish94/chamferdist | ee753899b583bdc06bd2505dc162fd54ba0c6c5b |
| gradslam/gradslam (conceptfusion) | 59ca872e3d265ad09f63c4793d011fad67064452 |

Set `GSA_PATH` to the Grounded-Segment-Anything checkout. For CUDA extension
builds, use the matching CUDA toolkit (the original setup used CUDA 11.8),
and set `CUDA_HOME` to its installation before compiling dependencies.

## Known legacy limitations

The batch runner defaults to mask confidence 0.95 while individual scripts
use 0.2. Historical results also mix similarity thresholds 1.0 and 1.2 and
other fusion settings. Output names do not identify all parameters. Runners
contain machine paths. Empty maps and output paths without a directory need
handling. Do not assume current runner defaults reproduce historical JSON.

## Data and evaluation contract

Inputs are a DA3 ScanNet-style export: `color/<integer>.jpg`,
`depth/<integer>.png` (millimeters), camera-to-world `pose/<integer>.txt`,
`intrinsic/intrinsic_{color,depth}.txt`, and `filename_mapping.json` mapping
local integer frame IDs to original annotation filenames. The exporter is
`Depth-Anything-3/tools/run_metric_depth.py` at
`01accac4c0b73c8d6d35226d86a1c43fc32ea9aa`; this is a separate dependency.
Annotations use `stream_annotations.json`; query lists live in DejaView's
`eval/object_test_lists/`. Dataset names, chapter and prefix must agree.

The evaluator chooses one object, then its latest top-t frames. `hit@1` uses
only the primary frame; TP/recall use any of the top-t frames. In-chapter rates
use evaluated queries present in the chapter as denominator. These scripts
implement the original current-chapter protocol, not DejaView's newer
current+previous metrics. Archived runs use t=1 and threshold=-1 with hit
threshold bypass. Their ghost/precision fields are not calibrated object
presence detection measurements.

## Historical experiment snapshot

`dejaview-baseline-legacy-v1` preserves the imported experimental implementation
and JSON outputs. JSON bytes are unchanged. `eval_results/legacy_manifest.json`
records checksums and metadata. Historical configs are supporting evidence,
not proof of which invocation generated each artifact. Missing provenance is
explicitly unknown; archiving did not rerun inference. Absolute paths in these
historical artifacts document the original machine.

Model weights, maps, images, detections, compiled dependencies and full run
logs are excluded. Keep upstream LICENSE and citation information.

## Corrected workflow (after the legacy tag)

Use a fresh `RUN_ID` for each experiment. Runners resolve repository-relative
code and accept externally supplied absolute data paths. They refuse existing
map outputs and result files; the evaluator's explicit `--overwrite` option
is available for intentional JSON replacement. Current results go under
`eval_results/runs/` and are ignored. Existing legacy result JSON is unchanged.

Mapping records the exact command, source commit, tracked dirty state and
success/failure in a run manifest. Evaluation records its arguments, revision,
protocol-input SHA-256 hashes, and the map manifest if available. A missing
manifest means map provenance is unknown. These records do not certify the
contents of large geometry/checkpoint files: preserve their versions with
the experiment. Do not change a checkout while a run is active.

Set the paths below for your machine (absolute paths recommended):

```bash
export DATASET_ROOT=/path/to/Depth-Anything-3/metric_depth_results
export ANNOTATION_ROOT=/path/to/metacam/rebuttal
export OBJECT_LIST_ROOT=/path/to/DejaView/eval/object_test_lists
export RUN_ID=pantry1_numeric_v1
export GPU_ID=0
```

All stages share `scripts/dejaview_env.sh`: mask confidence 0.95, similarity
sum 1.2, stride 1, minimum detections 2, filter interval 10, and merge gates
0.7/0.8/0.8. Map names include these adjustable settings and RUN_ID. This is
a declared configuration, not a claim that it generated every legacy result.
The upstream algorithm is unchanged.

```bash
./run_cg_detect.sh pantry1
./run_cg_mapping.sh pantry1
CHAPTER_ID=1 ./run_cg_evaluator.sh pantry1
./run_cg_visualize.sh pantry1 remote
# Alternatively, run all stages for a subset:
ONLY="pantry1 pantry2" ./run_cg_all.sh
```

Choose either individual stages or the batch command for a new run. Detection
caches are shared across mapping experiments. Existing caches are rejected by
default; after checking coverage, stride and checkpoint provenance, use
`REUSE_DETECTIONS=1` to reuse them explicitly. A larger mapping stride can
reuse a denser cache; a sparser cache cannot serve missing mapping frames.
For mixed dataset families, the batch runner accepts
`STANDARD_ANNOTATION_ROOT` and `PANTRY_ANNOTATION_ROOT`. `DATASET_CONFIG`,
`PYTHON`, `MAP_PKL`, `MAPPING`, `OUTPUT_PATH`, and `CONTRIBUTOR_TOP_T` can be
set explicitly. For map creation, change `SAVE_SUFFIX` to control the output;
`MAP_PKL` is intended for evaluating/viewing existing maps.

Empty maps now produce misses/true negatives without an np.stack failure.
Invalid chapters fail explicitly. Viewer images can be relocated with
`--color_dir`; an empty map produces an explanatory error.

Validation: `python -m unittest discover -s tests -p test_dejaview_baseline.py -v`
checks empty-map initialization/evaluation, numeric contributor ordering,
output preservation, stage/batch argument routing and failed-run provenance.
Model construction is mocked in the empty-map test; no weights are downloaded.

Both the shell syntax and new/modified Python syntax were checked. Tests run
in the existing baseline environments. No full GPU mapping, accuracy rerun,
or browser rendering validation was performed. The headless backend change
is a fallback at import time, not a guarantee for every GUI configuration.
