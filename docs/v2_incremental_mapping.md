# v2 incremental continuation (2026-10-26)

The legacy `scripts/v2_campaign.py` reconstructs cumulative T1/T2/T3 prefixes
independently. Do not use its `--full` launcher to resume the active final scene.

The user's requested final-scene continuation is running in an isolated source
snapshot, so the already-running T2 mapper and sealed results are preserved:

`/root/dejaview_experiments/baselines/conceptgraphs/continuation-r01/`

The workflow reuses 978 cached T2 frames and processes only 468 new run3 frames,
with 8 old boundary images reinferred solely for aligning the new DA3 geometry.
Old geometry is not replaced; old semantic inference and mapping are not repeated.
Mapping resumes from the native T2 raw endpoint at global index978, preserving
the native association parameters and periodic filter/merge indices.

Entry point: `launch.sh`; durable session: `tmux -L conceptgraphs`,
`cg-continuation-r01`. Never SIGCONT the held legacy scheduler while the
replacement controller is active. The controller retires only that scheduler
after T2's native completion receipt has been sealed.

Output root:
`/root/dejaview_experiments/runs/conceptgraphs-continuation-v2-20261026-r01`

This is a separately recorded continuation, not bitwise equivalent to an
uninterrupted stream: old T2 raw output was already terminally denoised, no RNG
checkpoint was available, serialized boxes are rebuilt by the native loader,
and old geometry export did not retain confidence at the alignment boundary.
Existing eight-scene results remain an independent-prefix partial benchmark;
any all-scene aggregate with the continuation is explicitly mixed-protocol.

Frontend remains native SAM ViT-H / OpenCLIP ViT-H-14 (1024D). A SAM2/SigLIP
unified-frontend comparison must be a separate map/run and text encoder, not a
mid-map replacement of the 1024D embeddings.

Detailed results and validation notes:
`/root/DejaView/docs/conceptgraphs_incremental_continuation.md`.
