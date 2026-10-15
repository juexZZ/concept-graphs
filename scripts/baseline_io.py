"""Small input/output helpers for the DejaView baseline workflow."""
import hashlib
import json
from pathlib import Path
import subprocess


def save_result(path, result, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def provenance(repo, args, inputs):
    """Hash small protocol inputs; record the revision without claiming map origin."""
    repo = Path(repo)
    try:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"protocol": "dejaview-current-chapter-v1", "code_commit": revision,
            "tracked_worktree_dirty": dirty, "arguments": vars(args),
            "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
            "map_generating_commit": None}
