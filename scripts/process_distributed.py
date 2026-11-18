"""End-to-end distributed preprocessor: render then COLMAP, scene by scene.

For each scene in the manifest (or the user-specified subset):
  1. `render_all.py --mode distributed --scenes <scene>` submits the
     per-frame render sbatches.
  2. Wait until those jobs leave the queue.
  3. `run_colmap_all.py --mode distributed --scenes <scene>` submits the
     per-frame COLMAP sbatches.
  4. Wait until those jobs leave the queue before starting the next scene.

The poll uses squeue and filters job names by the `render_` / `colmap_`
prefixes our orchestrators use, so manually-submitted sbatches with
different names (interactive sessions, other workloads) don't gate
progress.

Usage:
    python process_distributed.py                       # all scenes
    python process_distributed.py --scenes alligator beetle
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
MANIFEST = DATASET_ROOT / "manifests" / "blender_moments.yaml"
RENDER_ALL = SCRIPT_DIR / "render_all.py"
RUN_COLMAP_ALL = SCRIPT_DIR / "run_colmap_all.py"

POLL_INTERVAL_SEC = 60
JOB_NAME_PREFIXES = ("render_", "colmap_")


def load_scenes(manifest_path):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    return [e["name"] for e in manifest.get("scenes", [])]


def run_in_env(env, *args):
    """Invoke `python <args>` inside the given conda env, regardless of the
    env the orchestrator itself was launched from."""
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env, "python", *args],
        check=True,
    )


def wait_for_pipeline_jobs(label):
    """Block until no render_*/colmap_* sbatch jobs are left in the user's queue."""
    user = os.environ.get("USER") or subprocess.check_output(["whoami"], text=True).strip()
    while True:
        result = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%j"],
            capture_output=True, text=True, check=True,
        )
        active = [
            n for n in result.stdout.split()
            if n and n.startswith(JOB_NAME_PREFIXES)
        ]
        if not active:
            return
        print(f"  [{label}] {len(active)} pipeline jobs still in queue; "
              f"polling again in {POLL_INTERVAL_SEC}s...", flush=True)
        time.sleep(POLL_INTERVAL_SEC)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="subset of scene names to process; defaults to all")
    args = parser.parse_args()

    all_scenes = load_scenes(MANIFEST)
    requested = args.scenes or all_scenes
    unknown = [s for s in requested if s not in all_scenes]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; available: {all_scenes}")

    for scene in requested:
        print(f"\n{'='*70}\n  {scene}: render (distributed)\n{'='*70}", flush=True)
        run_in_env("blender_moments", str(RENDER_ALL),
                   "--mode", "distributed", "--scenes", scene)
        wait_for_pipeline_jobs(f"{scene}: render")
        print(f"\n{'='*70}\n  {scene}: COLMAP (distributed)\n{'='*70}", flush=True)
        run_in_env("bm_colmap", str(RUN_COLMAP_ALL),
                   "--mode", "distributed", "--scenes", scene)
        wait_for_pipeline_jobs(f"{scene}: COLMAP")


if __name__ == "__main__":
    main()
