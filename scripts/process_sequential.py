"""End-to-end sequential preprocessor: render then COLMAP, scene by scene.

For each scene in the manifest (or the user-specified subset), runs:
  1. `render_all.py --mode sequential --scenes <scene>`
  2. `run_colmap_all.py --mode sequential --scenes <scene>`

Each scene finishes both stages before the next starts. No cluster
needed — both stages run in-process on the local GPU. Per-frame
skip-if-exists inside each child script makes resumption cheap.

Usage:
    python process_sequential.py                       # all scenes
    python process_sequential.py --scenes alligator beetle
"""
import argparse
import subprocess
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
MANIFEST = DATASET_ROOT / "manifests" / "blender_moments.yaml"
RENDER_ALL = SCRIPT_DIR / "render_all.py"
RUN_COLMAP_ALL = SCRIPT_DIR / "run_colmap_all.py"


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
        print(f"\n{'='*70}\n  {scene}: render (sequential)\n{'='*70}", flush=True)
        run_in_env("blender_moments", str(RENDER_ALL), "--mode", "sequential", "--scenes", scene)
        print(f"\n{'='*70}\n  {scene}: COLMAP (sequential)\n{'='*70}", flush=True)
        run_in_env("bm_colmap", str(RUN_COLMAP_ALL),
                   "--mode", "sequential", "--scenes", scene)


if __name__ == "__main__":
    main()
