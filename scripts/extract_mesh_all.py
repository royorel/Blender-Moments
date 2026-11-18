"""Top-level orchestrator for ground-truth mesh extraction across Blender-Moments.

Reads the dataset manifest and dispatches per-scene mesh extraction via
extract_mesh.py. Each per-scene invocation opens the .blend once and
emits all N frames inside one Blender process, so extraction wall-time
for the full matrix (9 scenes x 10 frames) is on the order of 5-6 min
on a single CPU. No SLURM machinery, sequential is plenty fast.

Outputs land at
  `<dataset_root>/synthetic_gt/<scene>/frame_<NNNN>.ply`
  `<dataset_root>/synthetic_gt/<scene>/scene_normalization.json`

Usage:
    # Extract everything (default).
    python extract_mesh_all.py

    # Subset of scenes, dry-run.
    python extract_mesh_all.py --scenes alligator ficus_4d --dry-run

    # Force re-extraction even if outputs exist.
    python extract_mesh_all.py --force
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_ROOT_DEFAULT = SCRIPTS_DIR.parent
MANIFEST_DEFAULT = DATASET_ROOT_DEFAULT / "manifests" / "blender_moments.yaml"
EXTRACT_MESH_PY = SCRIPTS_DIR / "extract_mesh.py"

CONDA_ENV = "blender_moments"
DEFAULT_N_FRAMES = 10

# Scenes excluded from ground-truth generation. None currently.
EXCLUDED_SCENES = set()


def _blender_env():
    """Environment for the `conda run -n blender_moments` subprocess. `conda
    run` swaps PATH/conda vars but inherits the caller's LD_LIBRARY_PATH /
    PYTHONPATH. When this orchestrator runs from another conda env, those
    leak in and the linker loads the wrong libtbb / libstdc++ when bpy
    dlopens, breaking the import. Strip them so the blender_moments env
    resolves its own libraries, with no special launch flags."""
    return {
        k: v for k, v in os.environ.items()
        if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")
    }


def _load_yaml(path):
    return yaml.safe_load(path.read_text())


def _resolve_blend_path(dataset_root, scene_name, source_blend):
    """Mirrors render_all.py's ensure_blend_symlink naming. Prefers
    the symlink under raw/<scene>/ if present, falls back to the
    manifest's source_blend (relative to raw/)."""
    basename = scene_name if scene_name.endswith("_4d") else f"{scene_name}_4d"
    symlink = dataset_root / "raw" / scene_name / f"{basename}.blend"
    if symlink.exists():
        return symlink
    src = Path(source_blend)
    if not src.is_absolute():
        src = dataset_root / "raw" / src
    return src


def _scene_done(output_dir, frame_indices):
    """Cache check, the PLYs at the selected frame indices plus
    scene_normalization.json must already be on disk."""
    if not (output_dir / "scene_normalization.json").exists():
        return False
    for frame_idx in frame_indices:
        if not (output_dir / f"frame_{frame_idx:04d}.ply").exists():
            return False
    return True


def _evenly_spaced_frames(start, end, n):
    """Mirror of extract_mesh.py's recipe, the cache check needs the
    same frame indices the worker would produce."""
    length = end - start + 1
    step = length / n
    offset = int(step / 2)
    return [start - 1 + offset + int(k * step) for k in range(n)]


def gt_mesh_frames(manifest, scene):
    """Ground-truth frames of a scene. `gt_mesh_frames` comes from the scene
    entry, else the manifest defaults, else DEFAULT_N_FRAMES; an integer N
    selects N evenly spaced frames of the scene's frame_range, a list gives
    the frame numbers."""
    spec = scene.get("gt_mesh_frames",
                     (manifest.get("defaults") or {}).get("gt_mesh_frames", DEFAULT_N_FRAMES))
    if isinstance(spec, int):
        return _evenly_spaced_frames(*scene["frame_range"], spec)
    return [int(f) for f in spec]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Restrict to these scene names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each scene's frames as '<scene>: <f1> <f2> ...' and exit")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if outputs are on disk")
    args = parser.parse_args()

    manifest = _load_yaml(args.manifest)
    all_scenes = manifest["scenes"]

    if args.scenes is not None:
        wanted = set(args.scenes)
        all_scenes = [s for s in all_scenes if s["name"] in wanted]

    scenes = [s for s in all_scenes if s["name"] not in EXCLUDED_SCENES]
    excluded_present = [s["name"] for s in all_scenes if s["name"] in EXCLUDED_SCENES]
    if excluded_present:
        print(f"[extract_mesh_all] skipping excluded scenes: {excluded_present}",
              flush=True)

    if args.dry_run:
        for scene in scenes:
            print(f"{scene['name']}: {' '.join(map(str, gt_mesh_frames(manifest, scene)))}")
        return

    print(f"[extract_mesh_all] {len(scenes)} scene(s) to extract", flush=True)

    for scene in scenes:
        name = scene["name"]
        frame_start, frame_end = scene["frame_range"]
        output_dir = args.dataset_root / "synthetic_gt" / name

        frame_indices = gt_mesh_frames(manifest, scene)
        if not args.force and _scene_done(output_dir, frame_indices):
            print(f"[extract_mesh_all] {name}: cache hit, skipping", flush=True)
            continue

        blend_path = _resolve_blend_path(args.dataset_root, name, scene["source_blend"])
        if not blend_path.exists():
            raise FileNotFoundError(f"no .blend for {name}, looked at {blend_path}")

        cmd = [
            "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
            "python", "-u", str(EXTRACT_MESH_PY),
            "--object_path", str(blend_path),
            "--output_dir", str(output_dir),
            "--frame_start", str(frame_start),
            "--frame_end", str(frame_end),
            "--frames", *map(str, frame_indices),
        ]
        print(f"[extract_mesh_all] {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, env=_blender_env())


if __name__ == "__main__":
    main()
