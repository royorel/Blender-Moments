"""Top-level orchestrator for per-frame COLMAP triangulation across Blender-Moments.

Reads the dataset manifest, locates each scene's rendered output, and
dispatches per-frame COLMAP in one of two modes:

  - sequential: run `run_colmap.py` once per scene (single process loops
    over the scene's frames in order).
  - distributed: submit one sbatch per frame via
    `run_colmap_distributed.py`, parallelising across the cluster.

Preflight: each requested scene must have a non-empty
`transforms_train.json` + `train/` next to it. Missing renders fail
fast before any sbatch goes out.

Outputs land at
  `<dataset_root>/processed/<scene>/sfm_init/frame_<N>/colmap_points3D.ply`

Usage:
    # Default: sequential on this machine.
    python run_colmap_all.py

    # Cluster fan-out, one sbatch per (scene, frame).
    python run_colmap_all.py --mode distributed

    # Subset of scenes, dry-run.
    python run_colmap_all.py --mode distributed --scenes alligator beetle --dry-run
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from render_all import drop_empty_settings


SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_ROOT_DEFAULT = SCRIPTS_DIR.parent
MANIFEST_DEFAULT = DATASET_ROOT_DEFAULT / "manifests" / "blender_moments.yaml"
SLURM_CONFIG_DEFAULT = DATASET_ROOT_DEFAULT / "slurm_config.yaml"
SLURM_TEMPLATE_DEFAULT = SCRIPTS_DIR / "run_colmap.slurm.template"
RUN_COLMAP_SEQUENTIAL = SCRIPTS_DIR / "run_colmap.py"
RUN_COLMAP_DISTRIBUTED = SCRIPTS_DIR / "run_colmap_distributed.py"


def _load_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_slurm_config(slurm_config_path, section="colmap"):
    """Load `slurm_config.yaml` (+ optional `slurm_config.local.yaml` override)
    and return the flat dict for the given sbatch type (`render` or `colmap`).

    The yaml has `common:` plus per-section sub-keys. The returned dict is
    `common` overlaid with the requested section; downstream consumers see
    a flat dict (`partition`, `conda_env`, etc.).
    """
    base = _load_yaml(slurm_config_path)
    local = _load_yaml(Path(slurm_config_path).with_name("slurm_config.local.yaml"))
    merged = _deep_merge(base, local)
    common = merged.get("common", {})
    specific = merged.get(section, {})
    return {**common, **specific}


def get_scene_entry(manifest, scene):
    for entry in manifest.get("scenes", []):
        if entry["name"] == scene:
            return entry
    available = [e["name"] for e in manifest.get("scenes", [])]
    raise KeyError(f"scene {scene!r} not in manifest; available: {available}")


def _format_module_loads(modules):
    if not modules:
        return ""
    return "\n".join(f"module load {m}" for m in modules)


def render_sbatch_script(template_path, *, job_name, output_log, command,
                         slurm_config):
    with open(template_path) as f:
        template = f.read()
    return drop_empty_settings(template.format(
        job_name=job_name,
        output_log=output_log,
        command=command,
        partition=slurm_config["partition"],
        account=slurm_config["account"],
        cpus_per_task=slurm_config["cpus_per_task"],
        mem=slurm_config["mem"],
        gres=slurm_config["gres"],
        constraint=slurm_config["constraint"],
        time=slurm_config["time"],
        module_loads=_format_module_loads(slurm_config.get("modules", [])),
        conda_init=slurm_config["conda_init"],
        conda_env=slurm_config["conda_env"],
        workdir=slurm_config["workdir"] or SCRIPTS_DIR,
    ))


def submit_sbatch(script_text, dry_run, extra_sbatch_args=None):
    if dry_run:
        print(script_text)
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".slurm", delete=False) as f:
        f.write(script_text)
        script_path = f.name
    cmd = ["sbatch", "--parsable"]
    if extra_sbatch_args:
        cmd.extend(extra_sbatch_args)
    cmd.append(script_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"sbatch failed (exit {result.returncode}) for {script_path}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout.strip()


def _source_path_for(dataset_root, scene):
    """Per-scene render output path (matches render_all.py's _output_dir_for)."""
    return Path(dataset_root) / "processed" / scene


def _check_renders_present(source_path):
    """Preflight: raise if the scene's renders aren't on disk yet.

    Looks for transforms_train.json + a non-empty train/ directory.
    Mirrors how render_all.py's run_setup_local writes the JSON first and
    per-frame jobs write PNGs after.
    """
    transforms = source_path / "transforms_train.json"
    train_dir = source_path / "train"
    if not transforms.exists():
        raise SystemExit(f"missing {transforms}; render this scene first")
    if not train_dir.is_dir() or not any(train_dir.iterdir()):
        raise SystemExit(f"{train_dir} is missing or empty; render this scene first")


def dispatch_sequential(source_path, frame_range, dry_run):
    cmd = [
        sys.executable, str(RUN_COLMAP_SEQUENTIAL),
        "--source_path", str(source_path),
        "--frame_start", str(frame_range[0]),
        "--frame_end", str(frame_range[1]),
    ]
    print(f"[{source_path.name}] sequential: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def dispatch_distributed(source_path, frame_range, slurm_config,
                         sbatch_log_dir, dry_run):
    """Submit one sbatch per (scene, frame) via run_colmap_distributed.py.

    Frames whose PLY already exists on disk are skipped at dispatch time
    so the cluster never sees no-op sbatches. The in-sbatch
    `process_frame` retains the same skip check as a safety net for
    races (e.g., concurrent fan-outs).
    """
    from colmap_pipeline import ply_nonempty

    f_start, f_end = frame_range
    jobids = []
    skipped = 0
    for frame in range(f_start, f_end + 1):
        ply_path = source_path / "sfm_init" / f"frame_{frame}" / "colmap_points3D.ply"
        if ply_nonempty(ply_path):
            skipped += 1
            continue
        command = (
            f"python {RUN_COLMAP_DISTRIBUTED} "
            f"--source_path {source_path} --frame {frame}"
        )
        job_name = f"colmap_blender_moments_{source_path.name}_f{frame:04d}"
        output_log = str(sbatch_log_dir / f"{job_name}_%j.log")
        script = render_sbatch_script(
            SLURM_TEMPLATE_DEFAULT,
            job_name=job_name, output_log=output_log,
            command=command, slurm_config=slurm_config,
        )
        jobid = submit_sbatch(script, dry_run=dry_run)
        if jobid is not None:
            jobids.append(jobid)
            print(f"  [{source_path.name}] frame {frame}: {jobid}")
    print(f"[{source_path.name}] {len(jobids)} jobs submitted "
          f"({f_start}..{f_end}); {skipped} frames skipped (PLY exists)")
    return jobids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sequential", "distributed"), default="sequential",
                        help="default: sequential (most users won't have cluster access)")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT,
                        help="root of the Blender-Moments dataset")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT,
                        help="path to the dataset manifest YAML")
    parser.add_argument("--slurm-config", type=Path, default=SLURM_CONFIG_DEFAULT,
                        help="slurm config YAML (only used in distributed mode)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="subset of scene names to process; defaults to all")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands/sbatch scripts without executing")
    args = parser.parse_args()

    manifest = _load_yaml(args.manifest)
    if not manifest:
        raise SystemExit(f"manifest not found or empty: {args.manifest}")

    all_scenes = [e["name"] for e in manifest.get("scenes", [])]
    requested = args.scenes if args.scenes is not None else all_scenes
    unknown = [s for s in requested if s not in all_scenes]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; available: {all_scenes}")

    # Preflight every requested scene before any dispatch so missing renders
    # fail fast (we never partially-submit a fan-out).
    for scene in requested:
        _check_renders_present(_source_path_for(args.dataset_root, scene))

    slurm_config = None
    sbatch_log_dir = None
    if args.mode == "distributed":
        slurm_config = load_slurm_config(args.slurm_config)
        required = ("partition", "account", "cpus_per_task", "mem", "gres",
                    "constraint", "time", "conda_init", "conda_env", "workdir")
        missing = [k for k in required if k not in slurm_config]
        if missing:
            raise SystemExit(f"slurm_config missing keys: {missing}")
        defaults = manifest.get("defaults") or {}
        rel = defaults.get("sbatch_log_dir", "logs/sbatch")
        p = Path(rel)
        sbatch_log_dir = p.resolve() if p.is_absolute() else (args.dataset_root / p).resolve()
        sbatch_log_dir.mkdir(parents=True, exist_ok=True)

    for scene in requested:
        entry = get_scene_entry(manifest, scene)
        source_path = _source_path_for(args.dataset_root, scene)
        frame_range = entry["frame_range"]

        if args.mode == "sequential":
            dispatch_sequential(source_path, frame_range, dry_run=args.dry_run)
        else:
            dispatch_distributed(source_path, frame_range, slurm_config,
                                 sbatch_log_dir=sbatch_log_dir,
                                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
