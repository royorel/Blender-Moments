"""Top-level orchestrator for rendering all Blender-Moments scenes.

Reads the dataset manifest, then dispatches per-scene rendering in one of
two modes:

  - sequential: run `render.py --animate` once per scene (single GPU,
    one Blender process renders all frames of that scene in order).
  - distributed: submit per-frame sbatch jobs via `render_distributed.py`,
    parallelising across the cluster.

Scene source semantics (from the manifest's `source` field):
  - pre_rendered: skipped (renders already exist on disk).
  - local_blender: rendered as described above.
  - objaverse_xl: not yet supported by this orchestrator.

Outputs land in `<dataset_root>/processed/<scene>/` (train/, test/, and the
two transforms_*.json files written directly into that directory).

Usage:
    # Render everything sequentially on this machine.
    python render_all.py --mode sequential

    # Submit per-frame sbatch jobs across the cluster.
    python render_all.py --mode distributed

    # Render just two scenes, dry-run mode (prints the planned commands).
    python render_all.py --mode distributed --scenes beetle toyota --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_ROOT_DEFAULT = SCRIPTS_DIR.parent
MANIFEST_DEFAULT = DATASET_ROOT_DEFAULT / "manifests" / "blender_moments.yaml"
SLURM_CONFIG_DEFAULT = DATASET_ROOT_DEFAULT / "slurm_config.yaml"
SLURM_TEMPLATE_DEFAULT = SCRIPTS_DIR / "render.slurm.template"
RENDER_SEQUENTIAL = SCRIPTS_DIR / "render.py"
RENDER_DISTRIBUTED = SCRIPTS_DIR / "render_distributed.py"


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


def load_slurm_config(slurm_config_path, section="render"):
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


def resolve_rig(manifest, rig_name):
    rig_defs = (manifest.get("defaults") or {}).get("rig_definitions") or {}
    if rig_name not in rig_defs:
        raise KeyError(
            f"rig {rig_name!r} not in manifest.defaults.rig_definitions; "
            f"available: {sorted(rig_defs)}"
        )
    return rig_defs[rig_name]


def resolve_blend_path(dataset_root, source_blend):
    p = Path(source_blend)
    if not p.is_absolute():
        p = Path(dataset_root) / "raw" / p
    return p


def ensure_blend_symlink(dataset_root, scene, blend_src):
    """Symlink the source .blend into raw/<scene>/<basename>.blend.

    `basename` is `<scene>_4d` for scenes that don't already end in
    `_4d`, or just `<scene>` when they do (avoids the `ficus_4d_4d`
    style double suffix). The `4d` substring still appears in the
    object_path either way, which is the gate render.py and
    render_distributed.py use for the animate path.
    """
    raw_scene_dir = Path(dataset_root) / "raw" / scene
    raw_scene_dir.mkdir(parents=True, exist_ok=True)
    basename = scene if scene.endswith("_4d") else f"{scene}_4d"
    target = raw_scene_dir / f"{basename}.blend"
    if not target.exists() and not target.is_symlink():
        target.symlink_to(blend_src.resolve())
    return target


def rig_cli_args(rig):
    return [
        "--train_count", str(rig["train_count"]),
        "--test_count", str(rig["test_count"]),
        "--cycles_samples", str(rig["cycles_samples"]),
        "--focal_mm", str(rig["focal_mm"]),
        "--resolution_x", str(rig["resolution_x"]),
        "--resolution_y", str(rig["resolution_y"]),
        "--min_elevation", str(rig["min_elevation"]),
        "--max_elevation", str(rig["max_elevation"]),
        "--min_azimuth", str(rig.get("min_azimuth", 0.0)),
        "--max_azimuth", str(rig.get("max_azimuth", 360.0)),
        "--seed", str(rig.get("seed", 42)),
        "--elevation_sampling", rig.get("elevation_sampling", "band"),
        "--pixel_filter", rig.get("pixel_filter", "blend"),
    ]


def _format_module_loads(modules):
    if not modules:
        return ""
    return "\n".join(f"module load {m}" for m in modules)


def drop_empty_settings(script_text):
    """Remove `#SBATCH --key=` lines and a `source` line whose value is empty,
    so keys left blank in slurm_config.yaml fall back to the cluster's own
    defaults instead of producing invalid directives."""
    script_text = re.sub(r"^#SBATCH --[\w-]+=\s*$\n", "", script_text, flags=re.M)
    return re.sub(r"^source\s*$\n", "", script_text, flags=re.M)


def render_sbatch_script(template_path, *, job_name, output_log, command,
                         slurm_config):
    """Render the slurm script for one sbatch submission."""
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _output_dir_for(dataset_root, scene):
    """Canonical per-scene render output path.

    `<dataset_root>/processed/<scene>/` — train/, test/, and transforms_*.json
    are written directly inside (no `<scene>_4d_renders_fixed_cameras` subdir).
    """
    return Path(dataset_root) / "processed" / scene


def dispatch_sequential(dataset_root, scene, blend_link, rig, frame_range,
                        fixed_cameras, dry_run):
    """Run render.py once per scene; it loops over frames itself."""
    output_dir = _output_dir_for(dataset_root, scene)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RENDER_SEQUENTIAL),
        "--object_path", str(blend_link),
        "--output_dir", str(output_dir),
        "--animate",
        *(["--fixed_cameras"] if fixed_cameras else []),
        *rig_cli_args(rig),
        "--frame_start", str(frame_range[0]),
        "--frame_end", str(frame_range[1]),
    ]
    print(f"[{scene}] sequential: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def _frame_outputs_exist(out_dir, frame, train_count, test_count):
    """True iff frame's last expected train + test PNGs exist and decode
    cleanly via PIL's header-only `verify()`.

    Render scripts write views in deterministic order, so a healthy last view
    implies the earlier views are also present. Checking just the last per
    split keeps this O(1) per frame; PIL.verify catches truncated or corrupt
    files without doing a full decode."""
    last_train = out_dir / "train" / f"frame_{frame:04d}_view_{train_count:04d}.png"
    last_test = out_dir / "test" / f"frame_{frame:04d}_view_{test_count:04d}.png"
    for p in (last_train, last_test):
        if not p.exists():
            return False
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            return False
    return True


def run_setup_local(scene, blend_link, rig, frame_range, output_dir,
                    fixed_cameras, conda_env, dry_run):
    """Run the JSON-writing setup pass inline on the dispatcher's host.

    `render.py --debug --animate` computes camera positions and writes the
    full transforms_train.json + transforms_test.json without touching the
    GPU. Doing this locally (rather than via sbatch) eliminates the queue
    wait, the dependency wiring, and the per-job Blender shutdown overhead.

    Routed through `conda run -n {conda_env}` so the bpy version matches what
    the sbatch workers use, regardless of which env launched the orchestrator.

    Skips Blender startup entirely if both transforms_*.json are already on
    disk. Camera positions are deterministic from scene.seed=42, so the JSONs
    are reproducible; if they exist, regenerating them is a ~30 s no-op.
    Delete them by hand to force a regeneration (e.g., after a rig change).
    """
    train_json = output_dir / "transforms_train.json"
    test_json = output_dir / "transforms_test.json"
    if train_json.exists() and test_json.exists():
        print(f"  [{scene}] setup (local): skipped ({train_json.name} + "
              f"{test_json.name} already present)")
        return

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "-u", str(RENDER_SEQUENTIAL),
        "--object_path", str(blend_link),
        "--output_dir", str(output_dir),
        "--animate", "--debug",
        *(["--fixed_cameras"] if fixed_cameras else []),
        *rig_cli_args(rig),
        "--frame_start", str(frame_range[0]),
        "--frame_end", str(frame_range[1]),
    ]
    print(f"  [{scene}] setup (local): {' '.join(cmd)}")
    if not dry_run:
        # bpy bundles its own SYCL Embree/TBB in bpy/lib (resolved via RUNPATH),
        # but LD_LIBRARY_PATH beats RUNPATH. If the launching shell has another
        # conda env active (e.g. the eval dispatcher's), that env's libembree4 /
        # libtbb shadow bpy's bundled ones and the import dies on a missing SYCL
        # symbol. Drop conda-env lib dirs so bpy uses its own bundled libs.
        clean = dict(os.environ)
        clean["LD_LIBRARY_PATH"] = ":".join(
            p for p in clean.get("LD_LIBRARY_PATH", "").split(":")
            if "/envs/" not in p
        )
        subprocess.run(cmd, check=True, env=clean)


def dispatch_distributed(dataset_root, scene, blend_link, rig, frame_range,
                         slurm_config, sbatch_log_dir, fixed_cameras, dry_run,
                         exclude_nodes=None):
    """Run setup locally (writes JSONs) + submit one per-frame sbatch per frame.

    All missing frames submit in parallel with no inter-job slurm dependency.
    On GPU architectures without precompiled Cycles kernels each worker
    compiles them first (see the render constraint in slurm_config.yaml).

    Per-frame jobs render PNGs only — they never touch the JSONs (those were
    written by run_setup_local above).
    """
    output_dir = _output_dir_for(dataset_root, scene)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    run_setup_local(scene, blend_link, rig, frame_range, output_dir,
                    fixed_cameras,
                    slurm_config["conda_env"], dry_run)

    f_start, f_end = frame_range
    rig_flags = " ".join(rig_cli_args(rig))
    fixed_flag = "--fixed_cameras " if fixed_cameras else ""
    extra_sbatch_args = [f"--exclude={exclude_nodes}"] if exclude_nodes else None
    jobids = []
    skipped = 0
    for frame in range(f_start, f_end + 1):
        if _frame_outputs_exist(output_dir, frame, rig["train_count"], rig["test_count"]):
            skipped += 1
            continue
        command = (
            f"python {RENDER_DISTRIBUTED} "
            f"--object_path {blend_link} --output_dir {output_dir} "
            f"--frame {frame} {fixed_flag}"
            f"{rig_flags}"
        )
        job_name = f"render_blender_moments_{scene}_f{frame:04d}"
        output_log = str(sbatch_log_dir / f"{job_name}_%j.log")
        script = render_sbatch_script(
            SLURM_TEMPLATE_DEFAULT,
            job_name=job_name, output_log=output_log,
            command=command, slurm_config=slurm_config,
        )
        jobid = submit_sbatch(script, dry_run=dry_run,
                              extra_sbatch_args=extra_sbatch_args)
        if jobid is not None:
            jobids.append(jobid)
            print(f"  [{scene}] frame {frame}: {jobid}")
    print(f"[{scene}] {len(jobids)} jobs submitted "
          f"({f_start}..{f_end}); {skipped} frames skipped (already rendered)")
    return jobids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sequential", "distributed"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT,
                        help="root of the Blender-Moments dataset")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT,
                        help="path to the dataset manifest YAML")
    parser.add_argument("--slurm-config", type=Path, default=SLURM_CONFIG_DEFAULT,
                        help="slurm config YAML (only used in distributed mode)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="subset of scene names to render; defaults to all")
    parser.add_argument("--fixed-cameras", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="fix camera positions across frames (project convention; "
                             "pass --no-fixed-cameras to resample per frame)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands/sbatch scripts without executing")
    parser.add_argument("--jobids-out", type=Path, default=None,
                        help="if set, write submitted render job IDs as JSON to this "
                             "path (consumed by upstream pipelines that need to "
                             "depend their build job on this scene's renders)")
    parser.add_argument("--exclude", default=None,
                        help="comma-separated SLURM nodes to exclude from the per-frame "
                             "render sbatches (sbatch --exclude), e.g. known-bad GPU nodes")
    args = parser.parse_args()

    manifest = _load_yaml(args.manifest)
    if not manifest:
        raise SystemExit(f"manifest not found or empty: {args.manifest}")

    all_scenes = [e["name"] for e in manifest.get("scenes", [])]
    requested = args.scenes if args.scenes is not None else all_scenes
    unknown = [s for s in requested if s not in all_scenes]
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}; available: {all_scenes}")

    slurm_config = None
    sbatch_log_dir = None
    if args.mode == "distributed":
        slurm_config = load_slurm_config(args.slurm_config)
        required = ("partition", "account", "cpus_per_task", "mem", "gres",
                    "constraint", "time", "conda_init", "conda_env", "workdir")
        missing = [k for k in required if k not in slurm_config]
        if missing:
            raise SystemExit(f"slurm_config missing keys: {missing}")
        # Resolve the slurm-log directory from the manifest defaults. Relative
        # paths are anchored at the dataset root so the bundle stays
        # self-contained; absolute paths are honored as-is.
        defaults = manifest.get("defaults") or {}
        rel = defaults.get("sbatch_log_dir", "logs/sbatch")
        p = Path(rel)
        sbatch_log_dir = p.resolve() if p.is_absolute() else (args.dataset_root / p).resolve()
        sbatch_log_dir.mkdir(parents=True, exist_ok=True)

    all_render_jobids = []
    for scene in requested:
        entry = get_scene_entry(manifest, scene)
        source = entry["source"]
        if source == "pre_rendered":
            print(f"[{scene}] pre_rendered, skipping render")
            continue
        if source == "objaverse_xl":
            print(f"[{scene}] objaverse_xl is not yet supported by render_all.py")
            continue
        if source != "local_blender":
            raise SystemExit(f"[{scene}] unknown source {source!r}")

        rig = resolve_rig(manifest, entry["rig"])
        blend_src = resolve_blend_path(args.dataset_root, entry["source_blend"])
        blend_link = ensure_blend_symlink(args.dataset_root, scene, blend_src)
        frame_range = entry["frame_range"]

        if args.mode == "sequential":
            dispatch_sequential(args.dataset_root, scene, blend_link, rig,
                                frame_range, fixed_cameras=args.fixed_cameras,
                                dry_run=args.dry_run)
        else:
            jobids = dispatch_distributed(
                args.dataset_root, scene, blend_link, rig, frame_range,
                slurm_config,
                sbatch_log_dir=sbatch_log_dir,
                fixed_cameras=args.fixed_cameras,
                dry_run=args.dry_run,
                exclude_nodes=args.exclude,
            )
            all_render_jobids.extend(jobids)

    if args.jobids_out is not None and not args.dry_run:
        args.jobids_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.jobids_out, "w") as f:
            json.dump({"jobids": all_render_jobids}, f, indent=2)
        print(f"wrote {len(all_render_jobids)} jobid(s) to {args.jobids_out}")


if __name__ == "__main__":
    main()
