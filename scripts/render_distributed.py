import os
import sys
import bpy
import math
import argparse
import BlenderNeRFRenderer
from mathutils import Vector


def get_scene_meshes():
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for key, obj in bpy.context.scene.objects.items():
        if isinstance(obj.data, (bpy.types.Mesh)) and ('Plane' not in key) \
            and (obj.users_collection[0].name not in ['Collection 6', 'Collection 13', 'Collection 14']):
            yield key, obj

def scene_bbox(single_obj=None, ignore_matrix=False):
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = [math.inf] * 3
    bbox_max = [-math.inf] * 3
    bbox_min_key = [''] * 3
    bbox_max_key = [''] * 3
    found = False
    for key, obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        scales = obj.scale
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            for i in range(len(coord)):
                if (coord[i] * scales[i]) < bbox_min[i]:
                    bbox_min[i] = coord[i] * scales[i]
                    bbox_min_key[i] = key

                if (coord[i] * scales[i]) > bbox_max[i]:
                    bbox_max[i] = coord[i] * scales[i]
                    bbox_max_key[i] = key

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max), bbox_min_key, bbox_max_key


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True, help="Path to the .blend file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="absolute directory where train/, test/, and transforms_*.json "
                             "live (orchestrator passes <datasets_root>/<dataset>/processed/<scene>/)")
    parser.add_argument("--frame", type=int, required=True, help="animation frame to render")
    parser.add_argument("--fixed_cameras", action="store_true", help="fix the camera positions across frames")
    parser.add_argument("--debug", action="store_true", help="run without writing image files")
    parser.add_argument("--dry-render", action="store_true",
                        help="Run the full per-frame pipeline (scene.frame_set + COS operator) "
                             "but skip the Cycles render. The JSON writers print their would-be "
                             "frame_data (t value + file_path) instead of being silenced, and "
                             "scene.frame_current is logged after scene.frame_set. Lets us see "
                             "what each sbatch would have written without spending GPU time.")
    parser.add_argument("--train_count", type=int, default=50, help="number of train cameras per frame (cos_nb_frames)")
    parser.add_argument("--test_count", type=int, default=10, help="number of test cameras per frame (cos_nb_test_frames)")
    parser.add_argument("--cycles_samples", type=int, default=128, help="cycles render samples per pixel")
    parser.add_argument("--min_elevation", type=float, default=0.0,
                        help="minimum camera elevation in degrees (0 = horizon, +90 = top, -90 = bottom)")
    parser.add_argument("--max_elevation", type=float, default=90.0,
                        help="maximum camera elevation in degrees (0 = horizon, +90 = top, -90 = bottom)")
    parser.add_argument("--min_azimuth", type=float, default=0.0,
                        help="minimum camera azimuth in degrees, measured from +x toward +y")
    parser.add_argument("--max_azimuth", type=float, default=360.0,
                        help="maximum camera azimuth in degrees, measured from +x toward +y")
    parser.add_argument("--seed", type=int, default=42, help="camera-sampling seed")
    parser.add_argument("--elevation_sampling", choices=("band", "blendernerf"), default="band",
                        help="'band': area-uniform on the elevation band; 'blendernerf': "
                             "BlenderNeRF's original upper-hemisphere sampling")
    parser.add_argument("--pixel_filter", choices=("blend", "sharp"), default="blend",
                        help="'blend': keep the .blend's own filter width (the published "
                             "data); 'sharp': cap it at 0.01 px")
    parser.add_argument("--focal_mm", type=float, default=50.0,
                        help="BlenderNeRF Camera focal length in mm (sets scene.focal)")
    parser.add_argument("--resolution_x", type=int, default=800,
                        help="render width in pixels (overrides .blend default)")
    parser.add_argument("--resolution_y", type=int, default=800,
                        help="render height in pixels (overrides .blend default)")
    args = parser.parse_args()

    if "4d" not in args.object_path:
        print("This scene is not animateable. Use render.py instead")
        sys.exit()

    # Path setup. The COS operator computes its output dir as
    # os.path.join(scene.save_path, bpy.path.clean_name(scene.cos_dataset_name))
    # so we split the orchestrator-supplied output_dir into its parent
    # (save_path) and leaf (cos_dataset_name).
    out_dir = os.path.abspath(args.output_dir)
    out_parent = os.path.dirname(out_dir.rstrip(os.sep))
    out_leaf = os.path.basename(out_dir.rstrip(os.sep))

    # Skip the Blender startup if this frame's outputs are already on disk.
    # In --dry-render mode we never skip — we want every frame logged.
    if not args.dry_render:
        last_train = os.path.join(out_dir, "train",
                                  f"frame_{args.frame:04d}_view_{args.train_count:04d}.png")
        last_test = os.path.join(out_dir, "test",
                                 f"frame_{args.frame:04d}_view_{args.test_count:04d}.png")
        if os.path.exists(last_train) and os.path.exists(last_test):
            print(f"frame {args.frame}: outputs present at {out_dir}, skipping")
            sys.exit(0)

    # load blend file
    bpy.ops.wm.open_mainfile(filepath=args.object_path)

    # setup the scene
    BlenderNeRFRenderer.register()
    # BlenderNeRF's register() installs a @persistent depsgraph handler
    # (helper.set_init_props) that overrides scene.save_path to the .blend's
    # directory on the first depsgraph update. Remove it before we set
    # save_path, otherwise our writes get redirected silently.
    from BlenderNeRFRenderer.helper import set_init_props
    if set_init_props in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(set_init_props)

    # Per-frame jobs render PNGs only. Camera positions are a scene property
    # (deterministic from scene.seed + view index), so the transforms_train /
    # transforms_test JSON files are written once by a setup pass (render.py
    # --debug --animate) before any per-frame job runs. Disable this operator's
    # JSON writers so concurrent per-frame jobs never touch the JSON.
    from BlenderNeRFRenderer.blender_nerf_operator import BlenderNeRF_Operator
    if args.dry_render:
        # Diagnostic mode: instead of silencing the JSON writers, swap them
        # for logging stubs so we can see what each per-frame sbatch would
        # have written (t value, file_path). Used to pinpoint where the
        # lego_4d off-by-one bug crept in during the original live run.
        def _log_frame_data(self, directory, filename, new_frame_data, indent=4):
            n = len(new_frame_data["frames"])
            if n == 0:
                print(f"[dry] {filename}: 0 entries", flush=True)
                return
            first = new_frame_data["frames"][0]
            last = new_frame_data["frames"][-1]
            print(f"[dry] {filename}: {n} entries, "
                  f"t_first={first.get('t')}, t_last={last.get('t')}, "
                  f"file_first={first.get('file_path')}, "
                  f"file_last={last.get('file_path')}",
                  flush=True)
        BlenderNeRF_Operator.append_to_json_file = _log_frame_data
        BlenderNeRF_Operator.save_json = _log_frame_data
    else:
        BlenderNeRF_Operator.append_to_json_file = lambda *a, **kw: None
        BlenderNeRF_Operator.save_json = lambda *a, **kw: None
    context = bpy.context
    scene = context.scene
    scene.save_path = out_parent
    # Camera intrinsics: enforce the dataset's canonical rig values regardless
    # of what each .blend file was authored with. scene.focal flows to the
    # BlenderNeRF Camera's lens via helper.py; resolution_* set the render
    # output dimensions consumed by get_camera_intrinsics.
    scene.focal = args.focal_mm
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    # Render with a transparent background so the alpha channel marks
    # subject vs background. Without this, ALIKED/Roma find features on the
    # textured world background and COLMAP triangulation produces no points.
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.cos_nb_frames = args.train_count
    scene.cos_nb_test_frames = args.test_count
    scene.min_elevation = args.min_elevation
    scene.max_elevation = args.max_elevation
    scene.min_azimuth = args.min_azimuth
    scene.max_azimuth = args.max_azimuth
    scene.elevation_sampling = args.elevation_sampling
    scene.animate = True
    scene.distributed = True
    scene.debug = args.debug
    scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    # Bounce/filter/denoising floors match objaverse-xl's renderer. We never
    # lower the .blend's existing values below ours so any stricter
    # (higher-quality) authoring choice is preserved.
    scene.cycles.diffuse_bounces = max(scene.cycles.diffuse_bounces, 1)
    scene.cycles.glossy_bounces = max(scene.cycles.glossy_bounces, 1)
    scene.cycles.transparent_max_bounces = max(scene.cycles.transparent_max_bounces, 3)
    # Same pixel filter as render.py's sequential path, so both modes render alike.
    if args.pixel_filter == "sharp":
        scene.cycles.filter_width = min(scene.cycles.filter_width, 0.01)
    scene.cycles.transmission_bounces = max(scene.cycles.transmission_bounces, 3)
    scene.cycles.use_denoising = True
    if args.debug or args.dry_render:
        scene.render_frames = False
        scene.render_test_frames = False

    if args.fixed_cameras:
        scene.fixed_cameras = True
    scene.cos_dataset_name = out_leaf
    scene.seed = args.seed
    scene.init_output_path = out_dir

    bbox_min, bbox_max, bbox_min_key, bbox_max_key = scene_bbox()

    # render images
    scene.frame_set(args.frame)
    if args.dry_render:
        print(f"[dry] --frame {args.frame} -> scene.frame_current = {scene.frame_current}",
              flush=True)
    render = scene.render
    render.fps = 25
    scene.cycles.samples = args.cycles_samples
    if not args.debug and not args.dry_render:
        scene.render_test_frames = True
    bpy.ops.object.camera_on_sphere()

    # Defensive post-render check: confirm the operator produced the expected
    # last train + test PNGs. If frame_{args.frame:04d}_*.png isn't on disk,
    # something went wrong (off-by-one filename naming, render aborted, etc.).
    # Fail loud so the dependent COLMAP/LMDB/Kalman sbatches get
    # DependencyNeverSatisfied and the dispatcher cleans up the scene instead
    # of going silently into ~250 wrong-indexed downstream sbatches.
    exit_code = 0
    if not args.debug and not args.dry_render:
        expected_train = os.path.join(
            out_dir, "train",
            f"frame_{args.frame:04d}_view_{args.train_count:04d}.png",
        )
        expected_test = os.path.join(
            out_dir, "test",
            f"frame_{args.frame:04d}_view_{args.test_count:04d}.png",
        )
        if not os.path.exists(expected_train) or not os.path.exists(expected_test):
            print(
                f"FATAL: expected PNGs missing for frame {args.frame}:\n"
                f"  train: {expected_train} exists={os.path.exists(expected_train)}\n"
                f"  test:  {expected_test} exists={os.path.exists(expected_test)}\n"
                f"This may be the lego_4d-style off-by-one filename bug. "
                f"Aborting (exit 1).",
                flush=True,
            )
            exit_code = 1

    # Hard exit: bypass Blender's CUDA-context shutdown, which hangs for many
    # minutes per process in bpy 4.5. PNGs are already synced to disk by the
    # render operator before it returns, so the bypass is safe.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
