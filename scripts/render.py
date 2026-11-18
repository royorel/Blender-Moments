import os
import sys
import bpy
import math
import argparse
import BlenderNeRFRenderer
from mathutils import Vector
from subprocess import Popen


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
                             "are written (orchestrator computes this as "
                             "<datasets_root>/<dataset>/processed/<scene>/)")
    parser.add_argument("--animate", action="store_true", help="render the animated scene")
    parser.add_argument("--fixed_cameras", action="store_true", help="fix the camera positions during animation")
    parser.add_argument("--distributed", action="store_true", help="dispatch per-frame renders via slurm instead of rendering in-process")
    parser.add_argument("--debug", action="store_true", help="run without writing image files")
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
    parser.add_argument("--frame_start", type=int, default=1, help="first animation frame (inclusive)")
    parser.add_argument("--frame_end", type=int, default=100, help="last animation frame (inclusive)")
    args = parser.parse_args()

    if args.distributed:
        popen_stdout = open("popen_logs/stdout.log", 'w')
        popen_stderr = open("popen_logs/stderr.log", 'w')

    if args.animate and "4d" not in args.object_path:
        print("This scene is not animateable.")
        sys.exit()

    # load blend file
    bpy.ops.wm.open_mainfile(filepath=args.object_path)

    # Path setup. The COS operator computes its output dir as
    # os.path.join(scene.save_path, bpy.path.clean_name(scene.cos_dataset_name))
    # so we split the orchestrator-supplied output_dir into its parent
    # (save_path) and leaf (cos_dataset_name).
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_parent = os.path.dirname(out_dir.rstrip(os.sep))
    out_leaf = os.path.basename(out_dir.rstrip(os.sep))

    # setup the scene
    BlenderNeRFRenderer.register()
    # BlenderNeRF's register() installs a @persistent depsgraph handler
    # (helper.set_init_props) that overrides scene.save_path to the .blend's
    # directory on the first depsgraph update. Remove it before we set
    # save_path, otherwise our writes get redirected silently.
    from BlenderNeRFRenderer.helper import set_init_props
    if set_init_props in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(set_init_props)
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
    scene.cos_nb_frames = args.train_count if args.animate else 250
    scene.cos_nb_test_frames = args.test_count
    scene.min_elevation = args.min_elevation
    scene.max_elevation = args.max_elevation
    scene.min_azimuth = args.min_azimuth
    scene.max_azimuth = args.max_azimuth
    scene.elevation_sampling = args.elevation_sampling
    scene.animate = args.animate
    scene.distributed = args.distributed
    scene.debug = args.debug
    if args.debug:
        # --debug mode writes JSON only, no rendering. Skip the CUDA-context
        # initialization entirely; without an active GPU consumer to tear it
        # down at exit, bpy 4.5 hangs for minutes in static-destructor cleanup.
        scene.render_frames = False
        scene.render_test_frames = False
    else:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        # Bounce/filter/denoising floors match objaverse-xl's renderer. We
        # never lower the .blend's existing values below ours so any stricter
        # (higher-quality) authoring choice is preserved.
        scene.cycles.diffuse_bounces = max(scene.cycles.diffuse_bounces, 1)
        scene.cycles.glossy_bounces = max(scene.cycles.glossy_bounces, 1)
        scene.cycles.transparent_max_bounces = max(scene.cycles.transparent_max_bounces, 3)
        scene.cycles.transmission_bounces = max(scene.cycles.transmission_bounces, 3)
        # filter_width: smaller = sharper. The published data kept each
        # .blend's own width ('blend'); 'sharp' caps it.
        if args.pixel_filter == "sharp":
            scene.cycles.filter_width = min(scene.cycles.filter_width, 0.01)
        scene.cycles.use_denoising = True

    if args.fixed_cameras:
        scene.fixed_cameras = True
    scene.cos_dataset_name = out_leaf

    scene.seed = args.seed
    scene.init_output_path = out_dir

    bbox_min, bbox_max, bbox_min_key, bbox_max_key = scene_bbox()

    # render images
    if scene.animate:
        # Refuse to overwrite existing transforms_*.json. If the seed has
        # changed between this invocation and the one that produced the
        # existing JSONs, regenerating them would silently de-sync the
        # cameras from the already-rendered images. To force a fresh setup,
        # delete both JSONs explicitly. Applied in both setup (--debug) and
        # render modes so the two paths behave identically.
        train_json = os.path.join(out_dir, "transforms_train.json")
        test_json = os.path.join(out_dir, "transforms_test.json")
        if os.path.exists(train_json) and os.path.exists(test_json):
            print(f"transforms_train.json + transforms_test.json already "
                  f"present in {out_dir}; refusing to overwrite. Delete "
                  f"both to force a fresh setup + render.")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        render = scene.render
        render.fps = 25
        scene.cycles.samples = args.cycles_samples
        if not args.debug:
            scene.render_test_frames = True
        for i in range(args.frame_start, args.frame_end + 1):
            # Image-level skip only applies to actual rendering. Setup mode
            # (--debug) must process every frame so its JSON entries are
            # written, regardless of which PNGs already exist on disk.
            if not args.debug:
                last_train = os.path.join(out_dir, "train",
                                          f"frame_{i:04d}_view_{scene.cos_nb_frames:04d}.png")
                last_test = os.path.join(out_dir, "test",
                                         f"frame_{i:04d}_view_{scene.cos_nb_test_frames:04d}.png")
                if os.path.exists(last_train) and os.path.exists(last_test):
                    print(f"frame {i}: outputs present, skipping")
                    continue
            if args.distributed:
                slurm_script = 'render_frame.slurm'
                fixed_cam = '--fixed_cameras ' if args.fixed_cameras else ''
                debug = '--debug ' if args.debug else ''
                run_cmd = ['sbatch', '--parsable', slurm_script, args.object_path, i, fixed_cam, debug]
                Popen(run_cmd, stdout=popen_stdout, stderr=popen_stderr)
            else:
                scene.frame_set(i)
                bpy.ops.object.camera_on_sphere()
    else:
        bpy.ops.object.camera_on_sphere()

    # Hard exit: bypass Blender's shutdown, which hangs for many minutes per
    # process in bpy 4.5. All work (PNGs in render mode, JSON entries in
    # --debug setup mode) is synced to disk before we reach this point, so
    # the bypass is safe in both paths.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
