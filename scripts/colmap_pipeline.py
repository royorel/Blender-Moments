"""Per-frame COLMAP triangulation pipeline.

The hloc fork referenced by this module (royorel/Hierarchical-Localization-Slim)
patches the aliked+roma matcher conf. `discard_match_th` is the per-pair
mean-certainty gate; Roma's 0.4 default was calibrated on high-
distinctiveness outdoor training data and rejects ~93% of pairs on
synthetic low-texture scenes (bicycle, shark). Setting it to 0 lifts
those scenes from ~0-170 points/frame to 1800+, while high-texture
scenes stay within 5% (reprojection error delta under 0.6 px, well below
COLMAP's 2 px filter).
"""

import collections
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from plyfile import PlyData

from hloc import extract_features, match_features, triangulation, pairs_from_retrieval


match_features.confs["aliked+roma"]["model"]["discard_match_th"] = 0.0


# ---------------------------------------------------------------------------
# COLMAP text I/O helpers
# ---------------------------------------------------------------------------

_Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
_Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def _write_cameras_text(cameras, path):
    header = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as f:
        f.write(header)
        for _, cam in cameras.items():
            line = " ".join(str(x) for x in [cam.id, cam.model, cam.width, cam.height, *cam.params])
            f.write(line + "\n")


def _write_images_text(images, path):
    header = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "# Number of images: {}, mean observations per image: 0\n".format(len(images))
    )
    with open(path, "w") as f:
        f.write(header)
        for _, img in images.items():
            line = " ".join(str(x) for x in [img.id, *img.qvec, *img.tvec, img.camera_id, img.name])
            f.write(line + "\n")
            f.write("\n")


def _write_points3D_text(points3D, path):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# Number of points: 0\n")


def _rotmat2qvec(R):
    """3x3 rotation matrix to (w, x, y, z) quaternion via axis-angle decomposition.

    The rotation axis is the real eigenvector of R associated with eigenvalue 1.
    The angle's magnitude comes from the trace; its sign is recovered with a
    cross-product test on a plane vector orthogonal to the axis.
    """
    eigvals, eigvecs = np.linalg.eig(R)
    eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
    u = eigvecs[:, np.argmax(eigvals)]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    theta = np.arccos((trace - 1) / 2)
    v = np.array([[u[1], -u[0], 0]]).T
    Rv = np.matmul(R, v)
    v_cross_Rv = np.cross(v.flatten(), Rv.flatten())
    is_negative = np.dot(v_cross_Rv, u) < 0
    theta = -theta if is_negative else theta
    qvec = np.concatenate(
        (np.array([np.cos(0.5 * theta)]), np.sin(0.5 * theta) * u), axis=0
    )
    return qvec


# ---------------------------------------------------------------------------
# Per-frame triangulation
# ---------------------------------------------------------------------------

def ply_nonempty(ply_path):
    """True iff `ply_path` exists and contains at least one vertex.

    Detects the 0-points failure mode where a PLY is written but contains no
    triangulated points — common when COLMAP's filter rejects every candidate.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        return False
    return len(PlyData.read(str(ply_path))["vertex"].data) > 0


def _required_square_pad(w, h):
    """Right + bottom pad needed to make the input square. Roma's
    `roma_outdoor` model is trained with square coarse_res (560x560) and
    upsample_res (864x864); after hloc's Resize-shorter-side, only square
    input images produce square output tensors that match the model's
    expected internal shapes. Returns (pad_right, pad_bottom); both 0 for
    already-square inputs (e.g. Blender-Moments' 800x800)."""
    if w > h:
        return 0, w - h
    if h > w:
        return h - w, 0
    return 0, 0


def _filter_pad_keypoints(feature_path, content_w, content_h, margin=0.5):
    """Drop ALIKED keypoints in the synthetic pad region (x >= content_w
    or y >= content_h, both with `margin` slack) from the feature h5 so they
    can't seed roma matches. Slices keypoints (N,2), keypoint_scores (N,),
    descriptors (D,N) in lockstep so the matcher sees consistent arrays."""
    with h5py.File(str(feature_path), "r+") as f:
        for name in list(f.keys()):
            grp = f[name]
            keypoints = grp["keypoints"][:]
            mask = ((keypoints[:, 0] < content_w - margin) &
                    (keypoints[:, 1] < content_h - margin))
            if mask.all():
                continue
            for key in ("keypoints", "keypoint_scores", "descriptors"):
                if key not in grp:
                    continue
                arr = grp[key][:]
                if arr.ndim == 1:
                    filtered = arr[mask]
                elif arr.shape[0] == mask.shape[0]:
                    filtered = arr[mask, ...]
                else:
                    filtered = arr[..., mask]
                attrs = dict(grp[key].attrs)
                del grp[key]
                grp.create_dataset(key, data=filtered)
                for k, v in attrs.items():
                    grp[key].attrs[k] = v


def _run_colmap(sfm_dir, focal_length, cx, cy, width, height, frames_data,
                focal_length_y=None, content_w=None, content_h=None,
                min_track_length=3, max_reproj_error=2):
    """Run hloc retrieval + extraction + matching + triangulation on one frame.

    Reads images from `sfm_dir/images/`, writes COLMAP intermediates under
    `sfm_dir/sparse/`, and exports `sfm_dir/colmap_points3D.ply` on success.

    `width` / `height` describe the staged image (which may be right- and/or
    bottom-padded past the LMDB image to make the input square for roma);
    when `content_w` and `content_h` are set, ALIKED keypoints in the
    synthetic pad region (x >= content_w or y >= content_h, both with 0.5px
    slack) are dropped after extraction so they can't seed matches there.
    """
    sfm_dir = Path(sfm_dir)
    image_dir = sfm_dir / "images"
    sparse_dir = sfm_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = sparse_dir / "0"
    recon_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = sparse_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    sfm_pairs = sparse_dir / "pairs-vlad.txt"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["aliked-n32"]
    matcher_conf = match_features.confs["aliked+roma"]

    # Non-square frames (e.g. selfcap) are matched at their native aspect rather
    # than square-padded: roma_outdoor accepts (H, W) coarse_res / upsample_res
    # tuples. coarse ~ native/sqrt(2), upsample ~ native, each rounded to a
    # multiple of 14 (roma's dinov2 backbone requirement). Override the conf's
    # square Resize transforms AND patch the roma factory hloc instantiates, so
    # the model's internal resolution matches the tensors it is handed. Also bump
    # ALIKED's resize_max so keypoints are detected near native resolution.
    non_square = int(width) != int(height)
    if non_square:
        import functools
        from torchvision.transforms import v2
        import torch
        from romatch import roma_outdoor as _roma_outdoor
        import hloc.matchers.roma as _hloc_roma
        r14 = lambda x: max(14, int(round(x / 14.0)) * 14)
        up14 = lambda x: max(14, math.ceil(x / 14.0) * 14)
        upsample_res = (up14(height), up14(width))
        coarse_res = (r14(height / math.sqrt(2)), r14(width / math.sqrt(2)))
        norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mk = lambda hw: v2.Compose([v2.Resize(hw), v2.PILToTensor(),
                                    v2.ToDtype(torch.float32, scale=True), norm])
        matcher_conf = {**matcher_conf, "transforms": [mk(coarse_res), mk(upsample_res)]}
        feature_conf = {**feature_conf,
                        "preprocessing": {**feature_conf["preprocessing"], "resize_max": 1920}}
        _hloc_roma.roma_outdoor = functools.partial(
            _roma_outdoor, coarse_res=coarse_res, upsample_res=upsample_res)
        print(f"  roma non-square: coarse_res={coarse_res}, upsample_res={upsample_res}, "
              f"resize_max=1920 (native {int(height)}x{int(width)})")

    _write_points3D_text({}, inputs_dir / "points3D.txt")
    fy = focal_length if focal_length_y is None else focal_length_y
    # Per-camera intrinsics when the transforms supply them (heterogeneous real
    # rigs), else one shared median PINHOLE (Blender-Moments' rigs).
    per_cam = all("cx_cam" in fr for fr in frames_data)

    cameras, im_data = {}, {}
    for i, frame_data in enumerate(frames_data):
        cid = (i + 1) if per_cam else 1
        if cid not in cameras:
            if per_cam:
                params = np.array([float(frame_data["fl_x_cam"]), float(frame_data["fl_y_cam"]),
                                   float(frame_data["cx_cam"]), float(frame_data["cy_cam"])])
            else:
                params = np.array([float(focal_length), float(fy), float(cx), float(cy)])
            cameras[cid] = _Camera(id=cid, model="PINHOLE", width=int(width), height=int(height),
                                   params=params)
        c2w = np.array(frame_data["transform_matrix"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        tvec = w2c[:3, 3].flatten()
        qvec = _rotmat2qvec(w2c[:3, :3]).flatten()
        image_name = "image_" + frame_data["file_path"].split("_")[-1]
        im_data[i + 1] = _Image(id=i + 1, qvec=qvec, tvec=tvec, camera_id=cid,
                                name=image_name, xys=None, point3D_ids=None)
    _write_cameras_text(cameras, inputs_dir / "cameras.txt")
    _write_images_text(im_data, inputs_dir / "images.txt")

    retrieval_path = extract_features.main(retrieval_conf, image_dir, sparse_dir)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
    feature_path = extract_features.main(feature_conf, image_dir, sparse_dir)
    if non_square:
        # hloc's roma matcher reads the feature h5 image_size as (h, w) but
        # extract_features stores it (w, h). For square inputs w==h so this is a
        # no-op, but non-square frames need the keypoint normalization to use the
        # correct axes or every match lands outside [-1, 1] and is rejected by
        # geometric verification. Store (h, w) so roma's indexing is correct;
        # image_size is consumed only by the matcher, not by triangulation.
        with h5py.File(str(feature_path), "r+") as f:
            for name in f.keys():
                sz = f[name]["image_size"][:]
                f[name]["image_size"][...] = sz[::-1]
    if content_w is not None and content_h is not None:
        _filter_pad_keypoints(feature_path, content_w, content_h)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"],
                                     sparse_dir, image_dir=image_dir)
    model = triangulation.main(recon_dir, inputs_dir, image_dir, sfm_pairs,
                               feature_path, match_path)

    # Drop low-quality 3D points (short tracks, high reprojection error).
    # Defaults (3 / 2px) suit the ~800px datasets; high-res rigs (e.g. selfcap)
    # pass a looser filter since their wide-baseline matching is sparser and the
    # px error scales with resolution.
    points_to_remove = [pid for pid, p in model.points3D.items()
                        if p.track.length() < min_track_length or p.error > max_reproj_error]
    for pid in points_to_remove:
        model.delete_point3D(pid)
    print(f"COLMAP: {len(model.points3D)} points after filtering "
          f"(track>={min_track_length}, err<={max_reproj_error}px)")

    model.export_PLY(sfm_dir / "colmap_points3D.ply")


def process_frame(source_path, frame, min_track_length=3, max_reproj_error=2,
                  native_unmasked=False, sfm_subdir="sfm_init"):
    """End-to-end COLMAP triangulation for a single frame.

    Args:
      source_path: Directory containing transforms_train.json + train/*.png
                   (the render output dir for one scene).
      frame: 1-indexed animation frame.
      native_unmasked: selfcap-only. Stage raw frames at native resolution with no
                   foreground masking and no square padding; _run_colmap then matches
                   them at their native aspect. Other datasets keep the square (mask-
                   if-present) path.
      sfm_subdir: output subdir under source_path (default `sfm_init`). selfcap passes
                   `sfm_init_backup` so the COLMAP recons (used only for pose BA) don't
                   collide with the shipped PCDs staged into `sfm_init` for the init.

    Output:
      <source_path>/<sfm_subdir>/frame_<N>/colmap_points3D.ply

    Returns the on-disk point count, or None if the frame was skipped.
    """
    source_path = Path(source_path)
    sfm_dir = source_path / sfm_subdir / f"frame_{frame}"
    ply_path = sfm_dir / "colmap_points3D.ply"
    if ply_nonempty(ply_path):
        print(f"frame {frame}: {ply_path} non-empty, skipping")
        return None

    transforms_path = source_path / "transforms_train.json"
    with open(transforms_path) as f:
        contents = json.load(f)
    frame_data = [fr for fr in contents["frames"] if fr["t"] == frame]
    if not frame_data:
        raise SystemExit(f"frame {frame}: no entries with t={frame} in {transforms_path}")
    print(f"frame {frame}: {len(frame_data)} train views")

    w_lmdb = int(contents["w"])
    h_lmdb = int(contents["h"])

    sfm_images = sfm_dir / "images"
    sfm_images.mkdir(parents=True, exist_ok=True)

    if native_unmasked:
        # selfcap: stage the raw frames at native resolution (no masking, no square
        # padding). _run_colmap matches non-square frames at their native aspect.
        staged_w, staged_h = w_lmdb, h_lmdb
        is_padded = False
        for fr in frame_data:
            src = (source_path / fr["file_path"]).resolve()
            suffix = src.name.split("_")[-1]
            dst = sfm_images / f"image_{suffix}"
            if dst.is_symlink() or dst.exists():
                try:
                    if Image.open(dst).size == (staged_w, staged_h):
                        continue
                except Exception:
                    pass
                dst.unlink()
            dst.symlink_to(src)
        print(f"frame {frame}: staged native images at {sfm_images}")
    else:
        pad_r, pad_b = _required_square_pad(w_lmdb, h_lmdb)
        staged_w, staged_h = w_lmdb + pad_r, h_lmdb + pad_b
        is_padded = (pad_r > 0 or pad_b > 0)
        # When per-view foreground masks are present (selfcap), black out the background
        # so COLMAP only extracts/triangulates the dynamic subject; the static background
        # is handled by the plate. Other datasets have no masks/ -> unchanged behavior.
        masks_dir = source_path / "masks"
        apply_masks = masks_dir.is_dir()
        for fr in frame_data:
            src = (source_path / fr["file_path"]).resolve()
            suffix = src.name.split("_")[-1]
            dst = sfm_images / f"image_{suffix}"
            if dst.is_symlink() or dst.exists():
                try:
                    if Image.open(dst).size == (staged_w, staged_h):
                        continue
                except Exception:
                    pass
                dst.unlink()
            mask_path = masks_dir / src.name
            if apply_masks and mask_path.exists():
                pil = Image.open(src).convert("RGB")
                m = Image.open(mask_path).convert("L").resize(pil.size)
                pil = Image.composite(pil, Image.new("RGB", pil.size, (0, 0, 0)), m)
                padded = Image.new("RGB", (staged_w, staged_h), (0, 0, 0))
                padded.paste(pil, (0, 0))
                padded.save(dst)
            elif pad_r == 0 and pad_b == 0:
                dst.symlink_to(src)
            else:
                pil = Image.open(src)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                padded = Image.new("RGB", (staged_w, staged_h), (0, 0, 0))
                padded.paste(pil, (0, 0))
                padded.save(dst)
        msg = f"frame {frame}: staged images at {sfm_images}"
        if pad_r > 0 or pad_b > 0:
            msg += (f" (padded {w_lmdb}x{h_lmdb}→{staged_w}x{staged_h} "
                    f"to square for roma)")
        print(msg)

    _run_colmap(str(sfm_dir), contents["fl_x"], contents["cx"], contents["cy"],
                staged_w, staged_h, frame_data,
                focal_length_y=contents.get("fl_y"),
                content_w=(w_lmdb if is_padded else None),
                content_h=(h_lmdb if is_padded else None),
                min_track_length=min_track_length, max_reproj_error=max_reproj_error)

    if not ply_path.exists():
        raise SystemExit(f"frame {frame}: pipeline did not produce {ply_path}")
    n_points = len(PlyData.read(str(ply_path))["vertex"].data)
    print(f"frame {frame}: {n_points} points written to {ply_path}")
    return n_points
