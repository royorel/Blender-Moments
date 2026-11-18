"""Per-scene synthetic-GT mesh extraction.

Opens a .blend, iterates N evenly-spaced frames over [frame_start,
frame_end] inclusive (or the explicit --frames), writes one triangle PLY
per frame to <output_dir>/frame_<NNNN>.ply (binary little-endian, world
coords), plus scene_normalization.json carrying the AABB diagonal across
the union of all extracted frames.

Runs INSIDE the `blender_moments` conda env (imports bpy directly). The
orchestrator (extract_mesh_all.py) invokes this via `conda run -n
blender_moments python -u extract_mesh.py ...`.

For each frame the extraction depsgraph-evaluates all render-visible
MESH objects (applies Multires, Subdivision, Displace, Solidify,
Skin, Armature posing, etc.) and concatenates them in world coords.
Quads / n-gons are fan-triangulated. Objects or collections with
hide_render set are skipped (rigging icospheres, control proxies,
etc.), as are light-emitting meshes (area-light planes modeled with
an emissive material), which are scene lighting rather than GT surface
and otherwise land far outside the object, corrupting the union AABB.
"""
import argparse
import json
import math
import struct
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", required=True, type=Path,
                        help="Path to the .blend file")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Where frame_<NNNN>.ply + scene_normalization.json land")
    parser.add_argument("--frame_start", required=True, type=int)
    parser.add_argument("--frame_end", required=True, type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--n_frames", type=int, default=10,
                           help="Number of evenly-spaced frames to extract")
    selection.add_argument("--frames", type=int, nargs="+",
                           help="Explicit frame numbers to extract")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_mesh] loading {args.object_path}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.object_path))

    frames = args.frames or _evenly_spaced_frames(args.frame_start, args.frame_end,
                                                  args.n_frames)
    print(f"[extract_mesh] frames: {frames}", flush=True)

    union_min = [math.inf, math.inf, math.inf]
    union_max = [-math.inf, -math.inf, -math.inf]

    for frame_idx in frames:
        bpy.context.scene.frame_set(frame_idx)
        verts, faces, fmin, fmax = _extract_world_mesh()
        out_ply = args.output_dir / f"frame_{frame_idx:04d}.ply"
        _write_ply(out_ply, verts, faces)
        for i in range(3):
            union_min[i] = min(union_min[i], fmin[i])
            union_max[i] = max(union_max[i], fmax[i])
        print(
            f"[extract_mesh] frame {frame_idx}: "
            f"{len(verts)} verts, {len(faces)} tris -> {out_ply.name}",
            flush=True,
        )

    diagonal = math.sqrt(sum((union_max[i] - union_min[i]) ** 2 for i in range(3)))
    norm = {
        "diagonal": diagonal,
        "min_bound": [float(x) for x in union_min],
        "max_bound": [float(x) for x in union_max],
    }
    norm_path = args.output_dir / "scene_normalization.json"
    norm_path.write_text(json.dumps(norm, indent=2))
    print(f"[extract_mesh] union-AABB diagonal = {diagonal:.6f} -> {norm_path.name}",
          flush=True)


def _evenly_spaced_frames(start, end, n):
    """N integer frame indices at the midpoints of N equal sub-
    intervals of [start, end] inclusive. Avoids the boundary frames,
    which are typically pose-reset / anchor frames in animations.

    Examples,
        [1, 100], n=10  ->  5, 15, 25, ..., 95
        [1, 125], n=10  ->  6, 18, 31, 43, 56, 68, 81, 93, 106, 118
        [1, 150], n=10  ->  7, 22, 37, ..., 142
    """
    length = end - start + 1
    step = length / n
    offset = int(step / 2)
    return [start - 1 + offset + int(k * step) for k in range(n)]


def _is_render_visible(obj):
    """True iff the object would appear in a render at the current
    frame. Skips objects the artist marked hide_render on the object
    itself or on any collection ancestor."""
    if obj.hide_render:
        return False
    for coll in obj.users_collection:
        if coll.hide_render:
            return False
    return True


def _material_is_pure_light(mat):
    """True iff the material's active surface output is a pure `Emission`
    shader, the canonical way an area light is modeled as a mesh plane.

    A `Principled BSDF` with a non-zero Emission Strength is deliberately
    NOT flagged, real objects legitimately use Principled emission for
    glowing sub-parts (e.g. a bike's 'Light And Reflector' material), and
    those must be kept."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    nodes = mat.node_tree.nodes
    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None:
        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        return False
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return False
    return surf.links[0].from_node.type == "EMISSION"


def _is_light_emitting(obj):
    """True iff the object is a dedicated light, i.e. it has materials and
    every one of them is a pure-Emission light plane. Such meshes are scene
    lighting, not GT surface, and are excluded. A real object that merely
    has one emissive sub-material among several (a bike reflector, an LED)
    is kept, since not all of its materials are pure lights."""
    mats = [slot.material for slot in obj.material_slots if slot.material is not None]
    return bool(mats) and all(_material_is_pure_light(m) for m in mats)


def _extract_world_mesh():
    """Walk all render-visible MESH objects in the evaluated depsgraph
    and concatenate verts / faces in world coords. Tracks per-frame
    AABB along the way (used by the caller for union-AABB
    accumulation)."""
    deps = bpy.context.evaluated_depsgraph_get()
    all_verts, all_faces = [], []
    bbox_min = [math.inf, math.inf, math.inf]
    bbox_max = [-math.inf, -math.inf, -math.inf]
    offset = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not _is_render_visible(obj):
            continue
        if _is_light_emitting(obj):
            print(f"[extract_mesh]   excluding light-emitting mesh '{obj.name}'",
                  flush=True)
            continue
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        mat = obj.matrix_world
        for v in mesh.vertices:
            wp = mat @ v.co
            all_verts.append((wp.x, wp.y, wp.z))
            for i, c in enumerate((wp.x, wp.y, wp.z)):
                if c < bbox_min[i]:
                    bbox_min[i] = c
                if c > bbox_max[i]:
                    bbox_max[i] = c
        for poly in mesh.polygons:
            idxs = list(poly.vertices)
            for i in range(1, len(idxs) - 1):
                all_faces.append((
                    offset + idxs[0],
                    offset + idxs[i],
                    offset + idxs[i + 1],
                ))
        offset += len(mesh.vertices)
        eval_obj.to_mesh_clear()
    return all_verts, all_faces, bbox_min, bbox_max


def _write_ply(path, verts, faces):
    n_verts = len(verts)
    n_faces = len(faces)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode())
        for v in verts:
            f.write(struct.pack("<fff", *v))
        for face in faces:
            f.write(struct.pack("<B3i", 3, *face))


if __name__ == "__main__":
    main()
