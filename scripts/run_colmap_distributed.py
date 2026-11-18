"""Single-frame COLMAP worker for sbatch fan-out.

One frame per sbatch job. Used by `run_colmap_all.py --mode distributed`
to parallelise COLMAP across the cluster.

Usage:
    python run_colmap_distributed.py --source_path <render_output_dir> --frame N

Identical pipeline body to `run_colmap.py`'s per-iteration step.
"""
import argparse
from pathlib import Path

from colmap_pipeline import process_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_path", type=Path, required=True,
                        help="directory containing transforms_train.json + train/*.png")
    parser.add_argument("--frame", type=int, required=True,
                        help="1-indexed animation frame")
    parser.add_argument("--min_track_length", type=int, default=3,
                        help="drop 3D points seen by fewer than this many views")
    parser.add_argument("--max_reproj_error", type=float, default=2.0,
                        help="drop 3D points with reprojection error above this (px)")
    parser.add_argument("--native-unmasked", dest="native_unmasked", action="store_true",
                        help="selfcap: stage raw native-resolution frames (no mask, no "
                             "square pad) and match at native aspect")
    parser.add_argument("--sfm_subdir", default="sfm_init",
                        help="output subdir under source_path (default sfm_init); selfcap "
                             "uses sfm_init_backup since its sfm_init holds shipped PCDs")
    args = parser.parse_args()

    if not (args.source_path / "transforms_train.json").exists():
        raise SystemExit(f"missing {args.source_path}/transforms_train.json")

    process_frame(args.source_path, args.frame,
                  min_track_length=args.min_track_length,
                  max_reproj_error=args.max_reproj_error,
                  native_unmasked=args.native_unmasked,
                  sfm_subdir=args.sfm_subdir)


if __name__ == "__main__":
    main()
