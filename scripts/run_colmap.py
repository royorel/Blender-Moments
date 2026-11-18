"""Sequential per-frame COLMAP runner.

Loops over `[frame_start, frame_end]` and triangulates each frame
in-process via `colmap_pipeline.process_frame`. Per-frame skip-if-exists
makes re-running the script cheap on partial outputs.

Usage:
    python run_colmap.py --source_path <render_output_dir> \
                         [--frame_start N] [--frame_end N]

The source path must contain `transforms_train.json` + `train/*.png`
(the output of `render.py` / `render_all.py` for one scene). PLYs land
at `<source_path>/sfm_init/frame_<N>/colmap_points3D.ply`.
"""
import argparse
import json
from pathlib import Path

from colmap_pipeline import process_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_path", type=Path, required=True,
                        help="directory containing transforms_train.json + train/*.png")
    parser.add_argument("--frame_start", type=int, default=1,
                        help="first frame to process (1-indexed, inclusive)")
    parser.add_argument("--frame_end", type=int, default=None,
                        help="last frame to process (inclusive). Defaults to "
                             "the max t in transforms_train.json.")
    args = parser.parse_args()

    transforms_path = args.source_path / "transforms_train.json"
    if not transforms_path.exists():
        raise SystemExit(f"missing {transforms_path}; run rendering first")

    if args.frame_end is None:
        with open(transforms_path) as f:
            contents = json.load(f)
        args.frame_end = max(fr["t"] for fr in contents["frames"])
        print(f"frame_end defaulted to max t = {args.frame_end}")

    for frame in range(args.frame_start, args.frame_end + 1):
        process_frame(args.source_path, frame)


if __name__ == "__main__":
    main()
