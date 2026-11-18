"""Download the Blender-Moments source .blend files into raw/, for generating
a new version of the data.

Each file is verified against the published checksum. A file already in raw/
with a matching checksum is skipped.

Usage:
    python download_blend.py                       # all scenes
    python download_blend.py --scenes lego_4d      # one scene
    python download_blend.py --dry-run             # list the files
"""
import argparse
from pathlib import Path

from hf_download import (DATASET_ROOT_DEFAULT, SCENES, fetch_verified, load_checksums, move,
                         sha256_file, work_dir_for)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES,
                        metavar="SCENE", help=f"scenes to fetch (default all): {', '.join(SCENES)}")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--revision", default=None,
                        help="dataset revision (branch, tag, or commit); default latest")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the files that would be fetched and exit")
    args = parser.parse_args()

    plan = [f"{s}/{s}.blend" for s in args.scenes]
    if args.dry_run:
        print("\n".join(plan))
        return

    with work_dir_for(args.dataset_root) as tmp:
        work_dir = Path(tmp)
        sums = load_checksums(work_dir, args.revision)
        for scene, repo_path in zip(args.scenes, plan):
            dest = args.dataset_root / "raw" / f"{scene}.blend"
            if dest.is_file() and sha256_file(dest) == sums[repo_path]:
                print(f"{repo_path}: already present and verified", flush=True)
                continue
            move(fetch_verified(repo_path, sums[repo_path], work_dir, args.revision), dest)
            print(f"{repo_path} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
