"""Download the Blender-Moments data (renders, ground-truth meshes, and
per-frame COLMAP point clouds) and unpack it into the dataset layout.

Each archive is checked against the published checksum and unpacked at the
dataset root, where zip's own per-file CRCs check every extracted file. A
small record of each unpacked archive (its checksum and file sizes) lets a
rerun skip archives whose files are all still in place, so an interrupted
run resumes where it stopped.

Usage:
    python download_data.py                                  # everything
    python download_data.py --scenes lego_4d book            # two scenes
    python download_data.py --components meshes              # meshes only
    python download_data.py --dry-run                        # list the archives
"""
import argparse
import json
import zipfile
from pathlib import Path

from hf_download import DATASET_ROOT_DEFAULT, SCENES, fetch_verified, load_checksums, work_dir_for


COMPONENTS = {
    "images": ["renders_train.zip", "renders_test.zip"],
    "meshes": ["gt_meshes.zip"],
    "sfm": ["sfm_init.zip"],
}
STATE_DIR = ".download_state"


def planned_files(scenes, components):
    return [f"{s}/{a}" for s in scenes for c in components for a in COMPONENTS[c]]


def record_path(root: Path, repo_path: str) -> Path:
    return root / STATE_DIR / f"{repo_path}.json"


def already_unpacked(root: Path, repo_path: str, digest: str) -> bool:
    """True when this archive version was unpacked here and every file it
    holds is still present at its recorded size."""
    record = record_path(root, repo_path)
    if not record.is_file():
        return False
    state = json.loads(record.read_text())
    return state["sha256"] == digest and all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in state["sizes"].items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES,
                        metavar="SCENE", help=f"scenes to fetch (default all): {', '.join(SCENES)}")
    parser.add_argument("--components", nargs="+", choices=list(COMPONENTS),
                        default=list(COMPONENTS),
                        help="what to fetch (default all)")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--revision", default=None,
                        help="dataset revision (branch, tag, or commit); default latest")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the archives that would be fetched and exit")
    args = parser.parse_args()

    plan = planned_files(args.scenes, args.components)
    if args.dry_run:
        print("\n".join(plan))
        return

    with work_dir_for(args.dataset_root) as tmp:
        work_dir = Path(tmp)
        sums = load_checksums(work_dir, args.revision)
        for repo_path in plan:
            if already_unpacked(args.dataset_root, repo_path, sums[repo_path]):
                print(f"{repo_path}: already downloaded", flush=True)
                continue
            archive = fetch_verified(repo_path, sums[repo_path], work_dir, args.revision)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(args.dataset_root)
                sizes = {i.filename: i.file_size for i in zf.infolist() if not i.is_dir()}
            archive.unlink()
            record = record_path(args.dataset_root, repo_path)
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"sha256": sums[repo_path], "sizes": sizes}))
            print(f"{repo_path}: {len(sizes)} files", flush=True)


if __name__ == "__main__":
    main()
