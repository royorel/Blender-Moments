"""Shared helpers for download_data.py and download_blend.py: fetch files from
the Blender-Moments HuggingFace dataset and verify them against its published
checksums. Needs only the standard library and huggingface_hub."""
import hashlib
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "royorel/Blender-Moments"
SCENES = ["alligator", "beetle", "bicycle", "book", "elephant", "ficus_4d",
          "lego_4d", "spiderman", "toyota"]
DATASET_ROOT_DEFAULT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(repo_path: str, work_dir: Path, revision) -> Path:
    """Fetch one repository file into work_dir, bypassing the HuggingFace
    cache so multi-gigabyte archives are not stored twice."""
    return Path(hf_hub_download(REPO_ID, repo_path, repo_type="dataset",
                                revision=revision, local_dir=work_dir))


def load_checksums(work_dir: Path, revision) -> dict:
    """{repo path: sha256} for every published file."""
    sums = {}
    for line in download("checksums.sha256", work_dir, revision).read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        sums[path] = digest
    return sums


def fetch_verified(repo_path: str, expected_sha: str, work_dir: Path, revision) -> Path:
    path = download(repo_path, work_dir, revision)
    actual = sha256_file(path)
    if actual != expected_sha:
        path.unlink()
        raise SystemExit(f"{repo_path}: checksum mismatch ({actual} != {expected_sha}); "
                         f"rerun to fetch it again")
    return path


def work_dir_for(dataset_root: Path):
    """A temporary directory on the dataset's filesystem, so verified files
    move into place without a cross-device copy."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=dataset_root, prefix=".download_")


def move(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
