"""Download OULAD from the UCI repository and verify its SHA-256 before extracting."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from riskml.config import DEFAULT_DATA_DIR, OULAD_SHA256, OULAD_URL

EXPECTED_FILES = (
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_oulad(dest: Path = DEFAULT_DATA_DIR, url: str = OULAD_URL) -> Path:
    """Fetch, verify and extract OULAD into ``dest``. Idempotent."""
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).exists() for f in EXPECTED_FILES):
        return dest
    archive = dest / "oulad.zip"
    if not archive.exists() or sha256_of(archive) != OULAD_SHA256:
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
        tmp.replace(archive)
    actual = sha256_of(archive)
    if actual != OULAD_SHA256:
        raise RuntimeError(f"OULAD checksum mismatch: expected {OULAD_SHA256}, got {actual}")
    with zipfile.ZipFile(archive) as zf:
        for name in EXPECTED_FILES:
            zf.extract(name, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print(f"OULAD ready in {download_oulad(args.dest)}")


if __name__ == "__main__":
    main()
