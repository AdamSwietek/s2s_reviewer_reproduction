"""Download and verify the separately distributed S2S input bundle."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://www.dropbox.com/scl/fo/yvqj2aku2f2qhld70dyvh/"
    "AIZWOsbXFX0_8aAU-8MsElo?rlkey=g5ng7brlr1do807ti17e676q5&dl=1"
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=os.environ.get("S2S_DATA_URL", DEFAULT_URL),
        help="Dropbox direct-download URL for the ZIP archive.",
    )
    parser.add_argument(
        "--sha256", default=os.environ.get("S2S_DATA_SHA256"),
        help="Expected SHA-256 digest. Required for a verified release.",
    )
    parser.add_argument(
        "--destination", type=Path,
        default=Path(os.environ.get("S2S_DATA_DIR", ROOT / "data")),
    )
    parser.add_argument("--archive", type=Path, help="Use an existing ZIP.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    placeholders = {"README.md", ".gitkeep"}
    existing = (
        [path for path in destination.iterdir() if path.name not in placeholders]
        if destination.exists() else []
    )
    if existing and not args.force:
        raise SystemExit(
            f"Destination is not empty: {destination}. Use --force only after "
            "confirming that its contents may be supplemented or overwritten."
        )
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="s2s-data-") as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / "s2s_nc_data.zip"
            print(f"Downloading {args.url}")
            with urllib.request.urlopen(args.url) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        archive = archive.expanduser().resolve()
        actual = sha256(archive)
        print(f"Archive SHA-256: {actual}")
        if args.sha256 and actual.lower() != args.sha256.lower():
            raise SystemExit(
                f"Checksum mismatch: expected {args.sha256}, received {actual}"
            )
        if not args.sha256:
            print("WARNING: no expected checksum was supplied; integrity is unverified.")
        safe_extract(archive, destination)

    print(f"Data extracted to {destination}")
    print("Next: python scripts/validate_s2s_data.py")


if __name__ == "__main__":
    main()
