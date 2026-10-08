"""Create the file-level checksum manifest for a frozen Dropbox bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("S2S_DATA_DIR", ROOT / "data")),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data_manifest.csv")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.name not in {"README.md", ".DS_Store"}
    )
    if not files:
        raise SystemExit(f"No bundle files found under {data_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        for index, path in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {path.relative_to(data_dir)}", flush=True)
            writer.writerow({
                "path": path.relative_to(data_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    print(f"Wrote {len(files)} records to {args.output}")


if __name__ == "__main__":
    main()
