"""Download the Amazon Reviews 2023 All_Beauty review file.

Source: Hou, Li, He, Yan, Chen and McAuley (2024), "Bridging Language and Items
for Retrieval and Recommendation", https://amazon-reviews-2023.github.io/
The dataset states no licence, so the file is downloaded here and never
committed to the repository.

Usage:
    uv run python scripts/download_data.py           # skips if already present
    uv run python scripts/download_data.py --force   # downloads again
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi

FILE_NAME = "All_Beauty.jsonl.gz"
# Both addresses are published by the dataset authors (dataset card and project page).
URLS = (
    f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/{FILE_NAME}",
    f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{FILE_NAME}",
)
REQUIRED_FIELDS = {"rating", "text", "parent_asin", "user_id", "timestamp"}
CHUNK_BYTES = 1 << 20
REPORT_EVERY_BYTES = 10 << 20


def download(url: str, target: Path) -> None:
    """Stream ``url`` into ``target`` through a temporary ``.part`` file."""
    partial = target.with_name(target.name + ".part")
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "dsa4213-a1-download/1.0"})
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        next_report = REPORT_EVERY_BYTES
        with partial.open("wb") as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                received += len(chunk)
                if received >= next_report:
                    of_total = f" of {total / 1e6:.0f}" if total else ""
                    print(f"  {received / 1e6:.0f}{of_total} MB", flush=True)
                    next_report += REPORT_EVERY_BYTES
    if total and received != total:
        raise OSError(f"incomplete download: {received} of {total} bytes")
    partial.replace(target)


def check_file(path: Path) -> None:
    """Fail loudly unless the file is readable gzip JSON Lines in the expected format."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"first record lacks fields {sorted(missing)}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="download folder")
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / FILE_NAME
    if target.exists() and not args.force:
        print(f"{target} already exists; use --force to download it again.")
    else:
        for url in URLS:
            print(f"Downloading {url}")
            try:
                download(url, target)
                break
            except (urllib.error.URLError, OSError) as error:
                print(f"  failed: {error}", file=sys.stderr)
        else:
            print(
                "Both addresses failed. Download the file in a browser from\n"
                f"  {URLS[1]}\nand save it as {target}, then re-run this script.",
                file=sys.stderr,
            )
            return 1

    check_file(target)
    checksum = sha256_of(target)
    (args.out / f"{FILE_NAME}.sha256").write_text(f"{checksum}  {FILE_NAME}\n", encoding="utf-8")
    print(f"OK: {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
