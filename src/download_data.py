"""Download the original UIT-ViSD4SA dataset."""

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/"

COUNTS = {
    "train": 7785,
    "dev": 1112,
    "test": 2225,
}


def main(folder):
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    for split, expected in COUNTS.items():
        path = out / f"{split}.jsonl"

        # Download only if the file does not already exist
        if not path.exists():
            urllib.request.urlretrieve(BASE + path.name, path)

        # Count original reviews without parsing aspect annotations
        with path.open(encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())

        if count != expected:
            raise ValueError(
                f"{split}: got {count} records; expected {expected}"
            )

        print(f"{split}: {count} raw reviews")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/visd4sa")
    args = parser.parse_args()

    main(args.output)