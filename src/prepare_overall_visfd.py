
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VISD4SA_DIR = ROOT / "data" / "visd4sa"
VISFD_DIR = ROOT / "data" / "visfd_raw"
OUTPUT_DIR = ROOT / "data" / "overall_visfd"

SPLITS = ("train", "dev", "test")


def normalize_text(text):
    """Chuẩn hóa để đối chiếu review giữa hai dataset."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.strip().casefold()
    return re.sub(r"\s+", " ", text)


def stars_to_sentiment(stars):
    """Chuyển số sao thành ba mức cảm xúc."""
    stars = int(stars)

    if stars in (1, 2):
        return "negative"
    if stars == 3:
        return "neutral"
    if stars in (4, 5):
        return "positive"

    raise ValueError(f"Số sao không hợp lệ: {stars}")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc UIT-ViSFD, không phụ thuộc cách chia split của nó.
    rating_by_text = defaultdict(set)

    for split in SPLITS:
        path = VISFD_DIR / f"{split.capitalize()}.csv"
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            text = row["comment"]
            stars = row["n_star"]

            if pd.isna(text) or pd.isna(stars):
                continue

            key = normalize_text(text)
            if not key:
                continue

            if not isinstance(stars, (int, float)) or not float(stars).is_integer():
                continue

            stars = int(stars)
            if stars not in (1, 2, 3, 4, 5):
                continue

            rating_by_text[key].add(stars)

    # Một review có nhiều số sao khác nhau sẽ không được ghép.
    valid_ratings = {
        key: next(iter(stars))
        for key, stars in rating_by_text.items()
        if len(stars) == 1
    }

    conflicting_ratings = sum(
        len(stars) > 1 for stars in rating_by_text.values()
    )

    # Đọc UIT-ViSD4SA và kiểm tra review xuất hiện ở nhiều split.
    original = {
        split: read_jsonl(VISD4SA_DIR / f"{split}.jsonl")
        for split in SPLITS
    }

    splits_by_text = defaultdict(set)

    for split, rows in original.items():
        for row in rows:
            splits_by_text[normalize_text(row["text"])].add(split)

    # Loại review xuất hiện ở nhiều split để tránh rò rỉ dữ liệu.
    cross_split_keys = {
        key for key, splits in splits_by_text.items()
        if len(splits) > 1
    }

    report = {
        "label_source": "UIT-ViSFD n_star",
        "label_mapping": {
            "1,2": "negative",
            "3": "neutral",
            "4,5": "positive"
        },
        "conflicting_rating_texts": conflicting_ratings,
        "cross_split_texts_excluded": len(cross_split_keys),
        "splits": {}
    }

    for split, rows in original.items():
        processed = []
        seen = set()
        reasons = Counter()

        for row in rows:
            text = row["text"]
            key = normalize_text(text)

            if not key:
                reasons["empty_text"] += 1
                continue

            if key in cross_split_keys:
                reasons["cross_split_duplicate"] += 1
                continue

            if key in seen:
                reasons["duplicate_within_split"] += 1
                continue

            seen.add(key)

            if key not in valid_ratings:
                if key in rating_by_text:
                    reasons["conflicting_rating"] += 1
                else:
                    reasons["unmatched_review"] += 1
                continue

            stars = valid_ratings[key]

            processed.append({
                "text": text,
                "overall": stars_to_sentiment(stars),
                "n_star": stars,
                "source": "visfd_rating"
            })

        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", processed)

        counts = Counter(row["overall"] for row in processed)

        report["splits"][split] = {
            "original": len(rows),
            "matched": len(processed),
            "excluded": len(rows) - len(processed),
            "excluded_reasons": dict(reasons),
            "labels": dict(counts)
        }

        print(f"\n{split.upper()}")
        print("Review gốc:", len(rows))
        print("Ghép thành công:", len(processed))
        print("Bị loại:", len(rows) - len(processed))
        print("Lý do loại:", dict(reasons))
        print("Phân bố nhãn:", dict(counts))

    report_path = OUTPUT_DIR / "preprocessing_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nĐã lưu dữ liệu vào:", OUTPUT_DIR)


if __name__ == "__main__":
    main()