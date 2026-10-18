"""Data loading and strict review-level label construction. No heuristic labels hidden."""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

POLARITIES = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
              'positive ': 'positive', 'negative ': 'negative'}


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip().casefold()


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_span(label, text, inclusive_end=False):
    """Accept dict or [start,end,category,polarity] / [start,end,'CAT#POL'].
    UIT-ViSD4SA original gold offsets are inclusive; internally use half-open offsets.
    Fail on unknown schema rather than silently corrupt labels.
    """
    if isinstance(label, dict):
        start = label.get('start', label.get('from'))
        end = label.get('end', label.get('to'))
        category = label.get('category', label.get('aspect'))
        polarity = label.get('polarity', label.get('sentiment'))
        if category is None and isinstance(label.get('label'), str):
            category = label['label']
    elif isinstance(label, (list, tuple)) and len(label) >= 3:
        start, end = label[0], label[1]
        category = label[2]
        polarity = label[3] if len(label) >= 4 else None
    else:
        raise ValueError(f'Unknown span schema: {label!r}')
    if isinstance(category, (list, tuple)) and len(category) == 2:
        category, polarity = category
    if isinstance(category, str) and '#' in category and polarity is None:
        category, polarity = category.rsplit('#', 1)
    if start is None or end is None or category is None or polarity is None:
        raise ValueError(f'Incomplete span: {label!r}')
    start, end = int(start), int(end)
    if inclusive_end:
        end += 1
    if not 0 <= start < end <= len(text):
        raise ValueError(f'Invalid offsets {start}:{end}, text length {len(text)}')
    polarity = str(polarity).lower().strip()
    if polarity not in ('positive', 'negative', 'neutral'):
        raise ValueError(f'Unknown polarity: {polarity}')
    return {'start': start, 'end': end, 'category': str(category), 'polarity': polarity,
            'text': text[start:end]}


def read_aspects(path, invalid_policy="raise", audit_path=None):
    """
    Read UIT-ViSD4SA annotations using end-exclusive offsets.

    invalid_policy:
        "raise": stop when an invalid annotation is found.
        "skip": exclude the entire review containing invalid spans.

    Raw dataset files are never modified.
    """
    if invalid_policy not in ("raise", "skip"):
        raise ValueError(f"Unknown invalid_policy: {invalid_policy}")

    rows = []
    rejected = []

    for line_no, item in enumerate(load_jsonl(path), 1):
        raw = item.get("text", item.get("text_raw"))
        labels = item.get("labels")

        try:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Missing or empty text")

            if not isinstance(labels, list):
                raise ValueError("Missing labels list")

            spans = [parse_span(label, raw) for label in labels]
            spans.sort(key=lambda x: (x["start"], x["end"]))

            if any(
                a["end"] > b["start"]
                for a, b in zip(spans, spans[1:])
            ):
                raise ValueError("Overlapping aspect spans")

            rows.append({
                "text": raw,
                "spans": spans,
                "source_line": line_no
            })

        except (ValueError, TypeError) as exc:
            rejected.append({
                "line": line_no,
                "reason": str(exc)
            })

            if invalid_policy == "raise":
                raise ValueError(
                    f"{path}:{line_no}: {exc}"
                ) from exc

    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        with audit_path.open("w", encoding="utf-8") as f:
            for item in rejected:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"{Path(path).name}: "
        f"{len(rows)} accepted, "
        f"{len(rejected)} rejected"
    )

    return rows


def load_stars_csv(path, text_col='comment', star_col='n_star'):
    import pandas as pd
    df = pd.read_csv(path)
    if text_col not in df or star_col not in df:
        raise ValueError(f'Expected columns {text_col}, {star_col}. Found {df.columns.tolist()}')
    return [(str(t), int(float(s))) for t,s in zip(df[text_col], df[star_col]) if pd.notna(t) and pd.notna(s)]


def star_polarity(stars):
    if stars not in (1,2,3,4,5): raise ValueError(f'Invalid stars {stars}')
    return 'negative' if stars <= 2 else ('neutral' if stars == 3 else 'positive')


def attach_overall_by_exact_text(splits, star_entries):
    """Only unambiguous normalized-text matching. No fuzzy guessing or split mixing."""
    grouped = defaultdict(set)
    for text, stars in star_entries:
        grouped[norm(text)].add(star_polarity(stars))
    stats = Counter()
    output = {}
    for split, rows in splits.items():
        matched = []
        for row in rows:
            possible = grouped.get(norm(row['text']), set())
            if len(possible) == 1:
                matched.append({**row, 'overall': next(iter(possible))})
                stats[f'{split}_matched'] += 1
            else:
                stats[f'{split}_unmatched_or_conflicting'] += 1
        output[split] = matched
    return output, dict(stats)


def load_splits(folder, invalid_policy="skip", remove_duplicate=True):
    folder = Path(folder)

    # Đọc dữ liệu và bỏ các review có annotation không hợp lệ
    splits = {
        split: read_aspects(
            folder / f"{split}.jsonl",
            invalid_policy=invalid_policy,
            audit_path=Path("outputs/data_audit") / f"{split}_rejected.jsonl"
        )
        for split in ("train", "dev", "test")
    }

    # Loại review trùng khỏi tập train
    if remove_duplicate:
        splits, removed = remove_duplicates(splits)
        print(f"Đã loại {len(removed)} review trùng khỏi train")

    return splits
    
def remove_duplicates(splits):
    """Loại review trùng khỏi train, giữ nguyên dev và test."""
    from collections import Counter

    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]

    # Đếm số lần xuất hiện của từng review trong train
    train_counts = Counter(norm(row["text"]) for row in train)

    # Các review đã xuất hiện trong dev hoặc test
    eval_texts = {
        norm(row["text"])
        for row in dev + test
    }

    cleaned_train = []
    removed = []

    for row in train:
        review_key = norm(row["text"])

        if train_counts[review_key] > 1:
            reason = "duplicate_in_train"
        elif review_key in eval_texts:
            reason = "duplicate_with_eval"
        else:
            cleaned_train.append(row)
            continue

        removed.append({
            "source_line": row.get("source_line"),
            "reason": reason
        })

    return {
        "train": cleaned_train,
        "dev": dev,
        "test": test
    }, removed