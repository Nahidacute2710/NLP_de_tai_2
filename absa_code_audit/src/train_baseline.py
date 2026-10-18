
"""Train baseline TF-IDF cho bài toán Overall Sentiment."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .data import load_splits, load_stars_csv
from .overall import build_overall, LABELS
from .audit import audit


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y, pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=LABELS
        ).tolist(),
        "label_order": LABELS,
        "report": classification_report(
            y, pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def main(args):
    # 1. Đọc dữ liệu theo nguồn nhãn.
    if args.overall_source == "visfd":
        splits = {}

        for split in ("train", "dev", "test"):
            path = Path(args.data) / f"{split}.jsonl"

            with path.open(encoding="utf-8") as f:
                splits[split] = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

        raw = splits
        stats = {
            "source": "UIT-ViSFD",
            "method": "rating_derived",
            "mapping": {
                "1-2": "negative",
                "3": "neutral",
                "4-5": "positive",
            },
        }

    else:
        raw = load_splits(args.data)

        stars = (
            load_stars_csv(
                args.stars_csv,
                args.text_col,
                args.star_col,
            )
            if args.stars_csv
            else []
        )

        splits, stats = build_overall(
            raw, stars, args.label_strategy
        )

    # 2. Kiểm tra dữ liệu.
    leak = audit(raw)

    if leak["cross_split_duplicate_text_count"] and not args.allow_overlap:
        raise ValueError(
            "Cross-split duplicate review text detected. "
            "Inspect audit; --allow-overlap only if explicitly justified."
        )

    if any(not splits[k] for k in ("train", "dev", "test")):
        raise ValueError("At least one split has zero overall labels")

    if len({r["overall"] for r in splits["train"]}) < 2:
        raise ValueError("Need at least two distinct training classes")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "label_provenance": stats,
        "overlap_audit": leak,
        "selection": [],
    }

    xtr = [r["text"] for r in splits["train"]]
    ytr = [r["overall"] for r in splits["train"]]

    xdv = [r["text"] for r in splits["dev"]]
    ydv = [r["overall"] for r in splits["dev"]]

    best, best_score = None, -1

    # 3. Chạy 9 cấu hình baseline như trước.
    experiments = {
        "Logistic Regression": [
            (
                c,
                LogisticRegression(
                    C=c,
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            )
            for c in args.c_values
        ],
        "Naive Bayes": [
            (alpha, MultinomialNB(alpha=alpha))
            for alpha in args.alpha_values
        ],
        "Linear SVM": [
            (
                c,
                LinearSVC(
                    C=c,
                    class_weight="balanced",
                    random_state=args.seed,
                    max_iter=5000,
                ),
            )
            for c in args.c_values
        ],
    }

    for model_name, configurations in experiments.items():
        for value, classifier in configurations:
            pipe = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=60000,
                    ),
                ),
                ("clf", classifier),
            ])

            pipe.fit(xtr, ytr)

            dev_metrics = metrics(ydv, pipe.predict(xdv))
            parameter = (
                "alpha" if model_name == "Naive Bayes" else "C"
            )

            result["selection"].append({
                "model": model_name,
                "parameter": parameter,
                "value": value,
                "dev": dev_metrics,
            })

            score = dev_metrics["macro_f1"]

            print(
                f"{model_name} | {parameter}={value} | "
                f"Dev Macro-F1={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best = pipe

                result["selected_model"] = model_name
                result["selected_parameter"] = parameter
                result["selected_value"] = value

    # 4. Đánh giá model tốt nhất trên Test.
    result["dev"] = metrics(ydv, best.predict(xdv))

    xte = [r["text"] for r in splits["test"]]
    yte = [r["overall"] for r in splits["test"]]
    result["test"] = metrics(yte, best.predict(xte))

    # 5. Lưu model và kết quả.
    joblib.dump(best, out / "baseline.joblib")

    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nKet qua cuoi cung:")
    print(json.dumps({
        "selected_model": result["selected_model"],
        "selected_parameter": result["selected_parameter"],
        "selected_value": result["selected_value"],
        "dev_macro_f1": result["dev"]["macro_f1"],
        "test_accuracy": result["test"]["accuracy"],
        "test_macro_f1": result["test"]["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data", default="data/visd4sa")

    p.add_argument(
        "--overall-source",
        choices=["majority", "visfd"],
        default="majority",
    )

    p.add_argument("--stars-csv")
    p.add_argument("--text-col", default="comment")
    p.add_argument("--star-col", default="n_star")
    p.add_argument("--output", default="outputs/baseline")

    p.add_argument(
        "--label-strategy",
        choices=["rating_then_majority", "majority_only", "rating_only"],
        default="rating_then_majority",
    )

    p.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
    )

    p.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-overlap", action="store_true")

    main(p.parse_args())
