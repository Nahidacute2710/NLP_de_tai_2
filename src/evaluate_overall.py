
"""Đánh giá PhoBERT Overall trên tập test."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_jsonl
from .segmentation import Segmenter, split_token_boundaries


def main():
    checkpoint = Path("models/phobert/overall/best")
    test_path = Path("data/overall_visfd/test.jsonl")
    # Lấy đường dẫn project dựa trên vị trí file code
    project_root = Path(__file__).resolve().parent.parent

# Tạo thư mục lưu kết quả đánh giá
    output_dir = project_root / "outputs" / "evaluation" / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Đang đọc dữ liệu test...")
    rows = load_jsonl(test_path)

    # Load model đã train, không cần train lại
    print("Đang tải PhoBERT Overall...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    # Lấy thứ tự nhãn từ model để không bị nhầm khi đánh giá
    labels = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]
    label2id = model.config.label2id

    print("Đang tải VnCoreNLP...")
    segmenter = Segmenter("models/vncorenlp")

    def preprocess(text):
        # Tách từ giống lúc huấn luyện
        try:
            words = split_token_boundaries(
                segmenter.aligned_words(text), text
            )
            return " ".join(w for w, _, _ in words)
        except ValueError:
            return text

    print(f"Đang xử lý {len(rows)} câu test...")
    texts = [preprocess(row["text"]) for row in rows]
    y_true = [label2id[row["overall"]] for row in rows]

    y_pred = []
    batch_size = 8

    # Dự đoán theo từng batch để tránh tốn quá nhiều RAM
    print("Đang chạy dự đoán...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = model(**encoded).logits

        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(predictions.cpu().tolist())

        print(
            f"\rĐã xử lý {min(start + batch_size, len(texts))}"
            f"/{len(texts)} câu",
            end="",
            flush=True,
        )

    print()

    # Tính các chỉ số trên tập test
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    metrics = {
        "test_samples": len(rows),
        "labels": labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }

    # Kiểm tra lại thư mục trước khi lưu
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lưu kết quả để dùng cho báo cáo
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Lưu từng câu để sau này phân tích lỗi dự đoán
    with (output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row, gold, pred in zip(rows, y_true, y_pred):
            record = {
                "text": row["text"],
                "gold": labels[gold],
                "predicted": labels[pred],
                "correct": gold == pred,
            }
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Vẽ Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Confusion Matrix - PhoBERT Overall")

    threshold = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # So sánh với kết quả đã lưu từ lần huấn luyện
    original_path = Path("models/phobert/overall/metrics.json")

    if original_path.exists():
        original = json.loads(
            original_path.read_text(encoding="utf-8")
        )["test"]

        print("\nSo sánh kết quả cũ và kết quả mới:")
        print(
            "Accuracy:",
            original["test_accuracy"], "vs", accuracy
        )
        print(
            "Macro-F1:",
            original["test_macro_f1"], "vs", macro_f1
        )

        if (
            not np.isclose(
                accuracy, original["test_accuracy"], atol=1e-8
            )
            or not np.isclose(
                macro_f1, original["test_macro_f1"], atol=1e-8
            )
        ):
            print("CẢNH BÁO: Kết quả chưa khớp, cần kiểm tra lại.")
        else:
            print("OK: Kết quả khớp với lần đánh giá ban đầu.")


if __name__ == "__main__":
    main()
