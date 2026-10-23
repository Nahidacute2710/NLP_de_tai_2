
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# File lưu các đánh giá đã được phân tích
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "dashboard_data" / "reviews.json"


def load_reviews():
    # Nếu chưa có dữ liệu thì trả về danh sách rỗng
    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_review(product, result):
    # Tạo một bản ghi từ kết quả dự đoán PhoBERT
    review = {
        "id": str(uuid4()),
        "product": product.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    reviews = load_reviews()
    reviews.append(review)

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Ghi file tạm rồi thay thế để tránh file JSON bị ghi dở
    temp_file = DATA_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    temp_file.replace(DATA_FILE)

    return review
