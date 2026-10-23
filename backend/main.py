from collections import Counter, defaultdict
from datetime import date, datetime
from fastapi import Query
from backend.storage import load_reviews
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import ABSAInference
from backend.storage import save_review

# Lấy đường dẫn gốc của project để tìm model
PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Vietnamese ABSA API",
    version="1.0.0",
    description="Overall and Aspect-Based Sentiment Analysis using PhoBERT",
)

# Cho phép frontend React gọi API từ localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Dữ liệu đầu vào để phân tích một review
class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


# Dữ liệu đầu vào để phân tích và lưu review
class SaveReviewRequest(BaseModel):
    product: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=10000)



# Load 2 model PhoBERT một lần, không load lại mỗi request
@lru_cache(maxsize=1)
def get_model():
    return ABSAInference(
        model_root=str(PROJECT_ROOT / "models/phobert"),
        vncorenlp=str(PROJECT_ROOT / "models/vncorenlp"),
        device="cpu",
    )


# Kiểm tra backend có hoạt động không
@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


# Nhận review, chạy PhoBERT và trả kết quả phân tích
@app.post("/api/v1/analyze")
def analyze(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Review text must not be empty",
        )

    try:
        return get_model().predict_review(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

# Phân tích và lưu review để thống kê trên dashboard
@app.post("/api/v1/reviews")
def create_review(request: SaveReviewRequest):
    if not request.product.strip() or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Product and review text must not be empty",
        )

    try:
        result = get_model().predict_review(request.text)
        return save_review(request.product, result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    

# Lấy danh sách review, có thể lọc theo sản phẩm và ngày ghi nhận
@app.get("/api/v1/reviews")
def get_reviews(
    product: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
):
    if from_date and to_date and from_date > to_date:
        raise HTTPException(
            status_code=422,
            detail="from_date must not be later than to_date",
        )

    reviews = load_reviews()

    if product:
        reviews = [
            r for r in reviews
            if r["product"].casefold() == product.strip().casefold()
        ]

    if from_date:
        reviews = [
            r for r in reviews
            if datetime.fromisoformat(r["created_at"]).date() >= from_date
        ]

    if to_date:
        reviews = [
            r for r in reviews
            if datetime.fromisoformat(r["created_at"]).date() <= to_date
        ]

    return {
        "total": len(reviews),
        "reviews": reviews,
    }

# Thống kê cảm xúc và các khía cạnh từ review đã lưu
@app.get("/api/v1/statistics")
def get_statistics(
    product: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
):
    # Dùng lại bộ lọc của API danh sách review
    filtered = get_reviews(product, from_date, to_date)["reviews"]

    overall_counts = Counter()
    aspect_counts = defaultdict(lambda: {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    })

    for review in filtered:
        overall_counts[review["overall_sentiment"].lower()] += 1

        for aspect in review["aspects"]:
            category = aspect["category"]
            polarity = aspect["polarity"].lower()
            if polarity in aspect_counts[category]:
                aspect_counts[category][polarity] += 1

    aspects = [
        {"category": category, **counts}
        for category, counts in sorted(aspect_counts.items())
    ]

    return {
        "total_reviews": len(filtered),
        "overall_distribution": {
            key: overall_counts[key]
            for key in ("positive", "neutral", "negative")
        },
        "aspect_statistics": aspects,
    }