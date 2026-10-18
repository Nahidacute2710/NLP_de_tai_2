# ABSA tiếng Việt — ML backend tham khảo (Google Colab)

**Phạm vi:** preprocessing, baseline TF-IDF + Logistic Regression, PhoBERT overall sentiment, PhoBERT BIO category+polarity, evaluation và hàm `ABSAInference.predict_review()`. **Không có dashboard.** Source là bản nghiên cứu/tham khảo; môn học yêu cầu nhóm tự thực hiện sản phẩm nộp.

## Quyết định phương pháp

- Dataset chính: UIT-ViSD4SA, giữ nguyên `train/dev/test` của nhà phát hành (dự kiến 7.784/1.113/2.225). Chú ý xác thực lại với file dataset thực tải.
- Weak overall labels: rating 1–2 negative, 3 neutral, 4–5 positive, **chỉ nếu ghép chính xác duy nhất review** với UIT-ViSFD; nếu không có rating, dùng majority aspect polarity (bỏ hòa/không có aspect). Lưu provenance trong JSON metrics. Đây **không phải gold overall labels** và có thể đo agreement với weak labels thay vì cảm xúc thực.
- Baseline: chọn C theo macro-F1 validation. PhoBERT fine-tune riêng classifier overall và BIO tagger.
- Không sử dụng hosted LLM API để dự đoán. Hugging Face chỉ cung cấp checkpoint khởi tạo, Colab là máy chạy.

## Chạy trên Google Colab

Mở `Colab_Runbook.ipynb`, chọn GPU và thực thi lần lượt. Hoặc dùng các lệnh sau **từ thư mục gốc dự án**:

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.download_data --output data/visd4sa
python -m src.audit --data data/visd4sa
python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='models/vncorenlp')"
python -m src.prepare --data data/visd4sa --vncorenlp models/vncorenlp
python -m src.train_baseline --label-strategy majority_only
python -m src.train_phobert --task overall --label-strategy majority_only --fp16
python -m src.train_phobert --task aspect --fp16
python -m src.evaluate_model --checkpoint outputs/phobert/aspect/best --split test
```

Để thử ghép rating, chuẩn bị file CSV `metadata/visfd.csv` gồm `comment,n_star` (cần xác minh đúng cột/nguồn và tỷ lệ match), rồi bỏ `--label-strategy majority_only` và thêm `--stars-csv metadata/visfd.csv` vào **cả baseline và overall PhoBERT**. Giữ cùng một chiến lược nhãn giữa hai mô hình mới có thể so sánh công bằng. Không âm thầm ép tỷ lệ match thấp thành rating-based training.

Nếu bước audit báo cross-split duplicates, **dừng để kiểm tra**, không dùng `--allow-overlap` để bỏ qua mà không có lý do. Nếu prepare báo không khớp offsets, kiểm tra annotation/word segmentation và tỷ lệ dữ liệu bị loại. Không sử dụng test để chọn tham số.

## Giao diện cho nhóm dashboard

```python
from src.inference import ABSAInference
predictor = ABSAInference(model_root='outputs/phobert', vncorenlp='models/vncorenlp')
result = predictor.predict_review('Máy chạy nhanh nhưng pin yếu')
```

Kết quả có `text`, `overall_sentiment`, `overall_confidence`, `aspects` (mỗi aspect có `start`, `end`, `text`, `category`, `polarity`, `confidence`) và `truncated`. `start/end` là Python half-open character offsets theo review gốc; có thể highlight bằng `text[start:end]`.

## Files quan trọng

- `src/data.py`, `src/overall.py`, `src/audit.py`: load gold, weak label provenance, duplicate checks.
- `src/segmentation.py`, `src/prepare.py`, `src/encoding.py`: VnCoreNLP, alignment, BIO labels.
- `src/train_baseline.py`, `src/train_phobert.py`: huấn luyện, lưu checkpoint/metrics.
- `src/decode.py`, `src/evaluate_model.py`, `src/evaluate_aspects.py`: giải mã và exact-span / pair F1.
- `src/inference.py`: inference contract.
- `tests/`: kiểm tra logic đơn vị.

## Giới hạn chưa được kiểm chứng

- Chưa huấn luyện PhoBERT thực tế; **chưa có scores, confusion matrix thực nghiệm hoặc ablation/OOD results**.
- Dữ liệu UIT-ViSD4SA và VnCoreNLP cần tải trên Colab rồi xác nhận schema annotation, offset conventions và coverage. `parse_span` hiện giả định **end inclusive** dựa trên định dạng mã gốc; nếu dữ liệu khác, phải xác minh và sửa trước train.
- Dự đoán BIO dùng word/subword truncation theo `max_length=256`, kết quả có thể bỏ sót spans trong review dài. Kiểm tra tỉ lệ truncation.
- Cần thêm log kết quả thí nghiệm, error analysis phân nhóm và ablation trước khi coi đạt toàn bộ rubric; giao diện do thành viên khác triển khai.

## Python 3.13 / lỗi cài dependency trên Colab

- `py-vncorenlp` yêu cầu `pyjnius`. `pyjnius` v1.7.0 bổ sung hỗ trợ Python 3.13; `requirements.txt` ghim `>=1.7,<1.8` và wrapper `py-vncorenlp==0.1.4`.
- Cài thư viện bằng `%pip install -U pip setuptools wheel`, sau đó `%pip install --prefer-binary -r requirements.txt` trong notebook.
- Chạy `python -m pip check` và import `jnius, py_vncorenlp, transformers, sklearn, seqeval` trước khi chạy tests. **Không coi 9 passed là dấu hiệu pip đã cài thành công**.
- Nếu vẫn lỗi `metadata-generation-failed`, gửi log đầy đủ (đặc biệt tên package ở phía trên dòng lỗi). Chưa có kiểm chứng cài dependencies end-to-end trên runtime Colab T4 Python 3.13.
