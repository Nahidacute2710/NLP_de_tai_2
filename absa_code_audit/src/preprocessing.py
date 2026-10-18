import re
import unicodedata


def clean_text(text):
    # Chuẩn hóa Unicode tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Thay nhiều khoảng trắng liên tiếp bằng một khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()