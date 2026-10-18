"""VnCoreNLP word-segmentation with raw-text character alignment.
Rejects ambiguous alignments instead of manufacturing BIO annotations.
"""
import re
import unicodedata
from pathlib import Path

TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}

def match_vncorenlp_emoji(token, original):
    # Kiểm tra ký tự emoji bị thay đổi khi đi qua VnCoreNLP
    if len(token) != len(original):
        return False

    for a, b in zip(token, original):
        if a == b:
            continue

        # Chỉ chấp nhận khi ký tự gốc khôi phục được ký tự VnCoreNLP
        if not (
            0xF000 <= ord(a) <= 0xFFFF
            and ord(b) > 0xFFFF
            and ord(b) - 0x10000 == ord(a)
        ):
            return False

    return True

def tone_signature(text):
    # So sánh từ mà không phụ thuộc vị trí đặt dấu thanh
    text = unicodedata.normalize("NFD", text.replace("_", " "))

    parts = []
    for syllable in text.split():
        chars = unicodedata.normalize("NFD", syllable)
        base = "".join(c for c in chars if c not in TONE_MARKS)
        tones = tuple(sorted(c for c in chars if c in TONE_MARKS))
        parts.append((base, tones))

    return parts

def split_token_boundaries(words, text):
    result = []

    for token, start, end in words:
        # Chỉ xử lý các token không chứa khoảng trắng
        if any(ch.isspace() for ch in text[start:end]):
            result.append((token, start, end))
            continue

        raw = text[start:end]

        # Tách dấu câu dính giữa hai chữ và chữ thường dính chữ hoa
        cuts = {0, len(raw)}

        for i in range(1, len(raw)):
            prev, curr = raw[i - 1], raw[i]

            if prev.islower() and curr.isupper():
                cuts.add(i)

            if prev.isalpha() and curr in ".!?":
                cuts.add(i)

            if prev in ".!?" and curr.isalpha():
                cuts.add(i)

        cuts = sorted(cuts)

        for a, b in zip(cuts, cuts[1:]):
            piece = raw[a:b]
            if piece:
                result.append((piece, start + a, start + b))

    return result

class Segmenter:
    def __init__(self, model_dir):
        import py_vncorenlp

        # Dùng đường dẫn tuyệt đối để Java tìm đúng file JAR
        model_dir = Path(model_dir).resolve()

        self.engine = py_vncorenlp.VnCoreNLP(
            annotators=['wseg'],
            save_dir=str(model_dir)
        )
        
    def aligned_words(self, text):
        sentences = self.engine.word_segment(text)
        tokens = [w for sentence in sentences for w in sentence.split()]
        words = []
        cursor = 0

        for token in tokens:
            # Dấu "_" đứng riêng là ký tự thật
            if token == "_":
                pieces = ["_"]
            else:
                pieces = token.split("_")

            regex = r'\s+'.join(re.escape(p) for p in pieces)

            # Ưu tiên tìm từ khớp chính xác với văn bản gốc
            found = re.search(regex, text[cursor:])

            if found is not None:
                start = cursor + found.start()
                end = cursor + found.end()
            else:
                # Nếu khác cách đặt dấu hoặc biểu diễn emoji
                pattern = r'(?=(\S+' + r'\s+\S+' * (len(pieces) - 1) + r'))'
                start = end = None

                for candidate in re.finditer(pattern, text[cursor:]):
                    candidate_text = candidate.group(1)

                    if (
                        tone_signature(candidate_text) == tone_signature(token)
                        or match_vncorenlp_emoji(token, candidate_text)
                    ):
                        start = cursor + candidate.start(1)
                        end = cursor + candidate.end(1)
                        break

                if start is None:
                    raise ValueError(
                        f'Segmentation offset alignment failed: {token!r} in {text!r}'
                    )

            # Không được bỏ sót nội dung giữa hai token
            if text[cursor:start].strip():
                raise ValueError(
                    f"Unaligned text: {text[cursor:start]!r}"
                )

            words.append((token, start, end))
            cursor = end

        # Kiểm tra phần văn bản còn lại sau token cuối
        if text[cursor:].strip():
            raise ValueError(
                f"Unaligned text at end: {text[cursor:]!r}"
            )

        return words


def word_labels(words, spans, text):
    cleaned_spans = []

    for span in spans:
        start, end = span["start"], span["end"]

        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        if start == end:
            raise ValueError("Empty aspect after trimming")

        cleaned_spans.append({
            **span,
            "start": start,
            "end": end
        })

    spans = cleaned_spans
    tags = []

    for _, start, end in words:
        overlaps = [
            s for s in spans
            if start < s["end"] and end > s["start"]
        ]

        if len(overlaps) > 1:
            raise ValueError("Segmenter word crosses multiple aspects")

        if not overlaps:
            tags.append("O")
            continue

        span = overlaps[0]

        if start < span["start"] or end > span["end"]:
            raise ValueError("Word segmentation crosses annotated aspect boundary")

        prefix = "B" if start == span["start"] else "I"

        tags.append(
            f"{prefix}-{span['category']}#{span['polarity'].upper()}"
        )

    for span in spans:
        selected = [
            (start, end)
            for _, start, end in words
            if start >= span["start"] and end <= span["end"]
        ]

        if not selected:
            raise ValueError(f"Span not covered: {span}")

        if selected[0][0] != span["start"] or selected[-1][1] != span["end"]:
            raise ValueError(f"Span not exactly covered: {span}")

        for (_, prev_end), (next_start, _) in zip(
            selected, selected[1:]
        ):
            if text[prev_end:next_start].strip():
                raise ValueError(f"Missing characters inside aspect: {span}")

    return tags

def build_bio_words(words, spans, text):
    # Chuẩn hóa ranh giới aspect, không thay đổi text gốc
    aspects = []

    for span in spans:
        start, end = span["start"], span["end"]

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if start == end:
            raise ValueError("Aspect rỗng sau khi bỏ khoảng trắng")

        aspects.append({
            **span,
            "start": start,
            "end": end,
        })

    aspects.sort(key=lambda s: (s["start"], s["end"]))

    # BIO một tầng không biểu diễn được aspect chồng lấn
    for prev, curr in zip(aspects, aspects[1:]):
        if prev["end"] > curr["start"]:
            raise ValueError("Overlapping aspect annotations")

    # Chia token tại ranh giới aspect nếu cần
    boundaries = set()
    for span in aspects:
        boundaries.add(span["start"])
        boundaries.add(span["end"])

    new_words = []

    for token, start, end in words:
        cuts = [start, end]
        cuts += [p for p in boundaries if start < p < end]
        cuts = sorted(set(cuts))

        for a, b in zip(cuts, cuts[1:]):
            # Bỏ đoạn chỉ chứa khoảng trắng
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1

            if a >= b:
                continue

            # Giữ lại vị trí trong text gốc
            # Dùng "_" giữa các âm tiết cho PhoBERT
            piece = "_".join(text[a:b].split())
            new_words.append((piece, a, b))

    tags = []

    for token, start, end in new_words:
        matches = [
            span for span in aspects
            if start < span["end"] and end > span["start"]
        ]

        if not matches:
            tags.append("O")
            continue

        if len(matches) != 1:
            raise ValueError("Token chồng lên nhiều aspect")

        span = matches[0]

        if start < span["start"] or end > span["end"]:
            raise ValueError("Không thể căn chỉnh ranh giới aspect")

        prefix = "B" if start == span["start"] else "I"

        tags.append(
            f"{prefix}-{span['category']}#{span['polarity'].upper()}"
        )

    # Kiểm tra mỗi aspect đều được các token bao phủ chính xác
    for span in aspects:
        selected = [
            (a, b)
            for _, a, b in new_words
            if a >= span["start"] and b <= span["end"]
        ]

        if not selected:
            raise ValueError("Aspect không có token tương ứng")

        if selected[0][0] != span["start"]:
            raise ValueError("Aspect bắt đầu không đúng vị trí")

        if selected[-1][1] != span["end"]:
            raise ValueError("Aspect kết thúc không đúng vị trí")

        for (_, prev_end), (next_start, _) in zip(
            selected, selected[1:]
        ):
            if text[prev_end:next_start].strip():
                raise ValueError("Aspect có ký tự bị bỏ sót")

    return new_words, tags, aspects
