from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
import json
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_OPENAI_CLIENT = None


def _request_completion(
    system_prompt: str, user_prompt: str, max_tokens: int, use_api: bool = False,
) -> str | None:
    global _OPENAI_CLIENT
    if not use_api or not LLM_API_KEY:
        return None
    try:
        if _OPENAI_CLIENT is None:
            from openai import OpenAI
            client_options = {"api_key": LLM_API_KEY}
            if LLM_BASE_URL:
                client_options["base_url"] = LLM_BASE_URL
            _OPENAI_CLIENT = OpenAI(**client_options)
        response = _OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as error:
        print(f"  ⚠️  Enrichment API failed: {error}")
        return None


def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str, use_api: bool = False) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    response = _request_completion(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", text, 150,
        use_api=use_api,
    )
    if response:
        return response
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(
    text: str, n_questions: int = 3, use_api: bool = False,
) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    response = _request_completion(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi có thể được trả lời từ đoạn văn. "
        "Trả về mỗi câu hỏi trên một dòng.", text, 200, use_api=use_api,
    )
    if response:
        questions = [
            line.strip().lstrip("0123456789.-)• ")
            for line in response.splitlines() if line.strip()
        ]
        if questions:
            return questions[:n_questions]
    return [f"{sentence.rstrip('.!?')}?" for sentence in _sentences(text)[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(
    text: str, document_title: str = "", use_api: bool = False,
) -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    response = _request_completion(
        "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về một câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 80, use_api=use_api,
    )
    if response:
        return f"{response}\n\n{text}"
    if document_title:
        context = f"Trích từ tài liệu {document_title}."
    else:
        first_sentence = (_sentences(text) or ["nội dung chính sách"])[0].rstrip(".!?")
        context = f"Ngữ cảnh: đoạn này đề cập đến {first_sentence}."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str, use_api: bool = False) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    response = _request_completion(
        'Trích xuất metadata và chỉ trả về JSON: {"topic":"...", "entities":[], '
        '"category":"policy|hr|it|finance", "language":"vi|en"}.', text, 150,
        use_api=use_api,
    )
    if response:
        try:
            return _parse_json_object(response)
        except (ValueError, json.JSONDecodeError):
            pass
    heading = next((line.strip("# ") for line in text.splitlines() if line.startswith("#")), "")
    return {"topic": heading or "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str, use_api: bool = False) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    system_prompt = (
        "Phân tích đoạn văn và chỉ trả về JSON hợp lệ với các khóa: "
        '"summary" (chuỗi), "questions" (mảng câu hỏi), "context" (một câu), '
        '"metadata" (topic, entities, category, language).'
    )
    response = _request_completion(
        system_prompt, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", 400,
        use_api=use_api,
    )
    if response:
        try:
            result = _parse_json_object(response)
            result.setdefault("summary", "")
            result.setdefault("questions", [])
            result.setdefault("context", "")
            result.setdefault("metadata", {})
            return result
        except (ValueError, json.JSONDecodeError):
            pass
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu {source}." if source else "Nội dung chính sách nội bộ.",
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    use_api: bool = False,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source, use_api=use_api)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text, use_api=use_api) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text, use_api=use_api) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source, use_api=use_api) if "contextual" in methods else text
            auto_meta = extract_metadata(text, use_api=use_api) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
