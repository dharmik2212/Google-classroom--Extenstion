from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, END
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract, requests, os
import logging
import time
import threading
from collections import OrderedDict
from io import BytesIO
import re
import json
from dotenv import load_dotenv
 # pragma: no cover
load_dotenv() 

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover
    InferenceClient = None

try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except Exception:  # pragma: no cover
    GatedRepoError = None
    HfHubHTTPError = None
    RepositoryNotFoundError = None

logger = logging.getLogger("classroom_bot")

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load .env early so env-overridable settings below are applied.
if load_dotenv is not None:
    # Use explicit path to avoid cwd-dependent behavior.
    load_dotenv(DOTENV_PATH)

DEBUG = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}

# Latency controls (env-overridable)
HF_REPO_ID = os.getenv("HF_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "0"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0"))
HF_DO_SAMPLE = os.getenv("HF_DO_SAMPLE", "").lower() in {"1", "true", "yes"}
HF_STOP_SEQUENCES: list[str] = []

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
MAX_URL_TEXT_CHARS = int(os.getenv("MAX_URL_TEXT_CHARS", "1400"))
MAX_URLS = int(os.getenv("MAX_URLS", "3"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "10"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "3"))
PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "140"))

DIAGRAM_OCR_DPI = int(os.getenv("DIAGRAM_OCR_DPI", "160"))
DIAGRAM_OCR_MAX_PAGES = int(os.getenv("DIAGRAM_OCR_MAX_PAGES", "2"))
DIAGRAM_OCR_MAX_CHARS = int(os.getenv("DIAGRAM_OCR_MAX_CHARS", "4000"))

MAX_DOC_TEXT_CHARS = int(os.getenv("MAX_DOC_TEXT_CHARS", "60000"))
DRIVE_FETCH_CONNECT_TIMEOUT = float(os.getenv("DRIVE_FETCH_CONNECT_TIMEOUT", "3"))
DRIVE_FETCH_READ_TIMEOUT = float(os.getenv("DRIVE_FETCH_READ_TIMEOUT", "12"))
DRIVE_FETCH_TIMEOUT = (DRIVE_FETCH_CONNECT_TIMEOUT, DRIVE_FETCH_READ_TIMEOUT)
MAX_DRIVE_BYTES = int(os.getenv("MAX_DRIVE_BYTES", "12000000"))

DRIVE_DOC_CACHE_TTL_SECONDS = int(os.getenv("DRIVE_DOC_CACHE_TTL_SECONDS", "900"))
DRIVE_DOC_CACHE_MAX = int(os.getenv("DRIVE_DOC_CACHE_MAX", "32"))

URL_FETCH_CONNECT_TIMEOUT = float(os.getenv("URL_FETCH_CONNECT_TIMEOUT", "3"))
URL_FETCH_READ_TIMEOUT = float(os.getenv("URL_FETCH_READ_TIMEOUT", "8"))
URL_FETCH_TIMEOUT = (URL_FETCH_CONNECT_TIMEOUT, URL_FETCH_READ_TIMEOUT)

URL_TEXT_CACHE_TTL_SECONDS = int(os.getenv("URL_TEXT_CACHE_TTL_SECONDS", "600"))
URL_TEXT_CACHE_MAX = int(os.getenv("URL_TEXT_CACHE_MAX", "128"))

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACE_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
)


def _safe_hf_error_hint(err: Exception) -> str:
    """Return a safe, user-facing hint without exposing secrets."""
    model_id = HF_REPO_ID

    response = getattr(err, "response", None)
    status = getattr(response, "status_code", None)
    headers = getattr(response, "headers", {}) or {}
    error_code = headers.get("X-Error-Code")
    request_id = getattr(err, "request_id", None)
    rid = f" (request_id={request_id})" if request_id else ""

    # Gated models commonly return 403 with X-Error-Code: GatedRepo.
    if (GatedRepoError is not None and isinstance(err, GatedRepoError)) or (
        status in (401, 403) and error_code == "GatedRepo"
    ):
        return (
            f"Access to this model is gated (HTTP {status}): model={model_id}. "
            f"Accept/request access on the model page and retry.{rid}"
        )

    if status == 401:
        return (
            f"Authentication failed (HTTP 401): model={model_id}. "
            f"Check your Hugging Face token and model access.{rid}"
        )

    if status == 403:
        return (
            f"Forbidden (HTTP 403): model={model_id}. "
            f"Token may lack permissions or required terms not accepted.{rid}"
        )

    if status == 404:
        return (
            f"Model not found (HTTP 404): model={model_id}. "
            f"Check HF_REPO_ID spelling / availability.{rid}"
        )

    if status:
        return f"Hugging Face request failed (HTTP {status}): model={model_id}.{rid}"

    # Fallback: keep it short and avoid dumping repr(err).
    return f"Hugging Face request failed: model={model_id}."

class QAState(TypedDict):
    question: str
    context: str
    images: NotRequired[list[str]]
    pageUrl: NotRequired[str | None]
    docRef: NotRequired[dict]
    accessToken: NotRequired[str | None]
    answer: NotRequired[str]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "i", "you", "we", "they", "he", "she", "them", "his", "her", "our", "your", "their", "my", "me",
    "do", "does", "did", "yes", "can", "could", "should", "would", "will", "just",
}


def select_relevant_snippets(
    text: str,
    question: str,
    *,
    max_chars: int,
    chunk_size: int,
    overlap: int,
) -> str:
    text = text or ""
    question = question or ""

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    q_words = {
        w
        for w in re.findall(r"[a-z0-9]+", question.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }
    if not q_words:
        return text[:max_chars]

    step = max(1, chunk_size - overlap)
    candidates: list[tuple[int, int, str]] = []

    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            words = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            score = len(words & q_words)
            candidates.append((score, start, chunk))
        if end >= len(text):
            break

    ranked = sorted(candidates, key=lambda t: (-t[0], t[1]))

    picked: list[tuple[int, str]] = []
    total = 0
    sep = "\n\n---\n\n"

    for score, start, chunk in ranked:
        if score <= 0:
            break
        add_len = len(chunk) + (len(sep) if picked else 0)
        if total + add_len > max_chars:
            continue
        picked.append((start, chunk))
        total += add_len
        if total >= max_chars:
            break

    if not picked:
        return text[:max_chars]

    picked.sort(key=lambda t: t[0])
    out = sep.join(c for _, c in picked)
    return out[:max_chars]


def _looks_like_visual_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(
        r"\b(diagram|figure|fig\.?|image|picture|chart|graph|plot|flowchart|block\s*diagram|circuit|schematic|axis|legend|caption)\b",
        q,
    ):
        return True

    # Common implicit references to a visual without naming it.
    if re.search(r"\b(explain|describe|interpret|what\s+does\s+this\s+mean|what\s+is\s+shown)\b", q) and re.search(
        r"\b(this|above|below|here|shown)\b", q
    ):
        return True

    return False


def _extract_page_hint_from_url(page_url: str | None) -> int | None:
    if not page_url:
        return None
    m = re.search(r"(?:[#?&]|^)page=(\d{1,4})\b", page_url)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return n if n >= 1 else None

def _looks_like_pdf(url: str, content_type: str | None) -> bool:
    if content_type and "application/pdf" in content_type.lower():
        return True
    url_lower = (url or "").lower()
    if ".pdf" in url_lower:
        return True
    return False


def _fetch_url_bytes(url: str) -> tuple[bytes, str | None]:
    r = requests.get(url, timeout=URL_FETCH_TIMEOUT)
    r.raise_for_status()
    return r.content, r.headers.get("content-type")


def _ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(img)


def _extract_pdf_text_with_ocr_fallback(pdf_bytes: bytes, *, char_limit: int) -> str:
    if fitz is None:
        return ""

    if char_limit <= 0:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: list[str] = []
    total_chars = 0
    ocr_pages_used = 0

    for i, page in enumerate(doc):
        if MAX_PDF_PAGES > 0 and i >= MAX_PDF_PAGES:
            break
        if total_chars >= char_limit:
            break

        page_text = ""
        try:
            page_text = (page.get_text("text") or "").strip()
        except Exception:
            page_text = ""

        if page_text:
            remaining = char_limit - total_chars
            if remaining <= 0:
                break
            chunk = page_text[:remaining]
            if len(chunk) >= 20:
                parts.append(chunk)
                total_chars += len(chunk)
                continue

        # OCR fallback only when there's little/no embedded text.
        if MAX_OCR_PAGES > 0 and ocr_pages_used >= MAX_OCR_PAGES:
            continue

        try:
            pix = page.get_pixmap(dpi=PDF_OCR_DPI, alpha=False)
            png_bytes = pix.tobytes("png")
            ocr_text = (_ocr_image_bytes(png_bytes) or "").strip()
            ocr_pages_used += 1
            if ocr_text:
                remaining = char_limit - total_chars
                if remaining <= 0:
                    break
                chunk = ocr_text[:remaining]
                if chunk:
                    parts.append(chunk)
                    total_chars += len(chunk)
        except Exception:
            continue

    return "\n\n".join(p for p in parts if p)


def _extract_pdf_diagram_ocr(
    pdf_bytes: bytes,
    *,
    question: str,
    page_hint_1based: int | None,
) -> str:
    """OCR a small set of likely-relevant pages to capture labels inside diagrams."""
    if fitz is None:
        return ""
    if not pdf_bytes:
        return ""
    if DIAGRAM_OCR_MAX_PAGES == 0:
        return ""

    max_pages = DIAGRAM_OCR_MAX_PAGES if DIAGRAM_OCR_MAX_PAGES > 0 else 2

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""

    # Tokenize question for page scoring.
    q_words = {
        w
        for w in re.findall(r"[a-z0-9]+", (question or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }

    candidate: list[int] = []

    if page_hint_1based is not None:
        idx = page_hint_1based - 1
        if 0 <= idx < doc.page_count:
            candidate.append(idx)
            # include neighbor page when possible
            if idx + 1 < doc.page_count:
                candidate.append(idx + 1)

    # Score pages using embedded text (fast). Even if diagrams are images, captions often exist.
    scored: list[tuple[int, int]] = []
    try:
        scan_pages = min(doc.page_count, 40)
        for i in range(scan_pages):
            try:
                t = (doc.load_page(i).get_text("text") or "")
            except Exception:
                t = ""
            if not t:
                scored.append((0, i))
                continue
            if not q_words:
                scored.append((0, i))
                continue
            words = set(re.findall(r"[a-z0-9]+", t.lower()))
            scored.append((len(words & q_words), i))
    except Exception:
        scored = []

    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        for score, i in scored[: max_pages * 3]:
            if score <= 0 and candidate:
                break
            candidate.append(i)

    # Dedup while preserving order
    seen = set()
    pages: list[int] = []
    for i in candidate:
        if i in seen:
            continue
        seen.add(i)
        pages.append(i)
        if len(pages) >= max_pages:
            break

    if not pages:
        pages = [0] if doc.page_count else []

    parts: list[str] = []
    total = 0
    for i in pages:
        if total >= DIAGRAM_OCR_MAX_CHARS:
            break
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=DIAGRAM_OCR_DPI, alpha=False)
            png_bytes = pix.tobytes("png")
            ocr = (_ocr_image_bytes(png_bytes) or "").strip()
        except Exception:
            continue

        if not ocr:
            continue
        chunk = f"Page {i + 1} OCR:\n{ocr}".strip()
        remaining = DIAGRAM_OCR_MAX_CHARS - total
        if remaining <= 0:
            break
        chunk = chunk[:remaining]
        parts.append(chunk)
        total += len(chunk) + 2

    return "\n\n".join(parts).strip()


def _bearer_token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    s = authorization.strip()
    if not s:
        return None
    if s.lower().startswith("bearer "):
        tok = s[7:].strip()
        return tok if tok else None
    return None


_drive_doc_cache_lock = threading.Lock()
_drive_doc_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()


def _cache_get_drive_doc(key: str) -> str | None:
    if DRIVE_DOC_CACHE_TTL_SECONDS <= 0 or DRIVE_DOC_CACHE_MAX <= 0:
        return None
    now = time.monotonic()
    with _drive_doc_cache_lock:
        item = _drive_doc_cache.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at <= now:
            _drive_doc_cache.pop(key, None)
            return None
        _drive_doc_cache.move_to_end(key)
        return value


def _cache_set_drive_doc(key: str, text: str) -> None:
    if DRIVE_DOC_CACHE_TTL_SECONDS <= 0 or DRIVE_DOC_CACHE_MAX <= 0:
        return
    now = time.monotonic()
    expires_at = now + DRIVE_DOC_CACHE_TTL_SECONDS
    with _drive_doc_cache_lock:
        _drive_doc_cache.pop(key, None)
        _drive_doc_cache[key] = (expires_at, text)

        expired_keys = [k for k, (exp, _) in _drive_doc_cache.items() if exp <= now]
        for k in expired_keys:
            _drive_doc_cache.pop(k, None)

        while len(_drive_doc_cache) > DRIVE_DOC_CACHE_MAX:
            _drive_doc_cache.popitem(last=False)


def _drive_api_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


def _drive_get_file_metadata(file_id: str, access_token: str) -> dict | None:
    if not file_id or not access_token:
        return None
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {
        "fields": "id,name,mimeType,modifiedTime,md5Checksum,size",
        "supportsAllDrives": "true",
    }
    r = requests.get(url, headers=_drive_api_headers(access_token), params=params, timeout=DRIVE_FETCH_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


def _read_stream_limited(r: requests.Response, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return r.content

    buf = bytearray()
    for chunk in r.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise RuntimeError(f"Drive download too large (>{max_bytes} bytes)")
    return bytes(buf)


def _drive_download_file_bytes(file_id: str, access_token: str) -> tuple[bytes, str | None]:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {"alt": "media", "supportsAllDrives": "true"}
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=DRIVE_FETCH_TIMEOUT,
        stream=True,
    )
    r.raise_for_status()
    content_type = r.headers.get("content-type")
    data = _read_stream_limited(r, max_bytes=MAX_DRIVE_BYTES)
    return data, content_type


def _drive_export_file_bytes(file_id: str, access_token: str, *, export_mime: str) -> tuple[bytes, str | None]:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
    params = {"mimeType": export_mime}
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=DRIVE_FETCH_TIMEOUT,
        stream=True,
    )
    r.raise_for_status()
    content_type = r.headers.get("content-type")
    data = _read_stream_limited(r, max_bytes=MAX_DRIVE_BYTES)
    return data, content_type


def _drive_get_pdf_bytes_for_visual(file_id: str, access_token: str) -> bytes:
    """Return PDF bytes suitable for page-render OCR (PDF or Google file exported to PDF)."""
    meta = _drive_get_file_metadata(file_id, access_token)
    if not meta:
        return b""
    mime = (meta.get("mimeType") or "").strip()

    is_google = mime.startswith("application/vnd.google-apps.")
    if is_google:
        # Export most Google formats to PDF for rendering.
        data, _ct = _drive_export_file_bytes(file_id, access_token, export_mime="application/pdf")
        return data

    # Non-google: only PDFs are renderable here.
    data, ct = _drive_download_file_bytes(file_id, access_token)
    ct_lower = (ct or "").lower()
    if "application/pdf" in ct_lower or mime.lower() == "application/pdf":
        return data
    return b""


def _extract_text_from_drive_file(*, file_id: str, access_token: str, question: str) -> str:
    if not file_id or not access_token:
        return ""

    meta = _drive_get_file_metadata(file_id, access_token)
    if not meta:
        return ""

    mime = (meta.get("mimeType") or "").strip()
    modified = (meta.get("modifiedTime") or "").strip()
    md5 = (meta.get("md5Checksum") or "").strip()
    size_str = (meta.get("size") or "").strip()
    try:
        size = int(size_str) if size_str else 0
    except Exception:
        size = 0

    # Cache key should change when file changes.
    cache_key = f"{file_id}|{modified}|{md5}|{mime}"
    cached = _cache_get_drive_doc(cache_key)
    if cached is not None:
        return cached

    if MAX_DRIVE_BYTES > 0 and size and size > MAX_DRIVE_BYTES:
        raise RuntimeError(f"Drive file too large ({size} bytes)")

    text = ""

    is_google = mime.startswith("application/vnd.google-apps.")
    if is_google:
        # Export based on type.
        export_mime = None
        if mime == "application/vnd.google-apps.document":
            export_mime = "text/plain"
        elif mime == "application/vnd.google-apps.spreadsheet":
            export_mime = "text/csv"
        else:
            # Slides/drawings/etc: export to PDF then parse.
            export_mime = "application/pdf"

        data, ct = _drive_export_file_bytes(file_id, access_token, export_mime=export_mime)

        if (ct or "").lower().startswith("application/pdf") or export_mime == "application/pdf":
            text = _extract_pdf_text_with_ocr_fallback(data, char_limit=MAX_DOC_TEXT_CHARS)
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
    else:
        data, ct = _drive_download_file_bytes(file_id, access_token)
        ct_lower = (ct or "").lower()
        if "application/pdf" in ct_lower or (mime or "").lower() == "application/pdf":
            text = _extract_pdf_text_with_ocr_fallback(data, char_limit=MAX_DOC_TEXT_CHARS)
        elif ct_lower.startswith("text/"):
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        else:
            # Attempt OCR for image-like content.
            if ct_lower.startswith("image/"):
                try:
                    text = _ocr_image_bytes(data)
                except Exception:
                    text = ""

    text = (text or "").strip()
    if text:
        # Reduce stored size to keep cache bounded.
        text = text[:MAX_DOC_TEXT_CHARS]
        _cache_set_drive_doc(cache_key, text)
    return text


_url_text_cache_lock = threading.Lock()
_url_text_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()


def _cache_get_url_text(url: str) -> str | None:
    if URL_TEXT_CACHE_TTL_SECONDS <= 0 or URL_TEXT_CACHE_MAX <= 0:
        return None

    now = time.monotonic()
    with _url_text_cache_lock:
        item = _url_text_cache.get(url)
        if not item:
            return None
        expires_at, value = item
        if expires_at <= now:
            _url_text_cache.pop(url, None)
            return None
        _url_text_cache.move_to_end(url)
        return value


def _cache_set_url_text(url: str, text: str) -> None:
    if URL_TEXT_CACHE_TTL_SECONDS <= 0 or URL_TEXT_CACHE_MAX <= 0:
        return

    now = time.monotonic()
    expires_at = now + URL_TEXT_CACHE_TTL_SECONDS
    with _url_text_cache_lock:
        _url_text_cache.pop(url, None)
        _url_text_cache[url] = (expires_at, text)

        # prune expired first
        expired_keys = [k for k, (exp, _) in _url_text_cache.items() if exp <= now]
        for k in expired_keys:
            _url_text_cache.pop(k, None)

        # prune over capacity (LRU)
        while len(_url_text_cache) > URL_TEXT_CACHE_MAX:
            _url_text_cache.popitem(last=False)


def extract_text_from_urls(urls: list[str], *, char_limit: int) -> str:
    if char_limit <= 0:
        return ""

    text_parts: list[str] = []
    total_chars = 0

    for url in (urls or [])[:MAX_URLS]:
        if total_chars >= char_limit:
            break
        if not url:
            continue

        cached = _cache_get_url_text(url)
        if cached is not None:
            if cached.strip():
                remaining = char_limit - total_chars
                chunk = cached[:remaining]
                text_parts.append(chunk)
                total_chars += len(chunk)
            continue

        try:
            content, content_type = _fetch_url_bytes(url)
        except Exception:
            continue

        if _looks_like_pdf(url, content_type):
            remaining = char_limit - total_chars
            pdf_text = _extract_pdf_text_with_ocr_fallback(content, char_limit=remaining)
            _cache_set_url_text(url, pdf_text)
            if pdf_text.strip():
                chunk = pdf_text[:remaining]
                text_parts.append(chunk)
                total_chars += len(chunk)
            continue

        # Otherwise treat as image.
        try:
            ocr_text = (_ocr_image_bytes(content) or "").strip()
            _cache_set_url_text(url, ocr_text)
            if ocr_text:
                remaining = char_limit - total_chars
                chunk = ocr_text[:remaining]
                text_parts.append(chunk)
                total_chars += len(chunk)
        except Exception:
            continue

    return "\n\n".join(text_parts)


def answer_node(state):
    question = (state.get("question") or "").strip()
    if not question:
        return {"answer": "i dont know due to insufficient information"}

    raw_context = state.get("context") or ""
    page_url = state.get("pageUrl")
    images = state.get("images") or []

    # Drive-backed open document retrieval.
    doc_ref = state.get("docRef") or None
    access_token = state.get("accessToken") or None
    file_id = doc_ref.get("fileId") if isinstance(doc_ref, dict) else None

    doc_text = ""
    doc_diagram_ocr = ""

    if file_id and access_token:
        try:
            doc_text = _extract_text_from_drive_file(file_id=file_id, access_token=access_token, question=question)
        except Exception:
            doc_text = ""

        if _looks_like_visual_question(question):
            try:
                pdf_bytes = _drive_get_pdf_bytes_for_visual(file_id, access_token)
                hint = _extract_page_hint_from_url(page_url)
                doc_diagram_ocr = _extract_pdf_diagram_ocr(pdf_bytes, question=question, page_hint_1based=hint)
            except Exception:
                doc_diagram_ocr = ""

    if doc_text.strip():
        doc_snips = select_relevant_snippets(
            doc_text,
            question,
            max_chars=MAX_CONTEXT_CHARS,
            chunk_size=CHUNK_SIZE_CHARS,
            overlap=CHUNK_OVERLAP_CHARS,
        )
        if doc_snips.strip():
            raw_context = f"Open Document (retrieved):\n{doc_snips}\n\n---\n\n" + raw_context

    extra_image_text = ""
    if doc_diagram_ocr.strip():
        extra_image_text = f"Document Diagram OCR (selected pages):\n{doc_diagram_ocr}".strip()

    prompt, fallback = _build_prompt_and_maybe_fallback(
        question=question,
        context=raw_context,
        images=images,
        extra_image_text=extra_image_text,
    )
    if fallback is not None:
        return {"answer": fallback}
    if not prompt:
        return {"answer": "i dont know due to insufficient information"}

    answer = _hf_generate_with_continuation(prompt)
    if not isinstance(answer, str) or not answer.strip():
        return {"answer": "i dont know due to insufficient information"}
    return {"answer": answer}


def _build_prompt_and_maybe_fallback(
    *,
    question: str,
    context: str,
    images: list[str],
    extra_image_text: str = "",
) -> tuple[str | None, str | None]:
    q = (question or "").strip()
    if not q:
        return None, "i dont know due to insufficient information"

    raw_context = context or ""
    picked_context = select_relevant_snippets(
        raw_context,
        q,
        max_chars=MAX_CONTEXT_CHARS,
        chunk_size=CHUNK_SIZE_CHARS,
        overlap=CHUNK_OVERLAP_CHARS,
    )

    imgs = images or []
    if not picked_context.strip() and not imgs:
        return None, "i dont know due to insufficient information"

    url_text = extract_text_from_urls(imgs, char_limit=MAX_URL_TEXT_CHARS) if imgs else ""
    url_text = (url_text or "")[:MAX_URL_TEXT_CHARS]
    if extra_image_text and extra_image_text.strip():
        # Keep prompt bounded.
        extra = extra_image_text.strip()[:MAX_URL_TEXT_CHARS]
        url_text = (url_text + "\n\n" + extra).strip()[: MAX_URL_TEXT_CHARS * 2]

    if not picked_context.strip() and not url_text.strip():
        return None, "i dont know due to insufficient information"

    image_urls = "\n".join(imgs[:MAX_URLS]) if imgs else ""

    prompt = f"""Answer only from the content below.
Format your response in GitHub-flavored Markdown.
Use fenced code blocks for code.
If the question asks for a comparison/differences, answer using a Markdown table when possible.
If the question asks for a comparison/differences, output ONLY the table (no extra notes before/after).
Do NOT include code blocks unless the user explicitly asks for code.
If the question asks about a diagram/figure/image, include the relevant image as Markdown (e.g., ![caption](url)) when a URL is provided, then explain it based only on the provided Text + Image Text.
Do not output raw HTML.
If the answer cannot be determined from the content below, reply exactly with:
i dont know due to insufficient information

Text:
{picked_context}

Image Text:
{url_text}

Image URLs:
{image_urls}

Question:
{q}
"""

    return prompt, None


def _looks_truncated(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 400:
        return False
    if t == "i dont know due to insufficient information":
        return False
    # Unclosed code fence.
    if t.count("```") % 2 == 1:
        return True
    # Ends with a likely continuation cue.
    if re.search(r"[,:;\-–]$", t):
        return True
    # Ends without sentence-ending punctuation (common for cutoffs).
    if not re.search(r"[.!?\)\]\"']$", t):
        return True
    return False


def _hf_generate_with_continuation(prompt: str) -> str:
    first = _hf_generate_text(prompt)
    if not isinstance(first, str) or not first.strip():
        return first
    if not _looks_truncated(first):
        return first

    tail = first[-1200:]
    cont_prompt = f"""{prompt}

The assistant's previous answer was cut off. Continue exactly where it stopped.
Do not repeat any text. Continue in GitHub-flavored Markdown.

Previous partial answer (tail):
{tail}
"""
    cont = _hf_generate_text(cont_prompt)
    if not isinstance(cont, str) or not cont.strip():
        return first
    return first.rstrip() + "\n" + cont.lstrip()


def _chunk_text_for_sse(text: str, *, max_chars: int = 24):
    if not text:
        return
    if max_chars <= 1:
        for ch in text:
            yield ch
        return

    parts = re.findall(r"\S+\s*", text)
    buf = ""
    for part in parts:
        if len(buf) + len(part) <= max_chars:
            buf += part
            continue

        if buf:
            yield buf
            buf = ""

        while len(part) > max_chars:
            yield part[:max_chars]
            part = part[max_chars:]

        buf = part

    if buf:
        yield buf


def _hf_generate_text(prompt: str) -> str:
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed")
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token missing (set HUGGINGFACEHUB_ACCESS_TOKEN or HF_TOKEN)")

    client = InferenceClient(model=HF_REPO_ID, token=HF_TOKEN)

    max_tokens = HF_MAX_NEW_TOKENS if HF_MAX_NEW_TOKENS > 0 else None

    # Prefer conversational API (some providers reject text-generation).
    try:
        kwargs = {
            "stream": False,
            "temperature": HF_TEMPERATURE,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = client.chat_completion(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = getattr(resp.choices[0].message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        # fall back to text-generation below
        pass

    # Fallback: Non-streaming text-generation (returns a string).
    try:
        try:
            kwargs = {
                "temperature": HF_TEMPERATURE,
                "do_sample": HF_DO_SAMPLE,
                "return_full_text": False,
            }
            if max_tokens is not None:
                kwargs["max_new_tokens"] = max_tokens
            if HF_STOP_SEQUENCES:
                kwargs["stop"] = HF_STOP_SEQUENCES
            return client.text_generation(prompt, **kwargs)
        except TypeError:
            kwargs = {
                "temperature": HF_TEMPERATURE,
                "do_sample": HF_DO_SAMPLE,
                "return_full_text": False,
            }
            if max_tokens is not None:
                kwargs["max_new_tokens"] = max_tokens
            if HF_STOP_SEQUENCES:
                kwargs["stop_sequences"] = HF_STOP_SEQUENCES
            return client.text_generation(prompt, **kwargs)
    except Exception as e:
        # Provide a safe hint for common HF errors (gated/auth).
        if HfHubHTTPError is not None and isinstance(e, HfHubHTTPError):
            raise RuntimeError(_safe_hf_error_hint(e))
        if GatedRepoError is not None and isinstance(e, GatedRepoError):
            raise RuntimeError(_safe_hf_error_hint(e))
        if RepositoryNotFoundError is not None and isinstance(e, RepositoryNotFoundError):
            raise RuntimeError(_safe_hf_error_hint(e))
        raise


def _hf_stream_tokens(prompt: str):
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed")

    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token missing (set HUGGINGFACEHUB_ACCESS_TOKEN or HF_TOKEN)")

    client = InferenceClient(model=HF_REPO_ID, token=HF_TOKEN)

    max_tokens = HF_MAX_NEW_TOKENS if HF_MAX_NEW_TOKENS > 0 else None

    # Prefer conversational streaming.
    try:
        kwargs = {
            "stream": True,
            "temperature": HF_TEMPERATURE,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return client.chat_completion(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
    except Exception:
        # fall back to text-generation streaming below
        pass

    common = {
        "temperature": HF_TEMPERATURE,
        "do_sample": HF_DO_SAMPLE,
        "stream": True,
    }
    if max_tokens is not None:
        common["max_new_tokens"] = max_tokens

    # Fallback: text-generation streaming.
    last_err: Exception | None = None
    attempts = [
        dict(stop=HF_STOP_SEQUENCES, return_full_text=False) if HF_STOP_SEQUENCES else dict(return_full_text=False),
        dict(stop=HF_STOP_SEQUENCES) if HF_STOP_SEQUENCES else dict(),
        dict(stop_sequences=HF_STOP_SEQUENCES, return_full_text=False) if HF_STOP_SEQUENCES else dict(return_full_text=False),
        dict(stop_sequences=HF_STOP_SEQUENCES) if HF_STOP_SEQUENCES else dict(),
        dict(return_full_text=False),
        dict(),
    ]
    for extra in attempts:
        try:
            return client.text_generation(prompt, **common, **extra)
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            if HfHubHTTPError is not None and isinstance(e, HfHubHTTPError):
                raise RuntimeError(_safe_hf_error_hint(e))
            if GatedRepoError is not None and isinstance(e, GatedRepoError):
                raise RuntimeError(_safe_hf_error_hint(e))
            if RepositoryNotFoundError is not None and isinstance(e, RepositoryNotFoundError):
                raise RuntimeError(_safe_hf_error_hint(e))
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("Unable to start HF streaming")


def _extract_stream_text(chunk) -> str:
    """Extract incremental text from HF streaming chunks (chat or text-generation)."""
    if chunk is None:
        return ""

    # text-generation streaming may yield raw strings
    if isinstance(chunk, str):
        return chunk

    # Some clients may yield dicts
    if isinstance(chunk, dict):
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            parts: list[str] = []
            for c in choices:
                if not isinstance(c, dict):
                    continue
                delta = c.get("delta") or {}
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        parts.append(content)
            return "".join(parts)
        # legacy-style
        tok = chunk.get("token")
        if isinstance(tok, dict):
            t = tok.get("text")
            return t if isinstance(t, str) else ""
        t = chunk.get("text")
        return t if isinstance(t, str) else ""

    # Chat completion streaming: choices[].delta.content
    choices = getattr(chunk, "choices", None)
    if choices is not None:
        parts: list[str] = []
        try:
            for c in choices:
                delta = getattr(c, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    parts.append(content)
        except Exception:
            pass
        if parts:
            return "".join(parts)

    # Legacy text-generation streaming: token.text or chunk.text
    token = getattr(chunk, "token", None)
    if token is not None:
        t = getattr(token, "text", None)
        if isinstance(t, str):
            return t

    t = getattr(chunk, "text", None)
    return t if isinstance(t, str) else ""

graph = StateGraph(QAState)
graph.add_node("answer", answer_node)
graph.set_entry_point("answer")
graph.add_edge("answer", END)
app_graph = graph.compile()

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # includes chrome-extension://<id>
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=86400
)

class AskRequest(BaseModel):
    question: str
    context: str
    images: list[str] = []
    pageUrl: str | None = None
    docRef: dict | None = None

@app.post("/ask")
async def ask(req: AskRequest, authorization: str | None = Header(default=None)):
    payload = {
        "question": req.question,
        "context": req.context,
        "images": req.images,
        "pageUrl": req.pageUrl,
        "docRef": req.docRef,
        "accessToken": _bearer_token_from_header(authorization),
    }

    try:
        result = app_graph.invoke(payload)
        return {"answer": result.get("answer") or "i dont know due to insufficient information"}
    except Exception as e:
        logger.exception("Unhandled error in /ask")
        if DEBUG:
            return {"answer": "server error please try again", "error": str(e)}
        return {"answer": "server error please try again"}


@app.post("/ask_stream")
async def ask_stream(req: AskRequest, authorization: str | None = Header(default=None)):
    # Merge the bearer token into the retrieval path (not into the prompt).
    access_token = _bearer_token_from_header(authorization)

    # If we detected a Drive file, attempt to retrieve relevant snippets from it.
    merged_context = req.context or ""
    extra_image_text = ""
    if req.docRef and isinstance(req.docRef, dict) and req.docRef.get("fileId") and access_token:
        try:
            doc_text = _extract_text_from_drive_file(
                file_id=str(req.docRef.get("fileId")),
                access_token=access_token,
                question=req.question,
            )
            if doc_text.strip():
                doc_snips = select_relevant_snippets(
                    doc_text,
                    req.question,
                    max_chars=MAX_CONTEXT_CHARS,
                    chunk_size=CHUNK_SIZE_CHARS,
                    overlap=CHUNK_OVERLAP_CHARS,
                )
                if doc_snips.strip():
                    merged_context = f"Open Document (retrieved):\n{doc_snips}\n\n---\n\n" + merged_context

            if _looks_like_visual_question(req.question):
                try:
                    pdf_bytes = _drive_get_pdf_bytes_for_visual(str(req.docRef.get("fileId")), access_token)
                    hint = _extract_page_hint_from_url(req.pageUrl)
                    ocr = _extract_pdf_diagram_ocr(pdf_bytes, question=req.question, page_hint_1based=hint)
                    if ocr.strip():
                        extra_image_text = f"Document Diagram OCR (selected pages):\n{ocr}".strip()
                except Exception:
                    pass
        except Exception:
            pass

    prompt, fallback = _build_prompt_and_maybe_fallback(
        question=req.question,
        context=merged_context,
        images=req.images,
        extra_image_text=extra_image_text,
    )

    def sse(data: dict) -> str:
        # Ensure single-line SSE frames; JSON escapes newlines safely.
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        try:
            if fallback is not None:
                yield sse({"type": "token", "text": fallback})
                yield sse({"type": "done"})
                return

            if not prompt:
                yield sse({"type": "token", "text": "i dont know due to insufficient information"})
                yield sse({"type": "done"})
                return

            streamed_any = False
            full_text_parts: list[str] = []
            try:
                for chunk in _hf_stream_tokens(prompt):
                    text = _extract_stream_text(chunk)
                    if not text:
                        continue
                    streamed_any = True
                    full_text_parts.append(text)
                    yield sse({"type": "token", "text": text})
            except Exception:
                # Some HF backends/models don't support streaming. Fall back to non-stream.
                full = _hf_generate_with_continuation(prompt)
                full_text_parts = [full] if isinstance(full, str) else []
                for piece in _chunk_text_for_sse(full, max_chars=24):
                    streamed_any = True
                    yield sse({"type": "token", "text": piece})

            # If the stream produced nothing (edge cases), emit the required fallback.
            if not streamed_any:
                yield sse({"type": "token", "text": "i dont know due to insufficient information"})

            # If the model/provider stopped early but we got a coherent partial answer,
            # try a single continuation and stream it.
            full_text = "".join(full_text_parts) if full_text_parts else ""
            if full_text and _looks_truncated(full_text):
                tail = full_text[-1200:]
                cont_prompt = f"""{prompt}

The assistant's previous answer was cut off. Continue exactly where it stopped.
Do not repeat any text. Continue in GitHub-flavored Markdown.

Previous partial answer (tail):
{tail}
"""
                cont = _hf_generate_text(cont_prompt)
                if isinstance(cont, str) and cont.strip():
                    for piece in _chunk_text_for_sse(cont, max_chars=24):
                        yield sse({"type": "token", "text": piece})

            yield sse({"type": "done"})
        except Exception as e:
            logger.exception("Unhandled error in /ask_stream")
            # Always include a sanitized error hint so the extension can show something actionable.
            err_text = str(e)
            if not err_text:
                err_text = "unknown error"
            yield sse({"type": "error", "message": "server error please try again", "error": err_text})
            yield sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
