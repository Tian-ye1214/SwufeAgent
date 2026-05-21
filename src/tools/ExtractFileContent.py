import re
from html.parser import HTMLParser
from pathlib import Path

import docx
import fitz
import pandas as pd

from path_sandbox import resolve_readable_path, runtime_repo_root, work_database_root

_WORK_DATABASE_ROOT = work_database_root()
PDF_PARSE_FAILURE_HINT = "（PDF 附件无法解析为文本，请尝试发送截图或纯文本。）"


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return "\n".join(chunk for chunk in self._chunks if chunk.strip())


def _clean_extracted_text(content: str) -> str:
    content = re.sub(r"[ \t]{2,}", " ", content)
    content = re.sub(r"^[ \t]+|[ \t]+$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n{2,}", "\n", content).strip("\n")
    content = re.sub(r"([,?!;:。？])\1+", r"\1", content)
    return content.strip()


def _resolve_read_path(name: str) -> Path:
    """读取用路径：限制在 WorkDatabase 与 src/skills。"""
    return resolve_readable_path(name, work_base=_WORK_DATABASE_ROOT, repo_root=runtime_repo_root())


def _read_pdf_pages(pdf) -> str:
    content = ""
    for page in pdf:
        text = page.get_text()
        if text:
            content += text + "\n"
    if not content.strip():
        raise ValueError("无法从PDF中提取文本内容")
    return content


def extract_text_from_pdf(file_path):
    """从 PDF 文件路径提取文本。"""
    try:
        with fitz.open(file_path) as pdf:
            return _read_pdf_pages(pdf)
    except Exception as e:
        print(f"PDF处理错误：{str(e)}")
        return None


def extract_text_from_pdf_bytes(data: bytes) -> str | None:
    """从 PDF 字节流提取并清洗文本。"""
    if not data:
        return None
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            content = _read_pdf_pages(pdf)
        return _clean_extracted_text(content)
    except Exception as e:
        print(f"PDF字节流处理错误：{str(e)}")
        return None


def is_pdf_content(data: bytes, *, media_type: str = "", filename: str = "") -> bool:
    mt = (media_type or "").split(";")[0].strip().lower()
    if mt in ("application/pdf", "application/x-pdf"):
        return True
    if (filename or "").lower().endswith(".pdf"):
        return True
    return len(data or b"") >= 4 and data[:4] == b"%PDF"


def pdf_attachment_text_block(data: bytes, *, filename: str | None = None) -> str:
    extracted = extract_text_from_pdf_bytes(data)
    if not extracted:
        return PDF_PARSE_FAILURE_HINT
    label = f"【PDF 附件：{filename}】" if filename else "【PDF 附件】"
    return f"{label}\n\n{extracted}"


def extract_text_from_excel(file):
    df = pd.read_excel(file)
    content = ""
    for column in df.columns:
        content += f"{column}:\n"
        content += df[column].to_string() + "\n\n"
    return content


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_html(html_file) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(Path(html_file).read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.text()


def extract_text(name: str) -> str:
    """
    Extract text from a file (PDF, Excel, Word, plain text, etc.).

    Parameters:
        name: File path relative to WorkDatabase, or under src/skills
    """
    try:
        file_path = _resolve_read_path(name)
        if not file_path.exists():
            return f"Error: File '{name}' does not exist"

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            content = extract_text_from_pdf(str(file_path))
        elif ext in (".xlsx", ".xls"):
            content = extract_text_from_excel(str(file_path))
        elif ext == ".docx":
            content = extract_text_from_docx(str(file_path))
        elif ext in (".txt", ".md", ".markdown", ".csv", ".json"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
        elif ext in (".html", ".htm"):
            content = extract_text_from_html(file_path)
        else:
            return f"Error: Unsupported file type '{ext}'"

        if content is None:
            return f"Error: Could not extract content from '{name}'"
        if isinstance(content, str):
            content = _clean_extracted_text(content)
        return content if content else "File is empty"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error extracting text: {e}"
