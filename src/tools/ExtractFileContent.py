import contextlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path

import docx
import fitz
import pandas as pd

from path_sandbox import resolve_readable_path, runtime_repo_root, work_database_root

_WORK_DATABASE_ROOT = work_database_root()

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


def _pdf_slug(name: str) -> str:
    stem = Path(name or "pdf").stem or "pdf"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return (slug or "pdf")[:80]


def _relative_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_WORK_DATABASE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _escape_markdown_cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|").strip()


def _markdown_table(rows: list[list]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [
        [_escape_markdown_cell(row[i]) if i < len(row) else "" for i in range(width)]
        for row in rows
    ]
    if not normalized:
        return ""
    header = normalized[0]
    body = normalized[1:] or [["" for _ in range(width)]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_page_tables(page) -> tuple[list[list[list]], list[str]]:
    tables: list[list[list]] = []
    warnings: list[str] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            finder = page.find_tables()
        for table in getattr(finder, "tables", []):
            rows = table.extract()
            if rows:
                tables.append(rows)
    except Exception as e:
        warnings.append(f"table extraction failed: {type(e).__name__}: {e}")
    return tables, warnings


def _extract_page_links(page) -> list[str]:
    links: list[str] = []
    for link in page.get_links():
        uri = link.get("uri")
        if uri:
            links.append(str(uri))
            continue
        target_page = link.get("page")
        if target_page is not None and target_page >= 0:
            links.append(f"page {target_page + 1}")
    return links


def _export_page_images(
    doc, page, page_number: int, output_root: Path
) -> tuple[list[dict], list[str]]:
    image_dir = output_root / "images"
    images: list[dict] = []
    warnings: list[str] = []
    for image_number, image_info in enumerate(page.get_images(full=True), 1):
        xref = image_info[0]
        try:
            extracted = doc.extract_image(xref)
            data = extracted.get("image")
            if not data:
                warnings.append(f"page {page_number} image {image_number}: empty image data")
                continue
            ext = (extracted.get("ext") or "png").lower()
            image_dir.mkdir(parents=True, exist_ok=True)
            output_path = image_dir / f"page-{page_number:03d}-image-{image_number:03d}.{ext}"
            output_path.write_bytes(data)
            images.append(
                {
                    "path": _relative_artifact_path(output_path),
                    "width": extracted.get("width") or image_info[2],
                    "height": extracted.get("height") or image_info[3],
                    "ext": ext,
                }
            )
        except Exception as e:
            warnings.append(
                f"page {page_number} image {image_number}: {type(e).__name__}: {e}"
            )
    return images, warnings


def _export_page_preview(page, page_number: int, output_root: Path) -> str:
    pages_dir = output_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    output_path = pages_dir / f"page-{page_number:03d}.png"
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(output_path)
    return _relative_artifact_path(output_path)


def _format_metadata(pdf) -> list[str]:
    lines = [
        "## Document Metadata",
        f"- Page count: {pdf.page_count}",
        f"- Encrypted: {str(bool(pdf.is_encrypted)).lower()}",
    ]
    metadata = pdf.metadata or {}
    for key in (
        "title",
        "author",
        "subject",
        "keywords",
        "creator",
        "producer",
        "creationDate",
        "modDate",
    ):
        value = metadata.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    return lines


def _format_pdf_document(pdf, source_name: str) -> str:
    output_root = _WORK_DATABASE_ROOT / "extracted" / _pdf_slug(source_name)
    lines = [f"# PDF Extraction: {Path(source_name).name or 'attachment.pdf'}", ""]
    lines.extend(_format_metadata(pdf))
    lines.append("")

    warnings: list[str] = []
    found_content = False
    for page_index, page in enumerate(pdf, 1):
        text = _clean_extracted_text(page.get_text("text") or "")
        tables, table_warnings = _extract_page_tables(page)
        images, image_warnings = _export_page_images(pdf, page, page_index, output_root)
        links = _extract_page_links(page)
        warnings.extend(f"page {page_index}: {w}" for w in table_warnings + image_warnings)

        image_only = not text and not tables and bool(images)
        found_content = found_content or bool(text or tables or images or links)

        lines.append(f"## Page {page_index}")
        lines.append(f"- image_only: {str(image_only).lower()}")
        if image_only:
            preview = _export_page_preview(page, page_index, output_root)
            lines.append(
                "- warning: OCR not performed; use read_image on the page preview "
                "if visual analysis is needed."
            )
            lines.append(f"- page_preview: {preview}")

        if text:
            lines.extend(["", "### Text", text])
        if tables:
            lines.extend(["", "### Tables"])
            for table_index, rows in enumerate(tables, 1):
                table_markdown = _markdown_table(rows)
                if table_markdown:
                    lines.extend([f"#### Table {table_index}", table_markdown])
        if images:
            lines.extend(["", "### Images"])
            for image_index, image in enumerate(images, 1):
                lines.append(
                    f"- image {image_index}: {image['path']} "
                    f"({image['width']}x{image['height']}, {image['ext']})"
                )
        if links:
            lines.extend(["", "### Links"])
            for link in links:
                lines.append(f"- {link}")
        lines.append("")

    if not found_content:
        warnings.append("No extractable text, tables, images, or links found; OCR not performed.")

    if warnings:
        lines.extend(["## Warnings", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines).strip()


def extract_text_from_pdf(file_path):
    """Extract structured Markdown from a PDF file path."""
    try:
        with fitz.open(file_path) as pdf:
            if pdf.needs_pass:
                return "Error: PDF is encrypted and requires a password"
            return _format_pdf_document(pdf, Path(file_path).name)
    except Exception as e:
        return f"Error: PDF processing failed - {type(e).__name__}: {e}"


def extract_text_from_pdf_bytes(data: bytes, *, filename: str | None = None) -> str | None:
    """Extract structured Markdown from PDF bytes."""
    if not data:
        return None
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                return "Error: PDF is encrypted and requires a password"
            return _format_pdf_document(pdf, filename or "attachment.pdf")
    except Exception as e:
        return f"Error: PDF processing failed - {type(e).__name__}: {e}"


def is_pdf_content(data: bytes, *, media_type: str = "", filename: str = "") -> bool:
    mt = (media_type or "").split(";")[0].strip().lower()
    if mt in ("application/pdf", "application/x-pdf"):
        return True
    if (filename or "").lower().endswith(".pdf"):
        return True
    return len(data or b"") >= 4 and data[:4] == b"%PDF"


def pdf_attachment_text_block(data: bytes, *, filename: str | None = None) -> str:
    extracted = extract_text_from_pdf_bytes(data, filename=filename)
    if not extracted:
        return "PDF attachment could not be parsed. OCR was not performed."
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
            return content if content else f"Error: Could not extract content from '{name}'"
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
