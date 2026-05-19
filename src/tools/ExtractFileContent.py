import fitz
import pandas as pd
import docx
import re

from pathlib import Path

from path_sandbox import resolve_readable_path, runtime_repo_root

_WORK_DATABASE_ROOT = runtime_repo_root() / "WorkDatabase"


def _clean_extracted_text(content: str) -> str:
    content = re.sub(r'[ \t]{2,}', ' ', content)
    content = re.sub(r'^[ \t]+|[ \t]+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{2,}', '\n', content).strip('\n')
    content = re.sub(r'([,?!;:。.])\1+', r'\1', content)
    return content.strip()


def _resolve_read_path(name: str) -> Path:
    """读取用路径：限制在 WorkDatabase 与 src/skills。"""
    return resolve_readable_path(name, work_base=_WORK_DATABASE_ROOT, repo_root=runtime_repo_root())


def extract_text_from_pdf(file_path):
    """从PDF文件路径提取文本"""
    try:
        content = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text = page.get_text()
                if text:
                    content += text + "\n"

        if not content.strip():
            raise ValueError("无法从PDF中提取文本内容")
        return content
    except Exception as e:
        print(f"PDF处理错误：{str(e)}")
        return None


def extract_text_from_pdf_bytes(data: bytes) -> str | None:
    """从 PDF 字节流提取并清洗文本（供微信等无落盘路径的场景）。"""
    if not data:
        return None
    try:
        content = ""
        with fitz.open(stream=data, filetype="pdf") as pdf:
            for page in pdf:
                text = page.get_text()
                if text:
                    content += text + "\n"
        if not content.strip():
            raise ValueError("无法从PDF中提取文本内容")
        return _clean_extracted_text(content)
    except Exception as e:
        print(f"PDF字节流处理错误：{str(e)}")
        return None


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
        elif ext in (".txt", ".md", ".csv", ".json"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
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
