import fitz
import pandas as pd
import docx
import re

from pathlib import Path

base_dir = Path("./WorkDatabase")


def _clean_extracted_text(content: str) -> str:
    content = re.sub(r'[ \t]{2,}', ' ', content)
    content = re.sub(r'^[ \t]+|[ \t]+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{2,}', '\n', content).strip('\n')
    content = re.sub(r'([,?!;:。.])\1+', r'\1', content)
    return content.strip()


def _resolve_read_path(name: str) -> Path:
    """读取用路径解析：绝对路径任意可读；相对路径相对于 WorkDatabase。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("路径不能为空")
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / name).resolve()


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


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text(file_path):
    """
    Extract text content from a specified file.

    This function supports text extraction from multiple file formats, including
    PDF, Word (doc/docx), Excel (xlsx/xls), and plain text (txt) files. After
    extraction, the text is automatically cleaned by removing excess whitespace,
    line breaks, and duplicate punctuation marks.

    Args:
        file_path (str): 本地文件路径（绝对路径任意；相对路径相对于 WorkDatabase）。需含扩展名。

    Returns:
        str: The extracted and cleaned text content. Returns None if an error
             occurs during processing.
    """
    try:
        resolved = _resolve_read_path(file_path)

        if not resolved.exists():
            print(f"文件不存在：{resolved}")
            return None

        file_type = resolved.suffix.lstrip('.').lower()

        if file_type == 'pdf':
            content = extract_text_from_pdf(resolved)
        elif file_type in ['doc', 'docx']:
            content = extract_text_from_docx(resolved)
        elif file_type in ['xlsx', 'xls']:
            content = extract_text_from_excel(resolved)
        elif file_type == 'txt':
            content = extract_text_from_txt(resolved)
        else:
            raise ValueError(f"不支持的文件格式：{file_type}")

        if content is None or len(content) == 0:
            raise ValueError("提取内容为空，请检查文件情况")

        return _clean_extracted_text(content)
    except ValueError as e:
        print(f"路径错误：{str(e)}")
        return None
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None
