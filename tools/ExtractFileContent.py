import fitz
import pandas as pd
import docx
import re

from pathlib import Path

base_dir = Path("./WorkDatabase")
def _safe_path(name: str) -> Path:
    path = (base_dir / name).resolve()
    if not str(path).startswith(str(base_dir.resolve())):
        raise ValueError("检测到路径遍历：不允许访问 base_dir 之外的目录")
    return path


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
        file_path (str): The path to the file to extract text from (relative to WorkDatabase directory). 
                         Must include the file extension.

    Returns:
        str: The extracted and cleaned text content. Returns None if an error
             occurs during processing.
    """
    try:
        safe_file_path = _safe_path(file_path)
        
        if not safe_file_path.exists():
            print(f"文件不存在：{safe_file_path}")
            return None
        
        file_type = safe_file_path.suffix.lstrip('.').lower()
        
        if file_type == 'pdf':
            content = extract_text_from_pdf(safe_file_path)
        elif file_type in ['doc', 'docx']:
            content = extract_text_from_docx(safe_file_path)
        elif file_type in ['xlsx', 'xls']:
            content = extract_text_from_excel(safe_file_path)
        elif file_type == 'txt':
            content = extract_text_from_txt(safe_file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_type}")

        if content is None:
            raise ValueError("无法提取文件内容")

        content = re.sub(r'[ \t]{2,}', ' ', content)
        content = re.sub(r'^[ \t]+|[ \t]+$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n{2,}', '\n', content).strip('\n')
        content = re.sub(r'([,?!;:。.])\1+', r'\1', content)
        return content
    except ValueError as e:
        print(f"安全错误：{str(e)}")
        return None
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None
