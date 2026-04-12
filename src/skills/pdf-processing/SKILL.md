---
name: pdf-processing
description: 从PDF文件中提取文本和表格、填写表单、合并文档。当处理PDF文件或用户提到PDF、表单、文档提取时使用此Skill。
---

# PDF 处理 Skill

本 Skill 提供 PDF 文件处理能力，包括文本提取、表格解析、表单填写等功能。

## 快速开始

### 基本文本提取

使用 `pdfplumber` 从 PDF 中提取文本：

```python
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF文件提取所有文本"""
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    return "\n\n".join(text_content)
```

### 表格提取

```python
def extract_tables_from_pdf(pdf_path: str) -> list:
    """从PDF中提取所有表格"""
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for table in tables:
                all_tables.append({
                    "page": page_num,
                    "data": table
                })
    return all_tables
```

## 工作流程

1. **确认文件路径**: 首先确认用户提供的 PDF 文件路径是否有效
2. **选择处理方式**: 
   - 纯文本提取 → 使用 `extract_text_from_pdf`
   - 表格数据 → 使用 `extract_tables_from_pdf`
   - 特定页面 → 指定页码范围
3. **输出结果**: 将提取的内容保存到文件或直接返回

## 最佳实践

- 对于大型 PDF 文件，建议分页处理以避免内存问题
- 表格提取前先检查页面是否包含表格结构
- 处理扫描版 PDF 时需要先进行 OCR 处理

## 相关资源

- 详细的表单填写指南请参考 [FORMS.md](FORMS.md)
- 高级 API 参考请参考 [REFERENCE.md](REFERENCE.md)

