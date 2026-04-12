# PDF 表单填写指南

本文档提供 PDF 表单填写的详细指南。

## 使用 PyPDF2 填写表单

```python
from PyPDF2 import PdfReader, PdfWriter

def fill_pdf_form(input_path: str, output_path: str, data: dict) -> bool:
    """
    填写 PDF 表单
    
    Parameters:
        input_path: 输入 PDF 文件路径
        output_path: 输出 PDF 文件路径
        data: 表单字段数据，格式为 {field_name: value}
        
    Returns:
        是否填写成功
    """
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()
        
        # 复制所有页面
        for page in reader.pages:
            writer.add_page(page)
        
        # 更新表单字段
        writer.update_page_form_field_values(
            writer.pages[0], 
            data
        )
        
        # 保存结果
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        return True
    except Exception as e:
        print(f"表单填写失败: {e}")
        return False
```

## 获取表单字段

```python
def get_form_fields(pdf_path: str) -> dict:
    """获取 PDF 中的所有表单字段"""
    reader = PdfReader(pdf_path)
    fields = reader.get_form_text_fields()
    return fields if fields else {}
```

## 注意事项

1. 确保 PDF 是可编辑的表单格式
2. 字段名称区分大小写
3. 某些字段可能是只读的，无法修改

