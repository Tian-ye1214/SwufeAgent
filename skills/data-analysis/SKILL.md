---
name: data-analysis
description: 数据分析和可视化。用于处理CSV/Excel数据、统计分析、生成图表和报告。当需要分析数据或创建可视化图表时使用此Skill。
---

# 数据分析 Skill

本 Skill 提供数据分析和可视化能力，支持多种数据格式和分析方法。

## 快速开始

### 读取数据

```python
import pandas as pd

# 读取 CSV
df = pd.read_csv("data.csv")

# 读取 Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# 读取 JSON
df = pd.read_json("data.json")
```

### 基础统计分析

```python
def analyze_data(df: pd.DataFrame) -> dict:
    """执行基础统计分析"""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'describe': df.describe().to_dict()
    }
```

### 数据可视化

```python
import matplotlib.pyplot as plt

def create_visualization(df: pd.DataFrame, x: str, y: str, chart_type: str = 'line'):
    """创建数据可视化图表"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'line':
        plt.plot(df[x], df[y])
    elif chart_type == 'bar':
        plt.bar(df[x], df[y])
    elif chart_type == 'scatter':
        plt.scatter(df[x], df[y])
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x}')
    plt.tight_layout()
    plt.savefig('chart.png', dpi=150)
    plt.close()
    
    return 'chart.png'
```

## 工作流程

1. **加载数据**: 从文件或数据库读取数据
2. **数据清洗**: 处理缺失值、异常值、重复值
3. **探索性分析**: 了解数据分布和特征
4. **统计分析**: 计算相关指标
5. **可视化**: 生成图表展示结果
6. **输出报告**: 汇总分析结论

## 常用分析方法

### 描述性统计
- 均值、中位数、众数
- 标准差、方差
- 分位数、极值

### 相关性分析
```python
# 计算相关系数矩阵
correlation = df.corr()
```

### 分组聚合
```python
# 按类别分组统计
grouped = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count']
})
```

## 最佳实践

- 在分析前先了解数据结构和质量
- 处理缺失值时根据业务场景选择合适的方法
- 选择合适的图表类型展示数据特征
- 保存分析过程中的中间结果

