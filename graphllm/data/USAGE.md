# 使用说明

## 快速开始

### 1. 安装依赖
```bash
# 系统依赖
sudo apt-get install poppler-utils

# Python依赖
pip install requests lxml PyPDF2
```

### 2. 基本使用

**搜索论文：**
```bash
python example_usage.py --show_preview
```

**下载PDF并转换为文本：**
```bash
python example_usage.py --download_pdfs --convert_text
```

**自定义关键词：**
```bash
python example_usage.py --keywords "graph neural network" --max_results 50 --download_pdfs --convert_text
```

**按引用排序：**
```bash
python example_usage.py --sort_by citation --download_pdfs --convert_text
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--keywords` | 搜索关键词 | 图相关关键词 |
| `--max_results` | 最大搜索结果数 | 100 |
| `--sort_by` | 排序方式 (time/citation/relevance) | time |
| `--download_pdfs` | 是否下载PDF | False |
| `--convert_text` | 是否转换为文本 | False |
| `--output_dir` | 输出目录 | papers |
| `--download_limit` | 下载数量限制 | 20 |
| `--show_preview` | 显示预览 | False |

## 完整示例

```bash
# 搜索知识图谱相关论文，下载并转换
python example_usage.py \
  --keywords "knowledge graph" "graph embedding" \
  --max_results 30 \
  --sort_by citation \
  --download_pdfs \
  --convert_text \
  --show_preview \
  --output_dir "kg_papers"
```

## 输出结构

```
output_dir/
├── graph_papers_20231201_103000.json  # 论文信息
├── pdfs/                              # PDF文件
└── texts/                             # 文本文件
``` 