# GROBID PDF文本提取使用说明

## 概述

本系统使用GROBID (GeneRation Of BIbliographic Data) 进行PDF结构化文本提取，相比传统的pdftotext工具，GROBID能够：

- 识别论文的章节结构
- 提取标题和段落
- 过滤无关内容（图表、参考文献等）
- 生成结构化的纯文本

## 安装和启动

### 1. 启动GROBID服务

```bash
# 给脚本执行权限
chmod +x start_grobid.sh

# 启动GROBID服务
./start_grobid.sh
```

### 2. 验证服务状态

```bash
# 检查服务是否正常运行
curl http://localhost:8070/api/isalive
```

## 功能特点

### 结构化提取
- 识别论文章节（Introduction, Methods, Results等）
- 保留章节标题和层级结构
- 过滤非核心内容

### 智能过滤
**跳过的章节：**
- References / Bibliography
- Related Work
- Experiments / Experimental Results
- Appendix / Appendices
- Acknowledgments
- Conclusions

**跳过的段落：**
- 包含"figure", "table", "fig.", "tab."的段落
- 算法描述段落
- 图表说明

### 输出格式

#### 结构化文本 (.txt)
```
# 论文标题

**Authors**: 作者1, 作者2

**Abstract**:
论文摘要内容...

## Introduction
章节内容...

### 子章节标题
子章节内容...
```

#### TEI XML文件 (.xml)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <titleStmt>
      <title>论文标题</title>
      <author>作者1</author>
      <author>作者2</author>
    </titleStmt>
    <profileDesc>
      <abstract>论文摘要</abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>章节内容...</p>
      </div>
    </body>
  </text>
</TEI>
```

## 使用方法

### 基本使用
```python
from download_graph_papers import ArxivPaperDownloader

downloader = ArxivPaperDownloader()

# 使用GROBID转换PDF
text_path = downloader.pdf_to_text(
    pdf_path="papers/pdfs/2301.12345.pdf",
    output_dir="papers/texts",
    grobid_url="http://localhost:8070"
)
```

### 命令行使用
```bash
# 下载并转换PDF
python example_usage.py --download_pdfs --convert_text
```

## 技术实现

### 1. GROBID服务调用
```python
# 发送PDF到GROBID API
response = requests.post(
    "http://localhost:8070/api/processFulltextDocument",
    files={'input': pdf_file}
)
```

### 2. TEI XML解析
```python
# 使用XPath提取结构化内容
body = root.xpath('//tei:text/tei:body', namespaces=namespaces)
for div in body.xpath('.//tei:div', namespaces=namespaces):
    # 提取章节标题和段落
```

### 3. 内容过滤
```python
# 跳过特定章节
skip_section_keywords = [
    "references", "related work", "experiment"
]

# 跳过特定段落
skip_paragraph_keywords = [
    "figure", "table", "fig.", "tab."
]
```

## 优势对比

| 特性 | pdftotext | GROBID |
|------|-----------|--------|
| 文本提取 | ✅ 基础 | ✅ 高级 |
| 结构识别 | ❌ 无 | ✅ 章节标题 |
| 内容过滤 | ❌ 无 | ✅ 智能过滤 |
| 格式保持 | ❌ 有限 | ✅ 结构化 |
| 处理速度 | ✅ 快 | ⚠️ 较慢 |
| 资源需求 | ✅ 低 | ⚠️ 需要Docker |

## 故障排除

### 1. GROBID服务无法启动
```bash
# 检查Docker状态
docker ps

# 查看容器日志
docker logs grobid

# 重启服务
docker restart grobid
```

### 2. 转换失败
```bash
# 检查服务健康状态
curl http://localhost:8070/api/isalive

# 检查端口占用
netstat -tlnp | grep 8070
```

### 3. 内存不足
```bash
# 增加Docker内存限制
docker run -d --name grobid -p 8070:8070 \
    --memory=4g --memory-swap=4g \
    lfoppiano/grobid:0.7.3
```

## 性能优化

### 1. 批量处理
```python
# 批量转换多个PDF
for pdf_file in pdf_files:
    text_path = downloader.pdf_to_text(pdf_file)
    time.sleep(1)  # 避免请求过频
```

### 2. 错误重试
```python
# 添加重试机制
max_retries = 3
for attempt in range(max_retries):
    try:
        text_path = downloader.pdf_to_text(pdf_path)
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise e
        time.sleep(2)
```

## 配置选项

### GROBID服务配置
```bash
# 自定义端口
docker run -d --name grobid -p 8080:8070 lfoppiano/grobid:0.7.3

# 自定义内存限制
docker run -d --name grobid -p 8070:8070 \
    --memory=2g --memory-swap=2g \
    lfoppiano/grobid:0.7.3
```

### 文本提取配置
```python
# 自定义过滤规则
skip_section_keywords = [
    "references", "related work", "experiment",
    "appendix", "bibliography", "acknowledgments"
]

skip_paragraph_keywords = [
    "figure", "table", "fig.", "tab.", "algorithm"
]
```

## 示例输出

### 输入PDF结构
```
1. Introduction
2. Related Work
3. Methodology
4. Experiments
5. Results
6. Conclusion
7. References
```

### 输出文本
```
# Graph Neural Networks: A Comprehensive Survey

**Authors**: Zhang, Li, Wang

**Abstract**:
Graph neural networks (GNNs) have emerged as a powerful framework for learning representations of graph-structured data...

## Introduction
Graph neural networks (GNNs) have emerged as a powerful framework for learning representations of graph-structured data...

## Methodology
We propose a novel graph neural network architecture that leverages attention mechanisms...

### Model Architecture
The proposed model consists of three main components...

## Results
Our experimental results demonstrate significant improvements over baseline methods...
```

### 输出目录结构
```
papers/
├── pdfs/
│   └── 2301.12345.pdf
└── texts/
    ├── 2301.12345.txt          # 结构化文本
    └── tei_xml/
        └── 2301.12345.xml      # TEI XML文件
```

## 注意事项

1. **服务依赖**: 需要Docker环境和GROBID服务
2. **处理时间**: 比pdftotext慢，但质量更高
3. **资源消耗**: 需要更多内存和CPU资源
4. **网络依赖**: 需要稳定的网络连接
5. **文件大小**: 大PDF文件处理时间较长 