# arXiv图相关论文下载器

这个模块提供了从arXiv爬取图相关论文的功能，支持关键词搜索、按被引次数排序并保存到本地。

## 功能特性

- 🔍 **关键词搜索**: 支持多个关键词组合搜索
- 📊 **智能排序**: 按被引次数排序（基于内容分析估算）
- 💾 **本地保存**: 将论文信息保存为JSON格式
- 📄 **PDF下载**: 支持下载论文PDF文件
- 🔄 **GROBID转换**: 使用GROBID服务将PDF转换为结构化文本
- 📝 **TEI XML**: 保存原始TEI XML文件供进一步分析
- 🎯 **章节提取**: 智能提取论文的关键章节内容（introduction和conclusion）
- 👥 **作者信息**: 提取论文作者信息，包括姓名、邮箱、机构等

## 安装依赖

### Python依赖
```bash
pip install requests lxml PyPDF2
```

### 系统依赖
需要安装poppler-utils来使用pdftotext功能：

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**CentOS/RHEL:**
```bash
sudo yum install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

## 基本使用

### 1. 直接运行主程序

```bash
python example_usage.py
```

这将：
- 搜索图相关的论文（使用预定义的关键词）
- 按时间排序
- 保存论文信息到本地

### 2. 使用命令行参数

```bash
# 基本搜索
python example_usage.py --show_preview

# 自定义关键词
python example_usage.py --keywords "graph neural network" "GNN" --max_results 50

# 按引用排序并下载PDF
python example_usage.py --sort_by citation --download_pdfs --convert_text

# 完整流程
python example_usage.py --keywords "knowledge graph" --max_results 30 --sort_by citation --download_pdfs --convert_text --output_dir "my_papers"

# 运行所有示例函数
python example_usage.py --run_examples
```

### 3. 使用API

```python
from download_graph_papers import ArxivPaperDownloader

# 创建下载器
downloader = ArxivPaperDownloader()

# 搜索论文
papers = downloader.search_papers(
    keywords=["graph neural network", "GNN"],
    max_results=100,
    sort_by="submittedDate",
    sort_order="descending"
)

# 按被引次数排序
papers_sorted = downloader.sort_by_citations(papers)

# 保存论文信息
output_file = downloader.save_papers(papers_sorted)

# 下载PDF并转换为文本
for paper in papers_sorted[:10]:
    result = downloader.download_and_convert_pdf(paper)
    if result['success']:
        print(f"成功处理: {paper['title']}")

# 提取作者信息
pdf_path = "papers/pdfs/2301.12345.pdf"
authors_info = downloader.extract_authors_from_pdf(pdf_path)
if authors_info:
    print(f"找到 {authors_info['total_count']} 位作者")
    for author in authors_info['authors']:
        print(f"- {author['name']} ({author['affiliation']})")

## API 参考

### ArxivPaperDownloader

主要的论文下载器类。

#### 方法

##### `search_papers(keywords, max_results=100, sort_by="relevance", sort_order="descending")`

搜索论文。

**参数:**
- `keywords` (List[str]): 关键词列表
- `max_results` (int): 最大结果数，默认100
- `sort_by` (str): 排序方式，可选值："relevance", "lastUpdatedDate", "submittedDate"
- `sort_order` (str): 排序顺序，可选值："ascending", "descending"

**返回:**
- `List[Dict]`: 论文列表

##### `sort_by_citations(papers)`

按被引次数排序论文。

**参数:**
- `papers` (List[Dict]): 论文列表

**返回:**
- `List[Dict]`: 排序后的论文列表

##### `save_papers(papers, output_dir="papers")`

保存论文信息到本地。

**参数:**
- `papers` (List[Dict]): 论文列表
- `output_dir` (str): 输出目录

**返回:**
- `str`: 保存的文件路径

##### `download_pdf(paper, output_dir="papers/pdfs")`

下载论文PDF。

**参数:**
- `paper` (Dict): 论文信息
- `output_dir` (str): 输出目录

**返回:**
- `bool`: 是否下载成功

##### `pdf_to_text(pdf_path, output_dir="papers/texts")`

将PDF转换为纯文本。

**参数:**
- `pdf_path` (str): PDF文件路径
- `output_dir` (str): 输出目录

**返回:**
- `Optional[str]`: 文本文件路径，如果失败返回None

##### `download_and_convert_pdf(paper, pdf_dir="papers/pdfs", text_dir="papers/texts")`

下载PDF并转换为文本。

**参数:**
- `paper` (Dict): 论文信息
- `pdf_dir` (str): PDF输出目录
- `text_dir` (str): 文本输出目录

**返回:**
- `Dict[str, Optional[str]]`: 包含PDF和文本文件路径的字典

##### `extract_authors_from_pdf(pdf_path, grobid_url="http://localhost:8070")`

从PDF文件中提取作者信息。

**参数:**
- `pdf_path` (str): PDF文件路径
- `grobid_url` (str): GROBID服务地址

**返回:**
- `Optional[Dict]`: 包含作者信息的字典

##### `extract_authors_from_tei(tei_xml)`

从TEI XML中提取作者信息。

**参数:**
- `tei_xml` (str): TEI XML字符串

**返回:**
- `Dict[str, any]`: 包含作者信息的字典

## 论文数据结构

每篇论文包含以下字段：

```python
{
    'arxiv_id': '2301.12345',           # arXiv ID
    'title': '论文标题',                 # 论文标题
    'summary': '论文摘要',               # 论文摘要
    'authors': ['作者1', '作者2'],       # 作者列表
    'published_date': '2023-01-01',     # 发布时间
    'categories': ['cs.AI', 'cs.LG'],   # 分类标签
    'links': [...],                     # 相关链接
    'download_url': 'https://...',      # PDF下载链接
    'citation_count': 42                # 被引次数（估算）
}
```

## 预定义关键词

程序默认使用以下图相关关键词：

- graph neural network
- graph attention
- graph convolution

## 作者信息提取

### 功能说明

系统支持从PDF文件或TEI XML中提取详细的作者信息，包括：

- **基本信息**: 姓名、邮箱、ORCID
- **机构信息**: 所属机构、地址、国家
- **角色信息**: 是否为通讯作者
- **结构化数据**: 姓名分离（名、姓）

### 使用示例

```python
from download_graph_papers import ArxivPaperDownloader

downloader = ArxivPaperDownloader()

# 从PDF提取作者信息
authors_info = downloader.extract_authors_from_pdf("paper.pdf")

# 从TEI XML提取作者信息
with open("paper.xml", "r") as f:
    tei_xml = f.read()
authors_info = downloader.extract_authors_from_tei(tei_xml)

# 查看结果
print(f"总作者数: {authors_info['total_count']}")
for author in authors_info['authors']:
    print(f"作者: {author['name']}")
    if author['email']:
        print(f"  邮箱: {author['email']}")
    if author['affiliation']:
        print(f"  机构: {author['affiliation']}")
    if author['is_corresponding']:
        print(f"  * 通讯作者")
```

### 运行作者信息提取示例

```bash
python extract_authors_example.py
```

这个脚本提供了交互式的作者信息提取功能，支持：
- 从单个PDF文件提取
- 从TEI XML文件提取
- 批量处理多个PDF文件
- graph embedding
- network embedding
- node classification
- link prediction
- graph clustering
- graph mining
- knowledge graph

## 输出文件

### JSON文件格式

```json
{
    "download_time": "2023-12-01T10:30:00",
    "total_papers": 150,
    "papers": [
        {
            "arxiv_id": "2301.12345",
            "title": "论文标题",
            "summary": "论文摘要",
            "authors": ["作者1", "作者2"],
            "published_date": "2023-01-01T00:00:00Z",
            "categories": ["cs.AI", "cs.LG"],
            "links": [...],
            "download_url": "https://arxiv.org/pdf/2301.12345.pdf",
            "citation_count": 42
        }
    ]
}
```

### 目录结构

```
papers/
├── graph_papers_20231201_103000.json  # 论文信息
├── pdfs/
│   ├── 2301.12345.pdf                 # 论文PDF
│   └── 2301.12346.pdf
└── texts/
    ├── 2301.12345.txt                 # 论文文本
    ├── 2301.12346.txt
    └── tei_xml/                       # TEI XML文件
        ├── 2301.12345.xml             # 结构化XML
        └── 2301.12346.xml
```

## 示例

查看 `example_usage.py` 文件获取更多使用示例。

## 注意事项

1. **请求频率**: 程序包含请求间隔，避免对arXiv服务器造成过大压力
2. **被引次数**: 当前版本使用基于内容的估算，实际应用中建议集成Google Scholar或Semantic Scholar API
3. **网络连接**: 需要稳定的网络连接来访问arXiv API
4. **存储空间**: 下载PDF文件会占用较多存储空间
5. **PDF转文本**: 需要安装poppler-utils，转换过程可能需要一些时间
6. **文本质量**: pdftotext提取的文本不包含表格和图片，仅提取纯文本内容

## 扩展功能

### 集成真实被引数据

可以扩展 `get_citation_count` 方法来集成真实的学术数据库API：

```python
def get_citation_count(self, paper: Dict) -> int:
    # 集成Google Scholar API
    # 集成Semantic Scholar API
    # 集成Scopus API
    pass
```

### 添加更多搜索选项

可以扩展搜索功能支持更多选项：

```python
def search_papers(self, 
                 keywords: List[str],
                 categories: List[str] = None,
                 date_from: str = None,
                 date_to: str = None,
                 max_results: int = 100):
    pass
```

## 文件结构

```
graphllm/data/
├── download_graph_papers.py    # 功能类文件（ArxivPaperDownloader类）
├── example_usage.py            # 主程序和示例（包含main函数和示例函数）
├── install_dependencies.sh     # 依赖安装脚本
└── README.md                   # 详细文档
```

## 许可证

本项目遵循MIT许可证。 