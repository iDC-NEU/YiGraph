#!/usr/bin/env python3
"""
使用示例：演示如何使用ArxivPaperDownloader
"""

from src.download_graph_papers import ArxivPaperDownloader
import json
import argparse
import time
import logging


def example_basic_search():
    """基本搜索示例"""
    print("=== 基本搜索示例 ===")

    # 创建下载器
    downloader = ArxivPaperDownloader()

    # 定义搜索关键词
    keywords = ["graph neural network", "GNN"]

    # 搜索论文
    papers = downloader.search_papers(
        keywords=keywords,
        max_results=50,
        sort_by="submittedDate",
        sort_order="descending"
    )

    print(f"找到 {len(papers)} 篇论文")

    # 显示前5篇论文
    for i, paper in enumerate(papers[:5], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   作者: {', '.join(paper['authors'][:2])}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   发布时间: {paper['published_date']}")


def example_custom_keywords():
    """自定义关键词搜索示例"""
    print("\n=== 自定义关键词搜索示例 ===")

    downloader = ArxivPaperDownloader()

    # 自定义关键词
    custom_keywords = [
        "knowledge graph",
        "graph embedding",
        "node classification"
    ]

    papers = downloader.search_papers(
        keywords=custom_keywords,
        max_results=30
    )

    # 按被引次数排序
    papers_sorted = downloader.sort_by_citations(papers)

    print(f"找到 {len(papers_sorted)} 篇论文，按被引次数排序:")

    for i, paper in enumerate(papers_sorted[:5], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   被引次数: {paper['citation_count']}")
        print(f"   作者: {', '.join(paper['authors'][:2])}")


def example_save_and_download():
    """保存和下载示例"""
    print("\n=== 保存和下载示例 ===")

    downloader = ArxivPaperDownloader()

    # 搜索特定领域的论文
    keywords = ["graph attention", "graph transformer"]
    papers = downloader.search_papers(keywords=keywords, max_results=20)

    if papers:
        # 保存论文信息
        output_file = downloader.save_papers(
            papers, output_dir="example_papers")
        print(f"论文信息已保存到: {output_file}")

        # 下载前3篇论文的PDF并转换为文本
        print("下载前3篇论文的PDF并转换为文本...")
        success_count = 0
        text_success_count = 0

        for paper in papers[:3]:
            result = downloader.download_and_convert_pdf(
                paper,
                pdf_dir="example_papers/pdfs",
                text_dir="example_papers/texts"
            )
            if result['pdf_path']:
                success_count += 1
            if result['text_path']:
                text_success_count += 1

        print(f"成功下载 {success_count}/3 篇PDF，成功转换 {text_success_count}/3 篇文本")


def example_load_saved_papers():
    """加载已保存的论文示例"""
    print("\n=== 加载已保存的论文示例 ===")

    try:
        with open("example_papers/graph_papers_*.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = data['papers']
        print(f"加载了 {len(papers)} 篇论文")

        # 显示论文统计信息
        categories = {}
        for paper in papers:
            for category in paper['categories']:
                categories[category] = categories.get(category, 0) + 1

        print("\n论文分类统计:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {category}: {count} 篇")

    except FileNotFoundError:
        print("未找到已保存的论文文件，请先运行保存示例")


def example_pdf_to_text():
    """PDF转文本示例"""
    print("\n=== PDF转文本示例 ===")

    downloader = ArxivPaperDownloader()

    # 假设已经有一个PDF文件
    pdf_path = "example_papers/pdfs/2301.12345.pdf"

    if os.path.exists(pdf_path):
        # 转换为文本
        text_path = downloader.pdf_to_text(pdf_path, "example_papers/texts")

        if text_path:
            print(f"PDF已转换为文本: {text_path}")

            # 读取并显示文本的前500个字符
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                    print(f"\n文本预览 (前500字符):")
                    print("-" * 50)
                    print(
                        text_content[:500] + "..." if len(text_content) > 500 else text_content)
                    print("-" * 50)
            except Exception as e:
                print(f"读取文本文件失败: {e}")
        else:
            print("PDF转文本失败")
    else:
        print(f"PDF文件不存在: {pdf_path}")
        print("请先运行下载示例来获取PDF文件")


def main():
    """主函数 - 参数化版本"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="arXiv图相关论文下载器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用 - 使用默认参数
  python example_usage.py
  
  # 自定义关键词和数量
  python example_usage.py --keywords "graph neural network" "GNN" --max_results 50
  
  # 按时间排序，下载并转换
  python example_usage.py --sort_by time --download_pdfs --convert_text
  
  # 按引用排序，只下载前10篇
  python example_usage.py --sort_by citation --max_results 10 --download_pdfs
  
  # 完整示例
  python example_usage.py --keywords "knowledge graph" "graph embedding" --max_results 30 --sort_by citation --download_pdfs --convert_text --output_dir "my_papers"
        """
    )

    # 添加参数
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=[
            "graph neural network",
            "graph attention",
            "graph convolution",
            "graph embedding",
            "network embedding",
            "node classification",
            "link prediction",
            "graph clustering",
            "graph mining",
            "knowledge graph"
        ],
        help='搜索关键词列表 (默认: 图相关关键词)'
    )

    parser.add_argument(
        '--max_results',
        type=int,
        default=100,
        help='最大搜索结果数量 (默认: 100)'
    )

    parser.add_argument(
        '--sort_by',
        choices=['time', 'citation', 'relevance'],
        default='time',
        help='排序方式: time(时间), citation(引用), relevance(相关性) (默认: time)'
    )

    parser.add_argument(
        '--sort_order',
        choices=['ascending', 'descending'],
        default='descending',
        help='排序顺序: ascending(升序), descending(降序) (默认: descending)'
    )

    parser.add_argument(
        '--download_pdfs',
        action='store_true',
        help='是否下载PDF文件'
    )

    parser.add_argument(
        '--convert_text',
        action='store_true',
        help='是否将PDF转换为文本'
    )

    parser.add_argument(
        '--output_dir',
        default='papers',
        help='输出目录 (默认: papers)'
    )

    parser.add_argument(
        '--download_limit',
        type=int,
        default=20,
        help='下载PDF的最大数量 (默认: 20)'
    )

    parser.add_argument(
        '--show_preview',
        action='store_true',
        help='是否显示论文预览'
    )

    parser.add_argument(
        '--preview_count',
        type=int,
        default=10,
        help='预览论文数量 (默认: 10)'
    )

    parser.add_argument(
        '--run_examples',
        action='store_true',
        help='是否运行示例函数'
    )

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    logger = logging.getLogger(__name__)

    # 如果指定运行示例，则运行示例函数
    if args.run_examples:
        logger.info("运行示例函数...")
        example_basic_search()
        example_custom_keywords()
        example_save_and_download()
        example_load_saved_papers()
        example_pdf_to_text()
        return

    # 创建下载器
    downloader = ArxivPaperDownloader()

    # 映射排序参数
    sort_mapping = {
        'time': 'submittedDate',
        'citation': 'relevance',  # 引用排序在后续处理
        'relevance': 'relevance'
    }

    # 搜索论文
    logger.info(f"开始搜索论文...")
    logger.info(f"关键词: {args.keywords}")
    logger.info(f"最大结果数: {args.max_results}")
    logger.info(f"排序方式: {args.sort_by}")

    papers = downloader.search_papers(
        keywords=args.keywords,
        max_results=args.max_results,
        sort_by=sort_mapping[args.sort_by],
        sort_order=args.sort_order
    )

    if not papers:
        logger.error("未找到相关论文")
        return

    logger.info(f"找到 {len(papers)} 篇论文")

    # 如果需要按引用排序，进行二次排序
    if args.sort_by == 'citation':
        logger.info("按被引次数排序...")
        papers_sorted = downloader.sort_by_citations(papers)
    else:
        papers_sorted = papers

    # 保存论文信息
    output_file = downloader.save_papers(papers_sorted, args.output_dir)
    logger.info(f"论文信息已保存到: {output_file}")

    # 显示论文预览
    if args.show_preview:
        logger.info(f"\n前{args.preview_count}篇论文:")
        for i, paper in enumerate(papers_sorted[:args.preview_count], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            if args.sort_by == 'citation':
                print(f"   被引次数: {paper['citation_count']}")
            print(f"   发布时间: {paper['published_date']}")
            print(f"   分类: {', '.join(paper['categories'][:3])}")

            # 下载PDF和转换文本
        if args.download_pdfs:
            logger.info(f"开始下载PDF...")
            success_count = 0
            text_success_count = 0
            tei_success_count = 0

            download_limit = min(args.download_limit, len(papers_sorted))

            for i, paper in enumerate(papers_sorted[:download_limit], 1):
                logger.info(f"处理第 {i}/{download_limit} 篇论文: {paper['title']}")

                if args.convert_text:
                    # 下载并转换
                    result = downloader.download_and_convert_pdf(
                        paper,
                        pdf_dir=f"{args.output_dir}/pdfs",
                        text_dir=f"{args.output_dir}/texts",
                        save_tei=True
                    )
                    if result['pdf_path']:
                        success_count += 1
                    if result['text_path']:
                        text_success_count += 1
                    if result['tei_path']:
                        tei_success_count += 1
                else:
                    # 只下载PDF
                    if downloader.download_pdf(paper, f"{args.output_dir}/pdfs"):
                        success_count += 1

                time.sleep(1)  # 避免请求过于频繁

            # 输出结果统计
            if args.convert_text:
                logger.info(f"处理完成！")
                logger.info(f"  - 成功下载: {success_count}/{download_limit} 篇PDF")
                logger.info(
                    f"  - 成功转换: {text_success_count}/{download_limit} 篇文本")
                logger.info(
                    f"  - 成功生成: {tei_success_count}/{download_limit} 篇TEI XML")
                logger.info(f"  - 输出目录: {args.output_dir}")
            else:
                logger.info(f"下载完成！成功下载 {success_count}/{download_limit} 篇PDF")
                logger.info(f"输出目录: {args.output_dir}/pdfs")

    # 如果没有指定下载，显示提示
    if not args.download_pdfs:
        logger.info(f"\n论文信息已保存到: {output_file}")
        logger.info("如需下载PDF，请使用 --download_pdfs 参数")
        logger.info("如需转换为文本，请使用 --convert_text 参数")

    # 下载PDF和转换文本
    if args.download_pdfs:
        logger.info(f"开始下载PDF...")
        success_count = 0
        text_success_count = 0
        tei_success_count = 0

        download_limit = min(args.download_limit, len(papers_sorted))

        for i, paper in enumerate(papers_sorted[:download_limit], 1):
            logger.info(f"处理第 {i}/{download_limit} 篇论文: {paper['title']}")

            if args.convert_text:
                # 下载并转换
                result = downloader.download_and_convert_pdf(
                    paper,
                    pdf_dir=f"{args.output_dir}/pdfs",
                    text_dir=f"{args.output_dir}/texts",
                    save_tei=True
                )
                if result['pdf_path']:
                    success_count += 1
                if result['text_path']:
                    text_success_count += 1
                if result['tei_path']:
                    tei_success_count += 1
            else:
                # 只下载PDF
                if downloader.download_pdf(paper, f"{args.output_dir}/pdfs"):
                    success_count += 1

            time.sleep(1)  # 避免请求过于频繁

        # 输出结果统计
        if args.convert_text:
            logger.info(f"处理完成！")
            logger.info(f"  - 成功下载: {success_count}/{download_limit} 篇PDF")
            logger.info(f"  - 成功转换: {text_success_count}/{download_limit} 篇文本")
            logger.info(
                f"  - 成功生成: {tei_success_count}/{download_limit} 篇TEI XML")
            logger.info(f"  - 输出目录: {args.output_dir}")
        else:
            logger.info(f"下载完成！成功下载 {success_count}/{download_limit} 篇PDF")
            logger.info(f"输出目录: {args.output_dir}/pdfs")


if __name__ == "__main__":
    main()
