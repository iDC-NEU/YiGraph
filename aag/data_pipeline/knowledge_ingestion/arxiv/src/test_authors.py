#!/usr/bin/env python3
"""
作者信息提取测试脚本

这个脚本用于测试作者信息提取功能是否正常工作。
"""

import os
import sys
import logging
from download_graph_papers import ArxivPaperDownloader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_grobid_service():
    """测试GROBID服务是否可用"""
    downloader = ArxivPaperDownloader()
    
    print("测试GROBID服务连接...")
    is_available = downloader._check_grobid_service()
    
    if is_available:
        print("✅ GROBID服务可用")
        return True
    else:
        print("❌ GROBID服务不可用")
        print("请确保GROBID服务正在运行：")
        print("  docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3")
        return False

def test_tei_parsing():
    """测试TEI XML解析功能"""
    downloader = ArxivPaperDownloader()
    
    # 创建一个简单的测试TEI XML
    test_tei = '''<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Test Paper Title</title>
        <author>John Doe</author>
        <author>Jane Smith</author>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <particDesc>
        <listOrg>
          <org>
            <orgName>Test University</orgName>
            <address>
              <addrLine>123 Test Street</addrLine>
              <country>Test Country</country>
            </address>
          </org>
        </listOrg>
      </particDesc>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This is a test introduction.</p>
      </div>
      <div>
        <head>Conclusion</head>
        <p>This is a test conclusion.</p>
      </div>
    </body>
  </text>
</TEI>'''
    
    print("测试TEI XML解析...")
    
    # 测试文本提取
    text_result = downloader._extract_structured_text_from_tei(test_tei)
    if text_result:
        print("✅ TEI XML文本提取成功")
        print(f"提取的文本长度: {len(text_result)} 字符")
    else:
        print("❌ TEI XML文本提取失败")
    
    # 测试作者信息提取
    authors_result = downloader.extract_authors_from_tei(test_tei)
    if authors_result and authors_result['total_count'] > 0:
        print("✅ 作者信息提取成功")
        print(f"找到 {authors_result['total_count']} 位作者:")
        for author in authors_result['authors']:
            print(f"  - {author['name']}")
    else:
        print("❌ 作者信息提取失败")

def test_pdf_processing():
    """测试PDF处理功能"""
    downloader = ArxivPaperDownloader()
    
    # 检查是否有测试PDF文件
    test_pdf = "example_papers/pdfs/test.pdf"
    
    if os.path.exists(test_pdf):
        print(f"测试PDF处理: {test_pdf}")
        
        # 测试PDF转文本
        text_path = downloader.pdf_to_text(test_pdf, "test_output")
        if text_path:
            print("✅ PDF转文本成功")
            print(f"文本文件: {text_path}")
        else:
            print("❌ PDF转文本失败")
        
        # 测试作者信息提取
        authors_info = downloader.extract_authors_from_pdf(test_pdf)
        if authors_info:
            print("✅ PDF作者信息提取成功")
            print(f"找到 {authors_info['total_count']} 位作者")
        else:
            print("❌ PDF作者信息提取失败")
    else:
        print(f"未找到测试PDF文件: {test_pdf}")
        print("请先下载一些PDF文件进行测试")

def main():
    """主测试函数"""
    print("作者信息提取功能测试")
    print("=" * 50)
    
    # 测试GROBID服务
    grobid_ok = test_grobid_service()
    print()
    
    # 测试TEI XML解析
    test_tei_parsing()
    print()
    
    # 测试PDF处理
    test_pdf_processing()
    print()
    
    print("测试完成！")
    
    if not grobid_ok:
        print("\n注意: GROBID服务不可用，PDF处理功能将无法正常工作。")
        print("请启动GROBID服务后重新运行测试。")

if __name__ == "__main__":
    main() 