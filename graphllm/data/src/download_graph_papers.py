# 该文件的功能是从arxiv上爬取与图相关的论文，能够根据关键词进行搜索，同时按照被引次数进行排序，并保存到本地

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivPaperDownloader:
    """arXiv论文下载器"""
    
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_papers(self, 
                     keywords: List[str], 
                     max_results: int = 100,
                     sort_by: str = "relevance",
                     sort_order: str = "descending") -> List[Dict]:
        """
        搜索论文
        
        Args:
            keywords: 关键词列表
            max_results: 最大结果数
            sort_by: 排序方式 ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: 排序顺序 ("ascending", "descending")
            
        Returns:
            论文列表
        """
        query = " OR ".join([f'all:"{keyword}"' for keyword in keywords])
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            logger.info(f"搜索关键词: {keywords}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            # 解析XML响应
            papers = self._parse_arxiv_response(response.text)
            logger.info(f"找到 {len(papers)} 篇论文")
            
            return papers
            
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """解析arXiv XML响应"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('.//atom:entry', namespace):
                paper = self._extract_paper_info(entry, namespace)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"XML解析失败: {e}")
            
        return papers
    
    def _extract_paper_info(self, entry, namespace) -> Optional[Dict]:
        """提取论文信息"""
        try:
            # 基本信息
            title = entry.find('atom:title', namespace)
            title_text = title.text.strip() if title is not None and title.text else ""
            
            summary = entry.find('atom:summary', namespace)
            summary_text = summary.text.strip() if summary is not None and summary.text else ""
            
            # 作者信息
            authors = []
            for author in entry.findall('atom:author/atom:name', namespace):
                if author.text:
                    authors.append(author.text.strip())
            
            # 发布时间
            published = entry.find('atom:published', namespace)
            published_date = published.text if published is not None else ""
            
            # arXiv ID
            id_elem = entry.find('atom:id', namespace)
            arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else ""
            
            # 分类
            categories = []
            for category in entry.findall('atom:category', namespace):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # 链接
            links = []
            for link in entry.findall('atom:link', namespace):
                href = link.get('href')
                title_attr = link.get('title', '')
                if href:
                    links.append({
                        'href': href,
                        'title': title_attr
                    })
            
            return {
                'arxiv_id': arxiv_id,
                'title': title_text,
                'summary': summary_text,
                'authors': authors,
                'published_date': published_date,
                'categories': categories,
                'links': links,
                'download_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }
            
        except Exception as e:
            logger.error(f"提取论文信息失败: {e}")
            return None
    
    def get_citation_count(self, paper: Dict) -> int:
        """
        获取论文被引次数（这里使用模拟数据，实际应用中需要集成学术数据库API）
        
        Args:
            paper: 论文信息
            
        Returns:
            被引次数
        """
        # TODO: 这里可以集成Google Scholar、Semantic Scholar等API
        # 目前使用基于标题和摘要的简单估算
        title_words = len(paper['title'].split())
        summary_words = len(paper['summary'].split())
        
        # 简单的估算逻辑（实际应用中需要真实的API调用）
        base_score = title_words + summary_words // 10
        
        # 根据关键词权重调整
        graph_keywords = ['graph', 'network', 'node', 'edge', 'graph neural', 'gnn']
        title_lower = paper['title'].lower()
        summary_lower = paper['summary'].lower()
        
        keyword_score = 0
        for keyword in graph_keywords:
            if keyword in title_lower:
                keyword_score += 5
            if keyword in summary_lower:
                keyword_score += 2
        
        return max(0, base_score + keyword_score)
    
    def sort_by_citations(self, papers: List[Dict]) -> List[Dict]:
        """按被引次数排序"""
        for paper in papers:
            paper['citation_count'] = self.get_citation_count(paper)
        
        return sorted(papers, key=lambda x: x['citation_count'], reverse=True)
    
    def save_papers(self, papers: List[Dict], output_dir: str = "papers") -> str:
        """
        保存论文信息到本地
        
        Args:
            papers: 论文列表
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"graph_papers_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # 添加保存时间
        papers_with_metadata = {
            'download_time': datetime.now().isoformat(),
            'total_papers': len(papers),
            'papers': papers
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(papers_with_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"论文信息已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return ""
    
    def download_pdf(self, paper: Dict, output_dir: str = "papers/pdfs") -> bool:
        """
        下载论文PDF
        
        Args:
            paper: 论文信息
            output_dir: 输出目录
            
        Returns:
            是否下载成功
        """
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_url = paper.get('download_url')
        if not pdf_url:
            logger.warning(f"论文 {paper['arxiv_id']} 没有下载链接")
            return False
        
        filename = f"{paper['arxiv_id']}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        try:
            logger.info(f"下载PDF: {paper['title']}")
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF已保存: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"下载PDF失败 {paper['arxiv_id']}: {e}")
            return False
    
    def pdf_to_text(self, pdf_path: str, output_dir: str = "papers/texts", 
                   grobid_url: str = "http://localhost:8070", 
                   save_tei: bool = True) -> Optional[str]:
        """
        使用GROBID将PDF转换为结构化文本
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            grobid_url: GROBID服务地址
            save_tei: 是否保存TEI XML文件
            
        Returns:
            文本文件路径，如果失败返回None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        text_filename = f"{pdf_basename}.txt"
        text_path = os.path.join(output_dir, text_filename)
        
        # 创建TEI XML输出目录
        tei_dir = os.path.join(output_dir, "tei_xml")
        os.makedirs(tei_dir, exist_ok=True)
        tei_filename = f"{pdf_basename}.xml"
        tei_path = os.path.join(tei_dir, tei_filename)
        
        try:
            # Step 1: 检查GROBID服务是否可用
            if not self._check_grobid_service(grobid_url):
                logger.error(f"GROBID服务不可用: {grobid_url}")
                return None
            
            # Step 2: 调用GROBID接口转换PDF为TEI XML
            logger.info(f"使用GROBID转换PDF: {pdf_path}")
            tei_xml = self._convert_pdf_to_tei(pdf_path, grobid_url)
            
            if not tei_xml:
                logger.error("PDF转TEI XML失败")
                return None
            
            # Step 3: 保存TEI XML文件
            if save_tei:
                with open(tei_path, 'w', encoding='utf-8') as f:
                    f.write(tei_xml)
                logger.info(f"TEI XML已保存: {tei_path}")
            
            # Step 4: 解析TEI XML并提取结构化文本
            structured_text = self._extract_structured_text_from_tei(tei_xml)
            
            if not structured_text:
                logger.error("从TEI XML提取文本失败")
                return None
            
            # Step 5: 写入输出文件
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(structured_text)
            
            logger.info(f"结构化文本已保存: {text_path}")
            return text_path
            
        except Exception as e:
            logger.error(f"PDF转文本失败 {pdf_path}: {e}")
            return None
    
    def _check_grobid_service(self, grobid_url: str) -> bool:
        """检查GROBID服务是否可用"""
        try:
            response = requests.get(f"{grobid_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"GROBID服务检查失败: {e}")
            return False
    
    def _convert_pdf_to_tei(self, pdf_path: str, grobid_url: str) -> Optional[str]:
        """将PDF转换为TEI XML"""
        try:
            # 准备文件上传
            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': (os.path.basename(pdf_path), pdf_file, 'application/pdf')}
                
                # 发送POST请求到GROBID
                response = requests.post(
                    f"{grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.text
                else:
                    logger.error(f"GROBID API返回错误: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"PDF转TEI XML失败: {e}")
            return None
    

    def _extract_structured_text_from_tei(self, tei_xml: str) -> Optional[str]:
        """从TEI XML提取结构化文本（只提取introduction和conclusion章节）"""
        try:
            from lxml import etree

            # 解析XML
            root = etree.fromstring(tei_xml.encode('utf-8'))
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            structured_text = []

            # 提取标题
            title_el = root.xpath('//tei:titleStmt/tei:title', namespaces=namespaces)
            if title_el and title_el[0].text:
                title = title_el[0].text.strip()
                structured_text.append(f"# {title}")

            # 提取作者
            authors_el = root.xpath('//tei:titleStmt/tei:author', namespaces=namespaces)
            authors = []
            for a in authors_el:
                if a.text:
                    authors.append(a.text.strip())
            if authors:
                structured_text.append(f"**Authors**: {', '.join(authors)}\n")

            # 提取摘要
            abstract_el = root.xpath('//tei:profileDesc/tei:abstract', namespaces=namespaces)
            if abstract_el:
                abstract_text = " ".join(abstract_el[0].itertext()).strip()
                structured_text.append("**Abstract**:\n" + self._clean_text(abstract_text))

            # 提取正文内容
            body = root.xpath('//tei:text/tei:body', namespaces=namespaces)
            if not body:
                logger.warning("未找到正文内容")
                return None
            body = body[0]

            # 只提取introduction和conclusion章节
            target_sections = ['introduction', 'conclusion']
            # 遍历所有一级章节（div）
            for div in body.xpath('./tei:div', namespaces=namespaces):
                # 获取章节标题
                head_el = div.xpath('./tei:head', namespaces=namespaces)
                section_title = ""
                if head_el:
                    section_title = " ".join([h.text.strip() for h in head_el if h.text])

                # 检查是否为目标章节
                if not any(target in section_title.lower() for target in target_sections):
                    continue  # 跳过非目标章节

                structured_text.append(f"**{section_title}**")

                # 提取段落内容
                paragraphs = div.xpath('./tei:p', namespaces=namespaces)
                for p in paragraphs:
                    paragraph_text = "".join(p.itertext()).strip()
                    if paragraph_text:
                        cleaned = self._clean_text(paragraph_text)
                        if cleaned:
                            structured_text.append(cleaned)            

            return "\n".join(structured_text)

        except Exception as e:
            logger.error(f"从TEI XML提取文本失败: {e}")
            return None


    def _extract_structured_text_from_tei_skip_section(self, tei_xml: str) -> Optional[str]:
        """从TEI XML提取结构化文本（包含标题、作者、摘要和正文层级结构）"""
        try:
            from lxml import etree

            # 解析XML
            root = etree.fromstring(tei_xml.encode('utf-8'))
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            structured_text = []

            # 提取标题
            title_el = root.xpath('//tei:titleStmt/tei:title', namespaces=namespaces)
            if title_el and title_el[0].text:
                title = title_el[0].text.strip()
                structured_text.append(f"# {title}\n")

            # 提取作者
            authors_el = root.xpath('//tei:titleStmt/tei:author', namespaces=namespaces)
            authors = []
            for a in authors_el:
                if a.text:
                    authors.append(a.text.strip())
            if authors:
                structured_text.append(f"**Authors**: {', '.join(authors)}\n")

            # 提取摘要
            abstract_el = root.xpath('//tei:profileDesc/tei:abstract', namespaces=namespaces)
            if abstract_el:
                abstract_text = " ".join(abstract_el[0].itertext()).strip()
                structured_text.append("**Abstract**:\n" + self._clean_text(abstract_text) + "\n")

            # 提取正文内容
            body = root.xpath('//tei:text/tei:body', namespaces=namespaces)
            if not body:
                logger.warning("未找到正文内容")
                return None
            body = body[0]

            # 跳过的章节关键词
            skip_section_keywords = [
                "references", "related work", "experiment", "experiments",
                "appendix", "appendices", "bibliography", "acknowledgments",
                "acknowledgements", "performance analysis"
            ]

            # 遍历顶层章节并递归处理子章节
            for div in body.xpath('./tei:div', namespaces=namespaces):
                structured_text.extend(
                    self._process_div(div, namespaces, skip_section_keywords, level=1)
                )

            return "\n".join(structured_text)

        except Exception as e:
            logger.error(f"从TEI XML提取文本失败: {e}")
            return None


    def _process_div(self, div, namespaces, skip_section_keywords, level=1) -> list[str]:
        """递归处理章节及其子章节"""
        structured = []

        # 提取章节标题
        head_elements = div.xpath('./tei:head', namespaces=namespaces)
        section_title = ""
        if head_elements:
            section_title = " ".join([head.text.strip() for head in head_elements if head.text])

        # 跳过整章节
        if any(keyword.lower() in section_title.lower() for keyword in skip_section_keywords):
            logger.info(f"跳过章节: {section_title}")
            return []

        # 添加章节标题，使用 Markdown 格式（支持层级嵌套）
        if section_title:
            structured.append(f"\n{'*' * (level + 1)} {section_title}\n")

        # 添加段落内容（只处理当前层的段落）
        paragraphs = div.xpath('./tei:p', namespaces=namespaces)
        for p in paragraphs:
            paragraph_text = "".join(p.itertext())
            if not paragraph_text:
                continue
            cleaned = self._clean_text(paragraph_text)
            if cleaned:
                structured.append(cleaned)

        # 递归处理子章节
        sub_divs = div.xpath('./tei:div', namespaces=namespaces)
        for sub_div in sub_divs:
            structured.extend(
                self._process_div(sub_div, namespaces, skip_section_keywords, level=level + 1)
            )

        return structured


    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = " ".join(text.split())
        
        # 移除特殊字符和格式
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # 移除过短的文本
        if len(text.strip()) < 10:
            return ""
        
        return text.strip()
    
    def download_and_convert_pdf(self, paper: Dict, 
                                pdf_dir: str = "papers/pdfs", 
                                text_dir: str = "papers/texts",
                                save_tei: bool = True) -> Dict[str, Optional[str]]:
        """
        下载PDF并转换为文本
        
        Args:
            paper: 论文信息
            pdf_dir: PDF输出目录
            text_dir: 文本输出目录
            save_tei: 是否保存TEI XML文件
            
        Returns:
            包含PDF、文本和TEI XML文件路径的字典
        """
        result = {
            'pdf_path': None,
            'text_path': None,
            'tei_path': None,
            'success': False
        }
        
        # 下载PDF
        if self.download_pdf(paper, pdf_dir):
            pdf_filename = f"{paper['arxiv_id']}.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            result['pdf_path'] = pdf_path
            
            # 转换为文本和TEI XML
            text_path = self.pdf_to_text(pdf_path, text_dir, save_tei=save_tei)
            result['text_path'] = text_path
            
            # 获取TEI XML文件路径
            if save_tei and text_path:
                tei_dir = os.path.join(text_dir, "tei_xml")
                tei_filename = f"{paper['arxiv_id']}.xml"
                tei_path = os.path.join(tei_dir, tei_filename)
                if os.path.exists(tei_path):
                    result['tei_path'] = tei_path
            
            result['success'] = text_path is not None
        
        return result

    def extract_authors_from_tei(self, tei_xml: str) -> Dict[str, any]:
        """
        从TEI XML中提取作者信息
        
        Args:
            tei_xml: TEI XML字符串
            
        Returns:
            包含作者信息的字典
        """
        try:
            from lxml import etree
            
            # 解析XML
            root = etree.fromstring(tei_xml.encode('utf-8'))
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            authors_info = {
                'authors': [],
                'affiliations': [],
                'emails': [],
                'total_count': 0,
                'corresponding_author': None
            }
            
            # 提取作者信息
            authors_el = root.xpath('//tei:titleStmt/tei:author', namespaces=namespaces)
            
            for author_el in authors_el:
                author_info = self._extract_single_author(author_el, namespaces)
                if author_info:
                    authors_info['authors'].append(author_info)
                    authors_info['total_count'] += 1
                    
                    # 检查是否为通讯作者
                    if author_info.get('is_corresponding', False):
                        authors_info['corresponding_author'] = author_info
            
            # 提取机构信息
            affiliations_el = root.xpath('//tei:profileDesc/tei:particDesc/tei:listOrg/tei:org', namespaces=namespaces)
            for aff_el in affiliations_el:
                aff_info = self._extract_affiliation(aff_el, namespaces)
                if aff_info:
                    authors_info['affiliations'].append(aff_info)
            
            # 如果没有找到机构信息，尝试从作者信息中提取
            if not authors_info['affiliations']:
                for author in authors_info['authors']:
                    if author.get('affiliation'):
                        authors_info['affiliations'].append({
                            'name': author['affiliation'],
                            'type': 'extracted_from_author'
                        })
            
            logger.info(f"提取到 {authors_info['total_count']} 位作者信息")
            return authors_info
            
        except Exception as e:
            logger.error(f"提取作者信息失败: {e}")
            return {
                'authors': [],
                'affiliations': [],
                'emails': [],
                'total_count': 0,
                'corresponding_author': None
            }
    
    def _extract_single_author(self, author_el, namespaces) -> Optional[Dict]:
        """提取单个作者信息"""
        try:
            author_info = {
                'name': '',
                'first_name': '',
                'last_name': '',
                'affiliation': '',
                'email': '',
                'is_corresponding': False,
                'orcid': '',
                'position': ''
            }
            
            # 提取姓名
            name_text = " ".join(author_el.itertext()).strip()
            if name_text:
                author_info['name'] = name_text
                
                # 尝试分离姓名
                name_parts = name_text.split()
                if len(name_parts) >= 2:
                    author_info['first_name'] = name_parts[0]
                    author_info['last_name'] = name_parts[-1]
                elif len(name_parts) == 1:
                    author_info['last_name'] = name_parts[0]
            
            # 提取邮箱
            email_el = author_el.xpath('.//tei:email', namespaces=namespaces)
            if email_el and email_el[0].text:
                author_info['email'] = email_el[0].text.strip()
            
            # 提取机构信息
            affiliation_el = author_el.xpath('.//tei:affiliation', namespaces=namespaces)
            if affiliation_el:
                aff_text = " ".join(affiliation_el[0].itertext()).strip()
                if aff_text:
                    author_info['affiliation'] = aff_text
            
            # 检查是否为通讯作者
            if author_el.get('role') == 'corresp' or 'corresponding' in name_text.lower():
                author_info['is_corresponding'] = True
            
            # 提取ORCID
            orcid_el = author_el.xpath('.//tei:idno[@type="orcid"]', namespaces=namespaces)
            if orcid_el and orcid_el[0].text:
                author_info['orcid'] = orcid_el[0].text.strip()
            
            return author_info
            
        except Exception as e:
            logger.error(f"提取单个作者信息失败: {e}")
            return None
    
    def _extract_affiliation(self, org_el, namespaces) -> Optional[Dict]:
        """提取机构信息"""
        try:
            affiliation_info = {
                'name': '',
                'type': '',
                'address': '',
                'country': ''
            }
            
            # 提取机构名称
            org_name_el = org_el.xpath('.//tei:orgName', namespaces=namespaces)
            if org_name_el and org_name_el[0].text:
                affiliation_info['name'] = org_name_el[0].text.strip()
            
            # 提取地址信息
            address_el = org_el.xpath('.//tei:address', namespaces=namespaces)
            if address_el:
                address_text = " ".join(address_el[0].itertext()).strip()
                if address_text:
                    affiliation_info['address'] = address_text
            
            # 提取国家信息
            country_el = org_el.xpath('.//tei:country', namespaces=namespaces)
            if country_el and country_el[0].text:
                affiliation_info['country'] = country_el[0].text.strip()
            
            return affiliation_info
            
        except Exception as e:
            logger.error(f"提取机构信息失败: {e}")
            return None
    
    def extract_authors_from_pdf(self, pdf_path: str, grobid_url: str = "http://localhost:8070") -> Optional[Dict]:
        """
        从PDF文件中提取作者信息
        
        Args:
            pdf_path: PDF文件路径
            grobid_url: GROBID服务地址
            
        Returns:
            包含作者信息的字典
        """
        try:
            # 检查GROBID服务
            if not self._check_grobid_service(grobid_url):
                logger.error(f"GROBID服务不可用: {grobid_url}")
                return None
            
            # 转换PDF为TEI XML
            logger.info(f"从PDF提取作者信息: {pdf_path}")
            tei_xml = self._convert_pdf_to_tei(pdf_path, grobid_url)
            
            if not tei_xml:
                logger.error("PDF转TEI XML失败")
                return None
            
            # 提取作者信息
            authors_info = self.extract_authors_from_tei(tei_xml)
            
            # 添加文件信息
            authors_info['source_file'] = pdf_path
            authors_info['extraction_time'] = datetime.now().isoformat()
            
            return authors_info
            
        except Exception as e:
            logger.error(f"从PDF提取作者信息失败: {e}")
            return None


def main():
    """主函数"""
    # 图相关关键词
    graph_keywords = [
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
    ]
    
    # 创建下载器
    downloader = ArxivPaperDownloader()
    
    # 搜索论文
    papers = downloader.search_papers(
        keywords=graph_keywords,
        max_results=200,
        sort_by="submittedDate",
        sort_order="descending"
    )
    
    if not papers:
        logger.error("未找到相关论文")
        return
    
    # 按被引次数排序
    papers_sorted = downloader.sort_by_citations(papers)
    
    # 保存论文信息
    output_file = downloader.save_papers(papers_sorted)
    
    if output_file:
        # 显示前10篇论文
        logger.info("\n前10篇论文:")
        for i, paper in enumerate(papers_sorted[:10], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            print(f"   被引次数: {paper['citation_count']}")
            print(f"   发布时间: {paper['published_date']}")
            print(f"   分类: {', '.join(paper['categories'][:3])}")
    
    # 询问是否下载PDF并转换为文本
    download_pdfs = input("\n是否下载前20篇论文的PDF并转换为文本? (y/n): ").lower().strip()
    
    if download_pdfs == 'y':
        logger.info("开始下载PDF并转换为文本...")
        success_count = 0
        text_success_count = 0
        
        for paper in papers_sorted[:20]:
            result = downloader.download_and_convert_pdf(paper)
            if result['pdf_path']:
                success_count += 1
            if result['text_path']:
                text_success_count += 1
            time.sleep(1)  # 避免请求过于频繁
        
        logger.info(f"处理完成，成功下载 {success_count}/20 篇PDF，成功转换 {text_success_count}/20 篇文本")


if __name__ == "__main__":
    main()