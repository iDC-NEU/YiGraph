"""
后处理模板库 - 改进版 (更强的容错性)
"""
import json
import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """模板类型枚举"""
    REQUIRES_PROCESSING = "requires_processing"      # 需要后处理
    NO_PROCESSING = "no_processing"                  # 工具直接返回结果
    OPTIONAL_PROCESSING = "optional_processing"      # 可选处理

@dataclass
class ProcessingTemplate:
    template_id: str
    description: str
    template_type: str
    parameters: Dict[str, str]  # 改名：{"n": "int", "reverse": "bool"}
    code_template: Optional[str] = None
    usage_examples: List[str] = None
    
    def __post_init__(self):
        if self.usage_examples is None:
            self.usage_examples = []

class TemplateLibrary:
    """改进的模板库管理器"""
    
    def __init__(self, templates_file: str = "processing_templates.json"):
        self.templates_file = templates_file
        self.templates: Dict[str, ProcessingTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """从文件加载模板"""
        if os.path.exists(self.templates_file):
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        template = ProcessingTemplate(**item)
                        self.templates[template.template_id] = template
                logger.info(f"✅ 加载了 {len(self.templates)} 个处理模板")
            except Exception as e:
                logger.error(f"❌ 加载模板文件失败: {e}，使用默认模板")
                self._initialize_default_templates()
        else:
            self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """初始化默认模板库 - 改进版"""
        default_templates = [
            # ==================== 需要处理的模板 ====================
            ProcessingTemplate(
                template_id="top_n_ranking",
                description="对字典按值排序并返回前N项",
                template_type="requires_processing",
                parameters=["n", "reverse"],
                code_template="""def process(data):
    n = {n}
    reverse = {reverse}
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=reverse)
    return dict(sorted_items[:n])""",
                usage_examples=[
                    "最重要的5个节点",
                    "Top 10 账户",
                    "排名前3的社区"
                ]
            ),
            
            ProcessingTemplate(
                template_id="threshold_filter",
                description="过滤字典中值大于/小于阈值的项",
                template_type="requires_processing",
                parameters=["threshold", "comparison"],
                code_template="""def process(data):
    threshold = {threshold}
    comparison = "{comparison}"
    
    if not isinstance(data, dict):
        return data
    
    operations = {{
        ">": lambda v: v > threshold,
        "<": lambda v: v < threshold,
        "==": lambda v: v == threshold,
        ">=": lambda v: v >= threshold,
        "<=": lambda v: v <= threshold
    }}
    
    if comparison not in operations:
        return data
    
    filter_func = operations[comparison]
    return {{k: v for k, v in data.items() if filter_func(v)}}""",
                usage_examples=[
                    "PageRank 大于 0.05 的节点",
                    "交易额超过 1000 的边",
                    "度中心性小于 5 的账户"
                ]
            ),
            
            # ==================== 不需要处理的模板 ====================
            ProcessingTemplate(
                template_id="simple_result",
                description="简单结果返回（如路径、距离等，不需要后处理）",
                template_type="no_processing",
                parameters=[],
                code_template=None,
                usage_examples=[
                    "A 到 B 的最短路径",
                    "两个节点间的距离",
                    "图的统计信息"
                ]
            ),
            
            # ==================== 默认模板 ====================
            ProcessingTemplate(
                template_id="default_truncate",
                description="默认处理：字典取前10项，其他原样返回",
                template_type="optional_processing",
                parameters=[],
                code_template="""def process(data):
    if isinstance(data, dict) and len(data) > 10:
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:10])
    return data""",
                usage_examples=[
                    "计算 PageRank (无特殊要求)",
                    "查看图信息"
                ]
            ),
            
            ProcessingTemplate(
                template_id="custom_code",
                description="自定义处理代码（回退选项）",
                template_type="requires_processing",
                parameters=["code"],
                code_template=None,  # 代码由用户提供
                usage_examples=[]
            )
        ]
        
        for template in default_templates:
            self.templates[template.template_id] = template
        
        self._save_templates()
        logger.info(f"✅ 初始化了 {len(default_templates)} 个默认模板")
    
    def _save_templates(self):
        """保存模板到文件"""
        try:
            data = [asdict(t) for t in self.templates.values()]
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 保存模板失败: {e}")
    
    def get_template(self, template_id: str) -> Optional[ProcessingTemplate]:
        """获取指定模板"""
        return self.templates.get(template_id)
    
    def add_template(self, template: ProcessingTemplate) -> bool:
        """添加新模板到库中"""
        if template.template_id in self.templates:
            logger.warning(f"⚠️  模板 {template.template_id} 已存在")
            return False
        
        self.templates[template.template_id] = template
        self._save_templates()
        logger.info(f"✅ 新增模板: {template.template_id}")
        return True
    
    def get_templates_documentation(self) -> str:
        """生成模板文档供 LLM 使用"""
        docs = []
        for template in self.templates.values():
            if template.template_id == "custom_code":
                continue
            
            type_indicator = {
                "requires_processing": "⭐ [需要后处理]",
                "no_processing": "✓ [工具直接返回]",
                "optional_processing": "○ [可选处理]"
            }.get(template.template_type, "")
            
            doc = f"""
### 模板: `{template.template_id}` {type_indicator}
**描述**: {template.description}
**参数**: {', '.join(template.parameters) if template.parameters else '无'}
**使用示例**:
{chr(10).join(f'  - "{ex}"' for ex in template.usage_examples)}
"""
            docs.append(doc)
        return "\n".join(docs)
