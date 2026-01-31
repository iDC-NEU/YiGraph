"""
依赖解析模块
负责分析上游节点输出 → 当前节点输入的数据转换
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from aag.models.graph_workflow_dag import WorkflowStep, StepOutputItem
from aag.reasoner.model_deployment import Reasoner
from aag.models.task_types import GraphAnalysisType
from aag.utils.data_utils import take_sample

logger = logging.getLogger(__name__)

class DataDependencyType(Enum):
    """依赖类型枚举"""
    NONE = "none"           # 无依赖
    GRAPH = "graph"         # 需要构造子图
    PARAMETER = "parameter" # 超参数依赖
    BOTH = "both"           # 同时需要子图和参数


@dataclass
class SingleDependencyItem:
    """单个依赖的数据项（来自 selected_outputs[i]）"""
    parent_step_id: int
    parent_step_output_id: int
    field_key: str
    field_type: Any
    field_desc: str
    use_as: str                  # "graph" | "parameter"
    value: Any                   # 真实数据：来自 StepOutputItem.value[field_key]
    reason: str                  # LLM 给的解释文本

@dataclass
class DataDependencyInfo:
    """
    一个父节点的整体依赖信息：
    可能包含多个 selected_outputs (multiple SingleDependencyItem)
    """
    parent_step_id: int
    parent_question: str
    dependency_type: DataDependencyType
    items: List[SingleDependencyItem]


class DataDependencyResolver:
    """
    依赖解析器
    
    职责：
    1. 分析上游节点的输出与当前节点的依赖关系
    2. 执行数据转换（子图构造、参数提取）
    3. 管理子图缓存
    """
    
    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner
        self.global_vertices: Optional[List[str]] = None
        self.global_edges: Optional[List[Tuple[str, str]]] = None
        
    def set_global_graph(self, graph_nodes: List[str] , graph_edges: List[Tuple[str, str]]):
        self.global_vertices = graph_nodes
        self.global_edges = graph_edges
    
    def resolve_dependencies(
        self,
        step_id: str,
        step: WorkflowStep,
        alg_des_info: dict,
        data_dependency_parents: List[WorkflowStep]
    ) -> Dict[str, Any]:
        """
            分析并转换上游依赖数据
        """
        if not data_dependency_parents:
            return {
                "graph_dependencies": [],
                "parameter_dependencies": [],
                "graph_input_adapter_result": None,
                "parameter_input_adapter_result": None
            }

        alg_des_doc = None
        if alg_des_info:
            alg_des_doc = (
                f"alg_name:{alg_des_info.get('name')}, running on engine:{alg_des_info.get('engine')}\n"
                f"**Input Parameters**:\n"
                f"```json\n{json.dumps(alg_des_info.get('input_params'), indent=2, ensure_ascii=False)}"
            )

        logger.info(f"📊 [依赖解析] 当前步骤:{step_id} | 上游节点:{len(data_dependency_parents)}")

        # ---------- Step 1：对每个父节点执行 “分类 + 定位” ----------
        dependencies: List[DataDependencyInfo] = []
        parent_step_map = {}
        for parent_step in data_dependency_parents:
            parent_step_map[parent_step.step_id]=parent_step
            if not parent_step.result:
                logger.warning(
                    f"⚠️ 父节点 {parent_step.step_id}: {parent_step.question}  没有 result，跳过依赖分析"
                )
                continue
            
            dep_info = self._classify_and_locate_dependency(
                current_step = step,
                current_algo_desc=alg_des_doc,
                parent_step=parent_step
            )
            dependencies.append(dep_info)

        # Step 2: 把所有依赖项拆分为 graph / parameter 两类
        graph_items: List[SingleDependencyItem] = []
        param_items: List[SingleDependencyItem] = []

        for dep in dependencies:
            for item in dep.items:
                if item.use_as == "graph":
                    graph_items.append(item)
                elif item.use_as == "parameter":
                    param_items.append(item)

        logger.info(
            f"📌 依赖整理完成 | graph={len(graph_items)} | parameter={len(param_items)}"
        ) 

        # Step 3: 根据任务类型执行不同策略
        graph_input_adapter_result = None
        parameter_input_adapter_result = None

        # Case A：当前任务是图算法 —— 需要子图优先
        if step.task_type == GraphAnalysisType.GRAPH_ALGORITHM: 
            logger.info("🧠 当前任务为图算法 → 交给 Reasoner 写转换代码")
            # 处理图依赖 ----
            if graph_items:
                conver_graph_llm_info  = self._convert_graph_dependencies(
                    current_question=step.question,
                    graph_items=graph_items,
                    parent_steps=parent_step_map
                )
                graph_input_adapter_result = conver_graph_llm_info.get("converted_graph")

            # 处理参数依赖 ----
            if param_items:
                logger.info("🔧 [参数依赖] 使用 LLM 生成参数映射代码...")

                # 把参数适配放在一个新的函数里， 新的函数里包着这个调用函数
                # todo: merged_parameters 换一个名字，其实调用这个函数是想说， 根据这些数据依赖项和当前问题的函数输入，做数据依赖项的适配工作。 具体来说： 首先执行self.reasoner.generate_parameter_mapping_code 根据当前的数据依赖项和当前图问题的函数的参数输入，找到适配的，输出适配的结果 {"description":xxxx, "适配字段":{"图算法函数字段1": "数据依赖项字段1"， "图算法函数字段2"：“数据依赖项字段2”}}
                convert_parameter_llm_info = self._convert_parameter_dependencies(
                    current_question=step.question,
                    alg_des_doc=alg_des_doc,
                    param_items=param_items,
                    parent_steps=parent_step_map
                )
                parameter_input_adapter_result = convert_parameter_llm_info.get("mapped_params")
        elif step.task_type == GraphAnalysisType.NUMERIC_ANALYSIS:
            logger.info("🧮 当前任务类型：Numeric Analysis → 返回原始依赖项")
        else:
            logger.info("ℹ️ 非图算法/非数值分析 → 不做适配，只返回依赖项")
        
        return {
            "graph_dependencies": graph_items,
            "parameter_dependencies": param_items,
            "graph_input_adapter_result": graph_input_adapter_result,
            "parameter_input_adapter_result": parameter_input_adapter_result
        }
            
    def _classify_and_locate_dependency(
        self,
        current_step: WorkflowStep,
        current_algo_desc: str,
        parent_step: WorkflowStep
    ) -> DataDependencyInfo:
        """
        使用 Reasoner 判定依赖类型，并精确定位依赖的上游数据项。

        该函数负责：dependency_type 判定 + selected_outputs 数据定位。
        """
        logger.info(
            f"🔍 [依赖分析] 父节点: {parent_step.step_id} | "
            f"{parent_step.question} → 当前节点: {current_step.step_id} | {current_step.question}"
        )

        # Step 1: 调用 LLM 进行依赖判定与定位
        analysis = self.reasoner.analyze_dependency_type_and_locate_dependency_data(
            current_question=current_step.question,
            task_type=current_step.task_type,
            current_algo_desc=current_algo_desc,
            parent_question=parent_step.question,
            parent_outputs_meta=parent_step.get_result_meta()
        )
        logger.info(f"analysis analysis:{analysis}")

        if not analysis:
            logger.error("❌ LLM 返回结果为空，视为无依赖")
            return DataDependencyInfo(
                parent_step_id=parent_step.step_id,
                parent_question=parent_step.question,
                dependency_type=DataDependencyType.NONE,
                items=[]
            )

        # Step 2: 解析依赖类型
        dep_type_str = analysis.get("dependency_type", "none")
        try:
            dep_type = DataDependencyType(dep_type_str)
        except ValueError:
            logger.error(f"⚠️ 无效 dependency_type='{dep_type_str}'，自动回退为 none")
            dep_type = DataDependencyType.NONE

        selected_outputs = analysis.get("selected_outputs", [])

        # Step 3: 遍历父节点输出 —— 找真实数据源 
        parent_outputs: Dict[int, StepOutputItem] = parent_step.result or {}            
        dependency_items = []

        for sel in selected_outputs:
            output_id = sel.get("output_id")
            field_key = sel.get("field_key")
            use_as = sel.get("use_as")
            reason = sel.get("reason", "")

            if output_id is None:
                logger.error("❌ selected_outputs 中缺少 output_id 字段")
                continue

            # 找到对应 StepOutputItem
            match = parent_outputs.get(output_id)
            if match is None:
                logger.error(f"❌ 未找到父节点 output_id={output_id}")
                continue
            
            value_dict = match.value or {}
            # add gjq: 改进字段查找逻辑，支持模糊匹配
            if field_key not in value_dict:
                # 尝试查找相似的字段名（例如 recipients 可能对应 results）
                similar_field = self._find_similar_field(field_key, list(value_dict.keys()))
                if similar_field:
                    logger.warning(
                        f"⚠️ 在 output_id={output_id} 中找不到字段 '{field_key}'，使用相似字段 '{similar_field}' 代替"
                    )
                    field_key = similar_field
                else:
                    logger.error(
                        f"❌ 在 output_id={output_id} 中找不到字段 field_key='{field_key}' | 可选字段:{list(value_dict.keys())}"
                    )
                    continue

            real_value = value_dict[field_key]


            # ---- 获取字段类型/描述 ----
            f_type = None
            f_desc = None
            if match.output_schema and match.output_schema.fields:
                field_schema = match.output_schema.fields.get(field_key)
                if field_schema:
                    f_type = field_schema.type
                    f_desc = field_schema.field_description

            dependency_items.append(
                SingleDependencyItem(
                    parent_step_id=parent_step.step_id,
                    parent_step_output_id=output_id,
                    field_key=field_key,
                    field_type=f_type,
                    field_desc=f_desc,
                    use_as=use_as,
                    value=real_value,
                    reason=reason
                )
            )

        logger.info(
            f"📌 依赖判定完成 | 父节点:{parent_step.step_id} | 类型:{dep_type.value} | "
            f"依赖项数量:{len(dependency_items)}"
        )

        return DataDependencyInfo(
            parent_step_id=parent_step.step_id,
            parent_question=parent_step.question,
            dependency_type=dep_type,
            items=dependency_items,
        )
    


    def _convert_graph_dependencies(
        self,
        current_question: str,
        graph_items: List[SingleDependencyItem],
        parent_steps: Dict[int, WorkflowStep]
    ) -> Dict[str, Any]:
        """
            图依赖转换：
            1. 调用 Reasoner 生成图构造代码（基于 graph_items）
            2. 执行生成的 Python 代码
            3. 返回 {"description": ..., "converted_graph": {...}, "raw_llm_response": ...}

            graph_items: List[SingleDependencyItem]
            parent_steps: { step_id -> WorkflowStep }
        """
        if not graph_items:
            return {
                "description": "No graph items were provided.",
                "converted_graph": None,
                "raw_llm_response": None
            }

        if not self.global_vertices or not self.global_edges:
            error_msg = "全局图数据未初始化，请先加载图数据集。"
            logger.error(f"❌ {error_msg}")
            return {
                "description": error_msg,
                "converted_graph": None,
                "raw_llm_response": None
            }

        # Step 1: 构造传给 LLM 的 dependency_items（带 sample_value）
        dependency_list = []

        for item in graph_items:
            parent_step = parent_steps[item.parent_step_id]
            raw_data = item.value
            sample_value = take_sample(raw_data)
            dependency_list.append({
                "field_key": item.field_key,
                "field_type": item.field_type,
                "field_desc": item.field_desc,
                "sample_value": sample_value,
                "parent_step_id": item.parent_step_id,
                "parent_step_question": parent_step.question,
                "reason": item.reason
            })

        llm_resp = self.reasoner.generate_graph_conversion_code(
            current_question=current_question,
            dependency_items=dependency_list,
        )

        if llm_resp is None:
            return {
                "description": "LLM returned empty graph conversion result.",
                "converted_graph": None,
                "raw_llm_response": None
            }

        # llm_resp 应该包含 {"code": "...", "description": "..."}
        code = llm_resp.get("code")
        description = llm_resp.get("description", "")
        if not code:
            return {
                "description": "LLM did not return valid code.",
                "converted_graph": None,
                "raw_llm_response": llm_resp
            }

        try:
            # 构造上游真实值字典：{ field_key: real_value }
            upstream_values = {i.field_key: i.value for i in graph_items}
            field_keys_in_order = [i.field_key for i in graph_items]

            exec_env = {
                "global_nodes": self.global_vertices,
                "global_edges": self.global_edges
            }
            exec(code, exec_env)

            fn = exec_env.get("transform_graph")
            if not fn or not callable(fn):
                return {
                    "description": "Generated code does not define a valid transform_graph function.",
                    "converted_graph": None,
                    "raw_llm_response": llm_resp
                }

            args = [ upstream_values[k] for k in field_keys_in_order ]
            args += [ self.global_vertices, self.global_edges ]

            converted_graph = fn(*args)
        except Exception as e:
            return {
                "description": f"Error executing generated code: {e}",
                "converted_graph": None,
                "raw_llm_response": llm_resp
            }

        return {
            "description": description,
            "converted_graph": converted_graph,
            "raw_llm_response": llm_resp
        }


    def _convert_parameter_dependencies(
        self,
        current_question: str,
        alg_des_doc: str,
        param_items: List[SingleDependencyItem],
        parent_steps: Dict[int, WorkflowStep]
    ) -> Dict[str, Any]:
        """
        参数依赖适配：
        1. 调用 Reasoner 生成参数映射代码
        2. 执行代码
        3. 返回 {"description":..., "mapped_params": {...}}
        """
        if not param_items:
            return {
                "mapped_params": {},
                "mapping_raw": None,
                "reasoning": None
            }

        # 构造输入
        dependency_list = []
        for item in param_items:
            parent_step = parent_steps[item.parent_step_id]
            raw_data = item.value
            sample_value = take_sample(raw_data)
            dependency_list.append({
                "field_key": item.field_key,
                "field_type": item.field_type,
                "field_desc": item.field_desc,
                "sample_value": sample_value,
                "parent_step_id": item.parent_step_id,
                "parent_step_question": parent_step.question,
                "reason": item.reason
            })

        mapping_result = self.reasoner.map_parameters(
            current_question=current_question,
            current_algo_desc=alg_des_doc,
            dependency_items=dependency_list
        )

        if "mapping" not in mapping_result:
            return {
                "mapped_params": {},
                "mapping_raw": mapping_result,
                "reasoning": mapping_result.get("explanation", "LLM returned unexpected format")
            }

        mapping = mapping_result["mapping"]
        final_params = {}

        for param_name, mp in mapping.items():
            from_field = mp.get("from_field")
            parent_step_id = mp.get("parent_step_id")
            extract_code = mp.get("extract_code")

            # 找父节点值
            matched_item = next(
                (x for x in param_items
                if x.parent_step_id == parent_step_id and x.field_key == from_field),
                None
            )

            if matched_item is None:
                logger.error(f"❌ 参数 {param_name} 的依赖字段找不到真实数据源")
                continue

            real_value = matched_item.value

            # 没有转换代码，直接使用值
            if extract_code is None:
                final_params[param_name] = real_value
                continue

            # 需要执行 extract_code
            exec_env = {"value": real_value, "param_value": None}

            try:
                exec(extract_code, exec_env)
                final_params[param_name] = exec_env["param_value"]
            except Exception as e:
                logger.error(f"❌ 执行 extract_code 失败: {e}\n代码:\n{extract_code}")
                continue

        return {
            "mapped_params": final_params,           # ⭐最终可输入图算法的参数值
            "mapping_raw": mapping_result,           # LLM 原始 mapping 输出
            "reasoning": mapping_result.get("explanation", "")
        }
    
    def _find_similar_field(self, target_field: str, available_fields: List[str]) -> Optional[str]:
        """
        add gjq: 查找相似的字段名
        
        Args:
            target_field: 目标字段名
            available_fields: 可用字段列表
            
        Returns:
            最相似的字段名，如果没有找到返回 None
        """
        if not available_fields:
            return None
        
        target_lower = target_field.lower()
        
        # 1. 精确匹配（忽略大小写）
        for field in available_fields:
            if field.lower() == target_lower:
                return field
        
        # 2. 语义映射（常见的字段名映射）
        semantic_mappings = {
            "recipients": ["results", "entities", "nodes", "accounts"],
            "senders": ["results", "entities", "nodes", "accounts"],
            "accounts": ["results", "entities", "nodes"],
            "transactions": ["results", "edges", "relationships"],
            "neighbors": ["results", "nodes", "entities"],
            "paths": ["results", "routes"],
            "entities": ["results", "nodes"],
        }
        
        # 检查是否有语义映射
        if target_lower in semantic_mappings:
            for candidate in semantic_mappings[target_lower]:
                for field in available_fields:
                    if field.lower() == candidate:
                        return field
        
        # 3. 部分匹配（包含关系）
        for field in available_fields:
            if target_lower in field.lower() or field.lower() in target_lower:
                return field
        
        # 4. 如果有 "results" 字段，作为默认回退
        for field in available_fields:
            if field.lower() == "results":
                return field
        
        return None

