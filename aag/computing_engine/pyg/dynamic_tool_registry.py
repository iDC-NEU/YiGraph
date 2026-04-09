"""
PyG (PyTorch Geometric) dynamic tool registry.
Extracted from 注册.txt — PyG-specific sections only.
Shared utilities (GenericToolOutput, _extract_full_description) are imported
from networkx_server to avoid duplication.
"""

import inspect
import logging
import re
from typing import Callable, Dict, Any, Optional
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.nn import conv
import torch.nn as nn
import torch.nn.functional as F

# 共用工具从 networkx_server 导入，避免重复定义
from aag.computing_engine.networkx_server.dynamic_tool_registry import (
    GenericToolOutput,
    _extract_full_description,
)

logger = logging.getLogger(__name__)

# ============================================================================
# PyG 卷积层列表
# ============================================================================
PYG_CONVOLUTIONS = [
    'GCNConv', 'SAGEConv', 'GATConv', 'GATv2Conv', 'TransformerConv',
    'ResGatedGraphConv', 'GravNetConv',
    'SuperGATConv', 'EGConv', 'PDNConv', 'GeneralConv',
    'HypergraphConv', 'LEConv', 'ClusterGCNConv',
    'GENConv', 'FiLMConv', 'MFConv',
    'FeaStConv',
    'PointTransformerConv', 'HeteroConv',
    'MixHopConv', 'SSGConv',
    'ARMAConv', 'SGConv', 'TAGConv'
]

#'GMMConv'因为没有边特征暂时还跑不了；SplineConv 需要额外的 C++/CUDA 扩展库 torch-spline-conv；pyg_ecconv，nnconv需要自定义nn.module；pyg_gatedgraphconv，CGConv，pyg_faconv，pyg_gcn2conv,pyg_dnaconv的 channels 参数与我们的通用包装逻辑发生了冲突
#待定：pyg_gravnetconv需要安装 torch-cluster；
#HGTConv 是用于异构图（Heterogeneous Graph）的卷积层，需要特定的数据格式;HEATConv 是一个专门用于异构图（Heterogeneous Graph）的卷积层;
#PANConv 使用 SparseTensor 来实现高效的图卷积操作，而 SparseTensor 需要 torch-sparse 包的支持
#RGCNConv，RGATConv， SignedConv需要边类型信息，但当前的测试数据没有提供
#SignedConv 图卷积层的特殊输入要求没有被满足
# pyg_hanconv调用工具会有问题调用图基本信息的工具
#pyg_chebconv需要特殊参数

# ============================================================================
# PyG 数据转换器
# ============================================================================
class GraphDataConverter:
    """NetworkX ↔ PyG 数据格式转换器"""

    @staticmethod
    def networkx_to_pyg(G: 'nx.Graph', node_features: Optional[torch.Tensor] = None) -> 'pyg.data.Data':
        """
        将 NetworkX 图转换为 PyG Data 对象

        Args:
            G: NetworkX 图
            node_features: 节点特征矩阵 [num_nodes, num_features]
                          如果为 None，则使用节点度作为特征

        Returns:
            PyG Data 对象
        """
        import networkx as nx
        from torch_geometric.utils import from_networkx

        # 使用 PyG 内置转换
        data = from_networkx(G)

        # 如果没有提供节点特征，使用度中心性
        if node_features is None:
            degrees = dict(G.degree())
            node_ids = sorted(G.nodes())
            node_features = torch.tensor(
                [[degrees[node]] for node in node_ids],
                dtype=torch.float
            )

        data.x = node_features

        return data

    @staticmethod
    def pyg_to_networkx(data: 'pyg.data.Data') -> 'nx.Graph':
        """PyG Data → NetworkX Graph"""
        from torch_geometric.utils import to_networkx
        return to_networkx(data, to_undirected=True)

# ============================================================================
# PyG 工具包装器
# ============================================================================
def _extract_pyg_shapes_info(conv_class) -> Dict[str, str]:
    """
    ✅ 精确匹配 PyG 的实际 Shapes 格式

    实际格式示例：
    Shapes:
        - **input:**
          node features :math:`...` or
          ...
        - **output:** node features :math:`...`
    """
    doc = inspect.getdoc(conv_class)
    if not doc:
        logger.warning(f"⚠️ {conv_class.__name__} 没有 docstring")
        return {}

    result = {}

    # ✅ 步骤1：提取整个 Shapes 部分
    shapes_match = re.search(
        r"Shapes:\s*\n(.*?)(?=\n\s{0,4}[A-Z][a-z]+:|$)",
        doc,
        re.DOTALL | re.IGNORECASE
    )

    if not shapes_match:
        logger.warning(f"⚠️ [{conv_class.__name__}] 未找到 Shapes 部分")
        return {}

    shapes_text = shapes_match.group(1)
    logger.debug(f"📄 [{conv_class.__name__}] Shapes 原文:\n{shapes_text[:300]}...")

    # ✅ 步骤2：提取 input
    input_match = re.search(
        r"-\s*\*\*input:\*\*\s*\n\s+(.*?)(?=\n\s*-\s*\*\*|$)",
        shapes_text,
        re.DOTALL
    )

    if input_match:
        input_text = input_match.group(1).strip()
        input_text = re.sub(r'\s+', ' ', input_text)
        result['input'] = input_text
        logger.info(f"✅ [{conv_class.__name__}] 提取到 input: {input_text[:100]}...")
    else:
        logger.warning(f"⚠️ [{conv_class.__name__}] 未找到 input 信息")

    # ✅ 步骤3：提取 output
    output_match = re.search(
        r"-\s*\*\*output:\*\*\s+(.*?)(?=\n\s*-\s*\*\*|$)",
        shapes_text,
        re.DOTALL
    )

    if output_match:
        output_text = output_match.group(1).strip()
        output_text = re.sub(r'\s+', ' ', output_text)
        result['output'] = output_text
        logger.info(f"✅ [{conv_class.__name__}] 提取到 output: {output_text[:100]}...")
    else:
        logger.warning(f"⚠️ [{conv_class.__name__}] 未找到 output 信息")

    return result


def _extract_pyg_param_description(conv_class, param_name: str) -> Dict[str, str]:
    """从 docstring 提取参数描述和类型"""
    doc = inspect.getdoc(conv_class)
    if not doc:
        return {"type": "", "description": ""}

    # 提取 Args 部分
    args_match = re.search(
        r"Args:\s*\n(.*?)(?=\n\s{0,4}[A-Z][a-z]+:|$)",
        doc,
        re.DOTALL | re.IGNORECASE
    )

    if not args_match:
        return {"type": "", "description": ""}

    args_text = args_match.group(1)

    # 匹配参数：param_name (type, optional): description
    param_pattern = rf"{re.escape(param_name)}\s*\(([^)]+)\):\s*(.*?)(?=\n\s{{0,8}}[a-z_]+\s*\(|$)"
    param_match = re.search(param_pattern, args_text, re.DOTALL | re.IGNORECASE)

    if param_match:
        param_type = param_match.group(1).strip()
        description = param_match.group(2).strip()
        description = re.sub(r'\s+', ' ', description)
        return {"type": param_type, "description": description}

    return {"type": "", "description": ""}


def _extract_all_pyg_params_from_docstring(conv_class) -> Dict[str, Dict[str, str]]:
    """从 docstring 提取所有参数"""
    doc = inspect.getdoc(conv_class)
    if not doc:
        return {}

    args_match = re.search(
        r"Args:\s*\n(.*?)(?=\n\s{0,4}[A-Z][a-z]+:|$)",
        doc,
        re.DOTALL | re.IGNORECASE
    )

    if not args_match:
        return {}

    args_text = args_match.group(1)
    params = {}

    # 匹配所有参数
    param_pattern = r"([a-z_]+)\s*\(([^)]+)\):\s*(.*?)(?=\n\s{0,8}[a-z_]+\s*\(|$)"

    for match in re.finditer(param_pattern, args_text, re.DOTALL | re.IGNORECASE):
        param_name = match.group(1).strip()
        param_type = match.group(2).strip()
        description = match.group(3).strip()
        description = re.sub(r'\s+', ' ', description)

        params[param_name] = {
            "type": param_type,
            "description": description
        }

    return params


def generate_pyg_input_schema(conv_name: str) -> Dict[str, Any]:
    """
    ✅ 完全从 docstring 自动生成 PyG 卷积层的 input schema
    """
    conv_class = getattr(conv, conv_name)

    # 1. 从 Args 提取参数
    params_info = _extract_all_pyg_params_from_docstring(conv_class)

    if not params_info:
        logger.warning(f"⚠️ 无法从 {conv_name} docstring 提取参数")
        return {"type": "object", "parameters": {}}

    parameters = {}
    required = []

    # 构建参数 schema
    for param_name, param_info in params_info.items():
        param_schema = {
            "description": param_info["description"] if param_info["description"] else f"Layer parameter: {param_name}",
        }

        # 保留原始类型信息
        if param_info["type"]:
            param_schema["type"] = param_info["type"]

        # 检查是否为必需参数（不包含 optional）
        if "optional" not in param_info["type"].lower():
            required.append(param_name)

        parameters[param_name] = param_schema

    # ⭐ 2. 从 Shapes 提取 input 信息
    shapes_info = _extract_pyg_shapes_info(conv_class)

    if shapes_info.get('input'):
        logger.info(f"✅ [{conv_name}] 正在添加 _input_shapes 参数...")
        parameters['_input_shapes'] = {
            "type": "string",
            "description": f"Expected input shapes: {shapes_info['input']}"
        }
        logger.info(f"✅ [{conv_name}] _input_shapes 已成功添加")
    else:
        logger.warning(f"⚠️ [{conv_name}] 没有找到 Shapes input 信息")

    # 3. 构建最终 schema
    schema = {
        "type": "object",
        "parameters": parameters
    }

    if required:
        schema["required"] = required

    logger.info(f"📋 [{conv_name}] 最终 schema 包含 {len(parameters)} 个参数: {list(parameters.keys())}")

    return schema


def generate_pyg_output_schema(conv_name: str) -> Dict[str, Any]:
    """
    ✅ 从 Shapes 提取 output 信息
    """
    conv_class = getattr(conv, conv_name)
    shapes_info = _extract_pyg_shapes_info(conv_class)

    if shapes_info.get('output'):
        return {
            "type": "string",
            "description": f"Output shape: {shapes_info['output']}"
        }
    else:
        return {
            "type": "string",
            "description": "Transformed node features"
        }


def _create_pyg_tool_function(
    conv_name: str,
    processor_getter: Callable,
    post_processing_decorator: Callable
) -> Callable:
    """为 PyG 卷积层创建工具函数"""

    conv_class = getattr(conv, conv_name)

    # 1. 解析卷积层的参数
    init_sig = inspect.signature(conv_class.__init__)
    conv_params = []
    conv_defaults = {}

    for param_name, param in init_sig.parameters.items():
        if param_name in ['self', 'in_channels', 'out_channels']:
            continue
        conv_params.append(param_name)
        if param.default != inspect.Parameter.empty:
            conv_defaults[param_name] = param.default

    # 2. 动态构建参数字符串（避免使用类对象的 repr）
    param_defs = [
        "in_channels: int = -1",
        "out_channels: int = 16",
        "mode: str = 'train'",
        "node_features: Optional[list] = None",
        "epochs: int = 100",
        "lr: float = 0.01"
    ]

    # 为每个卷积层参数添加定义
    for param_name in conv_params:
        param_defs.append(f"{param_name}: Any = None")

    # 3. 构造函数代码
    param_list = ', '.join(conv_params)
    func_code = f"""
@post_processing_decorator
def pyg_tool_wrapper({', '.join(param_defs)}) -> str:
    # 收集卷积层参数，使用预存的默认值
    conv_kwargs = {{}}
    for param_name in [{', '.join([f"'{p}'" for p in conv_params])}]:
        # 从局部变量中获取参数值
        param_value = locals()[param_name]
        if param_value is not None:
            conv_kwargs[param_name] = param_value
        elif param_name in conv_defaults:
            conv_kwargs[param_name] = conv_defaults[param_name]

    # 收集所有参数
    all_kwargs = {{
        'in_channels': in_channels,
        'out_channels': out_channels,
        'mode': mode,
        'node_features': node_features,
        'epochs': epochs,
        'lr': lr,
        **conv_kwargs
    }}

    return _execute_pyg_conv('{conv_name}', processor_getter, all_kwargs)
"""

    # 4. 执行函数定义
    namespace = {
        'post_processing_decorator': post_processing_decorator,
        'processor_getter': processor_getter,
        '_execute_pyg_conv': _execute_pyg_conv,
        'Optional': Optional,
        'Any': Any,
        'conv_defaults': conv_defaults,
        'torch': torch,
        'nn': nn,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid,
        'Identity': nn.Identity,
        'Linear': nn.Linear,
        'Dropout': nn.Dropout,
        'BatchNorm1d': nn.BatchNorm1d,
        'LayerNorm': nn.LayerNorm,
    }

    exec(func_code, namespace)
    pyg_tool_wrapper = namespace['pyg_tool_wrapper']

    pyg_tool_wrapper.__doc__ = f"{conv_name} - PyTorch Geometric layer"
    return pyg_tool_wrapper


def _execute_pyg_conv(conv_name: str, processor_getter: Callable, kwargs: dict) -> str:
    """通用的 PyG 卷积层执行函数"""
    try:
        conv_class = getattr(conv, conv_name)
        processor = processor_getter()

        if processor.graph is None:
            return GenericToolOutput(
                algorithm=f"pyg_{conv_name}",
                success=False,
                error="Graph not initialized",
                summary="Call 'initialize_graph' first"
            ).model_dump_json()

        logger.info(f"📥 {conv_name} 收到参数: {kwargs}")

        # 提取参数
        mode = kwargs.pop('mode')
        node_features = kwargs.pop('node_features')
        epochs = kwargs.pop('epochs')
        lr = kwargs.pop('lr')
        in_channels = kwargs.pop('in_channels')
        out_channels = kwargs.pop('out_channels')

        # ✅ 关键修复：强制自动推断
        if node_features is None and in_channels != -1:
            logger.warning(
                f"⚠️ 未提供 node_features 但 in_channels={in_channels}，"
                f"强制设置为 -1 自动推断"
            )
            in_channels = -1

        # 数据转换
        if node_features is not None:
            node_features = torch.tensor(node_features, dtype=torch.float)

        pyg_data = GraphDataConverter.networkx_to_pyg(processor.graph, node_features)

        # 自动推断
        if in_channels == -1:
            in_channels = pyg_data.x.size(1)
            logger.info(f"✅ 自动推断 in_channels = {in_channels}")

        # 验证维度匹配
        if in_channels != pyg_data.x.size(1):
            return GenericToolOutput(
                algorithm=f"pyg_{conv_name}",
                success=False,
                error=f"Dimension mismatch: in_channels={in_channels} but features have {pyg_data.x.size(1)} dimensions",
                summary="Please set in_channels=-1 or provide matching node_features"
            ).model_dump_json()

        # 过滤有效参数
        init_sig = inspect.signature(conv_class.__init__)
        valid_params = {
            k: v for k, v in kwargs.items()
            if k in init_sig.parameters and v is not None
        }

        logger.info(f"🏗️ {conv_name}(in={in_channels}, out={out_channels}, {valid_params})")

        model = conv_class(in_channels, out_channels, **valid_params)

        # 执行训练或推理
        if mode == 'train':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_history = []

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = model(pyg_data.x, pyg_data.edge_index)
                loss = torch.nn.functional.mse_loss(out, pyg_data.x)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                embeddings = model(pyg_data.x, pyg_data.edge_index)

            result = {
                "mode": "train",
                "node_embeddings": embeddings.tolist(),
                "training_loss": loss_history,
                "final_loss": loss_history[-1] if loss_history else 0.0
            }
        else:
            model.eval()
            with torch.no_grad():
                embeddings = model(pyg_data.x, pyg_data.edge_index)

            result = {
                "mode": "infer",
                "node_embeddings": embeddings.tolist()
            }

        return GenericToolOutput(
            algorithm=f"pyg_{conv_name}",
            success=True,
            result=result,
            summary=f"{conv_name} executed successfully"
        ).model_dump_json()

    except Exception as e:
        logger.error(f"❌ {conv_name} error: {e}", exc_info=True)
        return GenericToolOutput(
            algorithm=f"pyg_{conv_name}",
            success=False,
            error=str(e),
            summary=f"{conv_name} failed"
        ).model_dump_json()


# ============================================================================
# PyG 工具注册函数
# ============================================================================
def register_pyg_tools(
    mcp,
    processor_getter: Callable,
    post_processing_decorator: Callable
) -> int:
    """
    ✅ 改进版：注册所有 PyG 卷积层，完全基于 docstring 生成 schema
    """
    logger.info("🚀 开始注册 PyG 卷积层 (完全基于 docstring)...")
    registered_count = 0

    input_schemas_map = {}
    output_schemas_map = {}

    for conv_name in PYG_CONVOLUTIONS:
        if not hasattr(conv, conv_name):
            logger.warning(f"⚠️ PyG 不包含卷积层: {conv_name}")
            continue

        tool_name = f"run_pyg_{conv_name.lower()}"

        try:
            conv_class = getattr(conv, conv_name)

            # ✅ 从 docstring 生成 schemas
            input_schema = generate_pyg_input_schema(conv_name)
            output_schema = generate_pyg_output_schema(conv_name)

            # 创建工具函数
            tool_func = _create_pyg_tool_function(
                conv_name,
                processor_getter,
                post_processing_decorator
            )

            # 提取完整描述
            description = _extract_full_description(conv_class.__doc__)

            # 注册到 MCP
            if hasattr(mcp, 'tool'):
                try:
                    decorated_func = mcp.tool(
                        name=tool_name,
                        description=description,
                        input_schema=input_schema
                    )(tool_func)
                except TypeError:
                    decorated_func = mcp.tool(
                        name=tool_name,
                        description=description
                    )(tool_func)
                    if hasattr(decorated_func, '__mcp_tool__'):
                        decorated_func.__mcp_tool__['input_schema'] = input_schema

            input_schemas_map[tool_name] = input_schema
            output_schemas_map[tool_name] = output_schema

            registered_count += 1
            logger.info(f"✅ 已注册 PyG 工具: {tool_name} (参数: {len(input_schema.get('parameters', {}))})")

        except Exception as e:
            logger.error(f"❌ 注册失败 '{tool_name}': {e}", exc_info=True)

    logger.info(f"📊 PyG 注册统计: ✅ {registered_count} 成功")

    # 导出 schemas（保存到 pyg/ 目录）
    if input_schemas_map:
        _export_schemas_local(input_schemas_map, "pyg_input")
    if output_schemas_map:
        _export_schemas_local(output_schemas_map, "pyg_output")

    return registered_count


def _export_schemas_local(schemas_map: Dict[str, Dict[str, Any]], schema_type: str):
    """
    ✅ 统一的 schema 导出函数（保存到 pyg/ 目录）

    Args:
        schemas_map: 工具名称到 schema 的映射
        schema_type: "pyg_input" 或 "pyg_output"
    """
    from pathlib import Path
    import json

    try:
        output_file = Path(__file__).parent / f"generated_{schema_type}_schemas.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_map, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 已导出 {len(schemas_map)} 个工具的 {schema_type} schemas")
        logger.info(f"   文件位置: {output_file.absolute()}")

    except Exception as e:
        logger.warning(f"⚠️ 无法导出 {schema_type} schemas: {e}")
