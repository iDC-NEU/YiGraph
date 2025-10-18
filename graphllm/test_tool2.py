"""
增强版验证脚本: 验证 MCP 客户端是否正确接收工具描述和自动生成 input_schema

功能:
1. 自动启动和关闭 MCP 服务器子进程
2. 连接 MCP 客户端并获取所有已注册的工具
3. ⭐ 验证客户端自动生成的 input_schema
4. 验证每个工具的描述信息和 output_schema
5. 在控制台以格式化 JSON 打印每个工具的完整信息（包括 input_schema）
6. 检查 output_schema 是否从预生成文件正确加载
7. 将完整验证结果保存到带时间戳的 JSON 报告

使用方法:
    python validate_mcp_schemas.py
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Set, Any
from datetime import datetime
from pathlib import Path

# 导入你的 MCP 客户端
from mcp_client import GraphMCPClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 配置 ---
SERVER_COMMAND = ["python", "graphllm/smart_qa_system.py"]  # 或 mcp_server.py
EXPECTED_OUTPUT_PROPERTIES: Set[str] = {'algorithm', 'success', 'result', 'error', 'summary'}
SCHEMA_FILE_PATH = Path("graphllm/generated_output_schemas.json")


def check_schema_file_exists() -> bool:
    """检查预生成的 schema 文件是否存在"""
    exists = SCHEMA_FILE_PATH.exists()
    if exists:
        logger.info(f"✅ 找到 schema 文件: {SCHEMA_FILE_PATH.absolute()}")
        try:
            with open(SCHEMA_FILE_PATH, 'r', encoding='utf-8') as f:
                schemas = json.load(f)
            logger.info(f"   文件包含 {len(schemas)} 个工具的 output_schema")
        except Exception as e:
            logger.warning(f"⚠️ 读取 schema 文件失败: {e}")
    else:
        logger.warning(f"⚠️ 未找到 schema 文件: {SCHEMA_FILE_PATH.absolute()}")
        logger.info("   提示: 先运行 MCP Server 以生成 schemas")
    return exists


def validate_input_schema(input_schema: Dict, tool_name: str) -> List[str]:
    """
    验证客户端自动生成的 input_schema 格式
    
    返回: 错误列表
    """
    errors = []
    
    # 检查基本结构
    if not isinstance(input_schema, dict):
        errors.append("input_schema 不是字典类型")
        return errors
    
    # 检查必需的顶层字段
    if 'type' not in input_schema:
        errors.append("input_schema 缺少 'type' 字段")
    elif input_schema['type'] != 'object':
        errors.append(f"input_schema.type 应为 'object'，实际为 '{input_schema['type']}'")
    
    if 'properties' not in input_schema:
        errors.append("input_schema 缺少 'properties' 字段")
        return errors
    
    properties = input_schema['properties']
    if not isinstance(properties, dict):
        errors.append("input_schema.properties 不是字典类型")
        return errors
    
    # ✅ 修改：无参数不再视为错误
    # if len(properties) == 0:
    #     errors.append("input_schema.properties 为空（工具无参数）")
    
    # 验证每个参数的定义
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            errors.append(f"参数 '{param_name}' 的 schema 不是字典类型")
            continue
        
        # ✅ 检查参数是否有 type 或 json_type 定义
        if 'type' not in param_schema and 'json_type' not in param_schema:
            errors.append(f"参数 '{param_name}' 缺少 'type' 或 'json_type' 字段")
        
        # 描述可选，但建议提供
        if 'description' not in param_schema:
            # 降级为警告，不阻止验证通过
            logger.debug(f"参数 '{param_name}' 缺少 'description' 字段（建议提供）")
    
    # 检查 required 字段（如果存在）
    if 'required' in input_schema:
        required = input_schema['required']
        if not isinstance(required, list):
            errors.append("input_schema.required 应为数组类型")
        else:
            # 检查 required 中的参数是否在 properties 中定义
            for req_param in required:
                if req_param not in properties:
                    errors.append(f"required 中的参数 '{req_param}' 未在 properties 中定义")
    
    return errors



async def validate_client_schemas():
    """
    主验证函数: 启动服务器，连接客户端，验证工具描述和 schema
    """
    client = None
    server_process = None
    
    # 输出数据结构
    output_data = {
        "validation_timestamp": datetime.now().isoformat(),
        "schema_file_status": {
            "exists": False,
            "path": str(SCHEMA_FILE_PATH.absolute()),
            "preloaded_output_schema_count": 0
        },
        "validation_summary": {},
        "tools": [],
        "errors": []
    }

    try:
        # ============ 步骤 1: 检查预生成的 output_schema 文件 ============
        logger.info("=" * 80)
        logger.info("📋 步骤 1: 检查预生成的 output_schema 文件")
        logger.info("=" * 80)
        
        schema_file_exists = check_schema_file_exists()
        output_data["schema_file_status"]["exists"] = schema_file_exists
        
        # ============ 步骤 2: 启动 MCP 服务器 ============
        logger.info("\n" + "=" * 80)
        logger.info("🚀 步骤 2: 启动 MCP 服务器")
        logger.info("=" * 80)
        
        logger.info(f"执行命令: {' '.join(SERVER_COMMAND)}")
        server_process = subprocess.Popen(
            SERVER_COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # 等待服务器初始化
        logger.info("⏳ 等待服务器启动 (5秒)...")
        time.sleep(5)
        
        # 检查服务器是否正常运行
        if server_process.poll() is not None:
            stderr_output = server_process.stderr.read()
            raise RuntimeError(f"❌ 服务器启动失败:\n{stderr_output}")
        
        logger.info("✅ 服务器进程已启动")
        
        # ============ 步骤 3: 连接客户端并检查 schema 加载 ============
        logger.info("\n" + "=" * 80)
        logger.info("🔌 步骤 3: 连接 MCP 客户端")
        logger.info("=" * 80)
        
        client = GraphMCPClient()
        await client.connect()
        logger.info("✅ 客户端连接成功")
        
        # 检查客户端是否加载了预生成的 output_schemas
        preloaded_output_count = len(client.output_schemas_cache)
        output_data["schema_file_status"]["preloaded_output_schema_count"] = preloaded_output_count
        
        if preloaded_output_count > 0:
            logger.info(f"✅ 客户端成功加载 {preloaded_output_count} 个预生成的 output_schema")
        else:
            logger.warning("⚠️ 客户端未加载任何预生成的 output_schema")
            logger.info("   将使用默认 output_schema 作为兜底")
        
        # ============ 步骤 4: 获取工具列表 ============
        logger.info("\n" + "=" * 80)
        logger.info("📋 步骤 4: 获取工具列表")
        logger.info("=" * 80)
        
        tools: List[Dict] = await client.list_tools()
        if not tools:
            raise ValueError("❌ 未发现任何工具，请检查服务器注册逻辑")
        
        logger.info(f"✅ 发现 {len(tools)} 个工具")
        
        # ============ 步骤 5: 逐个验证工具 ============
        logger.info("\n" + "=" * 80)
        logger.info("🔍 步骤 5: 开始验证每个工具")
        logger.info("=" * 80)
        
        passed_count = 0
        failed_count = 0
        input_schema_generation_count = 0  # 统计成功生成 input_schema 的工具数
        
        for i, tool in enumerate(tools, 1):
            print(f"\n{'=' * 30} [ 工具 {i}/{len(tools)} ] {'=' * 30}")
            
            name = tool.get('name', '')
            description = tool.get('description', '')
            
            # ⭐ 关键：获取客户端自动生成的 input_schema
            input_schema = client.get_tool_schema(name, 'input')
            
            # ⭐ 关键：获取工具的 output_schema（通过客户端方法）
            output_schema = client.get_tool_schema(name, 'output')
            
            # 构建工具详细信息（包含 input_schema 和 output_schema）
            tool_details = {
                "name": name,
                "description": description,
                "input_schema": input_schema,  # ⭐ 输出客户端生成的 input_schema
                "output_schema": output_schema,
                "schema_sources": {
                    "input_schema": "client_generated",  # input 始终由客户端生成
                    "output_schema": "unknown"
                }
            }
            
            # 判断 output_schema 来源
            if name in client.output_schemas_cache:
                tool_details["schema_sources"]["output_schema"] = "preloaded_file"
            else:
                tool_details["schema_sources"]["output_schema"] = "default_fallback"
            
            # 统计 input_schema 生成情况
            if input_schema and isinstance(input_schema, dict) and input_schema.get('properties'):
                input_schema_generation_count += 1
            
            # ⭐ 在控制台打印工具完整信息（JSON格式，包含 input_schema）
            print(json.dumps(tool_details, indent=2, ensure_ascii=False))
            
            # --- 执行验证逻辑 ---
            errors = []
            warnings = []
            
            # 验证 1: 工具名称
            if not isinstance(name, str) or not name.strip():
                errors.append("name 为空或类型无效")
            
            # 验证 2: 描述信息
            if not isinstance(description, str):
                errors.append("description 类型无效")
            elif len(description.strip()) < 20:
                warnings.append("description 过短（建议至少20字符）")
            
            # ⭐ 验证 3: input_schema 格式和内容（客户端自动生成的）
            input_schema_errors = validate_input_schema(input_schema, name)
            if input_schema_errors:
                errors.extend([f"input_schema: {err}" for err in input_schema_errors])
            else:
                # 检查参数数量
                param_count = len(input_schema.get('properties', {}))
                if param_count == 0:
                    warnings.append("input_schema 无参数定义（工具可能不需要参数）")
                else:
                    logger.debug(f"   input_schema 包含 {param_count} 个参数")
            
            # 验证 4: output_schema 格式
            if not isinstance(output_schema, dict):
                errors.append("output_schema 不是字典类型")
            elif 'properties' not in output_schema:
                errors.append("output_schema 缺少 'properties' 字段")
            else:
                # 验证必需字段
                actual_properties = set(output_schema['properties'].keys())
                missing_props = EXPECTED_OUTPUT_PROPERTIES - actual_properties
                if missing_props:
                    errors.append(
                        f"output_schema 缺少预期字段: {', '.join(missing_props)}"
                    )
            
            # 验证 5: output_schema 来源检查
            if tool_details["schema_sources"]["output_schema"] == "default_fallback":
                warnings.append(
                    "output_schema 使用默认值（未从预生成文件加载）"
                )
            
            # 构建完整的验证结果
            validation_result = {
                **tool_details,
                "validation_status": "passed" if not errors else "failed",
                "errors": errors,
                "warnings": warnings,
                "statistics": {
                    "input_param_count": len(input_schema.get('properties', {})),
                    "input_required_count": len(input_schema.get('required', [])),
                    "output_property_count": len(output_schema.get('properties', {}))
                }
            }
            
            # 记录验证结果
            if errors:
                failed_count += 1
                logger.error(
                    f"❌ 工具 '{name}' 验证失败:\n"
                    f"   错误: {'; '.join(errors)}"
                )
                if warnings:
                    logger.warning(f"   警告: {'; '.join(warnings)}")
            else:
                passed_count += 1
                logger.info(f"✅ 工具 '{name}' 验证通过")
                if warnings:
                    logger.warning(f"   警告: {'; '.join(warnings)}")
            
            output_data["tools"].append(validation_result)
            print(f"{'=' * 75}\n")
        
        # ============ 步骤 6: 生成验证总结 ============
        total_tools = len(tools)
        success_rate = (passed_count / total_tools * 100) if total_tools > 0 else 0
        
        output_data["validation_summary"] = {
            "total_tools": total_tools,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "success_rate": f"{success_rate:.1f}%",
            "input_schema_generation": {
                "generated_count": input_schema_generation_count,
                "generation_rate": f"{(input_schema_generation_count/total_tools*100):.1f}%" if total_tools > 0 else "0%"
            },
            "output_schema_sources": {
                "preloaded_count": sum(
                    1 for t in output_data["tools"] 
                    if t["schema_sources"]["output_schema"] == "preloaded_file"
                ),
                "default_count": sum(
                    1 for t in output_data["tools"] 
                    if t["schema_sources"]["output_schema"] == "default_fallback"
                )
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生错误: {e}", exc_info=True)
        output_data["errors"].append({
            "type": type(e).__name__,
            "message": str(e)
        })
    
    finally:
        # ============ 清理资源 ============
        if client:
            await client.disconnect()
            logger.info("🔌 客户端已断开")
        
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                logger.info("🛑 服务器进程已终止")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ 服务器进程未能及时终止，强制结束")
                server_process.kill()
                server_process.wait()
        
        # ============ 保存验证报告 ============
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_data.get("errors"):
            output_filename = f"schema_validation_error_{timestamp_str}.json"
        else:
            output_filename = f"schema_validation_report_{timestamp_str}.json"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📁 完整验证报告已保存: {output_filename}")
        
        # ============ 打印最终总结 ============
        print("\n" + "=" * 80)
        print("📊 验证总结")
        print("=" * 80)
        
        summary = output_data.get("validation_summary", {})
        if summary:
            print(f"总工具数: {summary.get('total_tools', 0)}")
            print(f"✅ 通过: {summary.get('passed_count', 0)}")
            print(f"❌ 失败: {summary.get('failed_count', 0)}")
            print(f"成功率: {summary.get('success_rate', '0%')}")
            
            print(f"\n📝 Input Schema 生成统计:")
            input_gen = summary.get('input_schema_generation', {})
            print(f"  - 成功生成数量: {input_gen.get('generated_count', 0)}")
            print(f"  - 生成成功率: {input_gen.get('generation_rate', '0%')}")
            
            print(f"\n📋 Output Schema 来源统计:")
            output_sources = summary.get('output_schema_sources', {})
            print(f"  - 从文件加载: {output_sources.get('preloaded_count', 0)}")
            print(f"  - 使用默认: {output_sources.get('default_count', 0)}")
        
        schema_status = output_data.get("schema_file_status", {})
        print(f"\n📄 Schema 文件状态:")
        print(f"  - 文件存在: {'✅ 是' if schema_status.get('exists') else '❌ 否'}")
        print(f"  - 预加载 output_schema 数量: {schema_status.get('preloaded_output_schema_count', 0)}")
        
        if output_data.get("errors"):
            print(f"\n❌ 验证过程中发现错误:")
            for err in output_data["errors"]:
                print(f"  - {err['type']}: {err['message']}")
        
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(validate_client_schemas())
