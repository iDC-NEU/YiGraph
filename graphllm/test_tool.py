"""
测试代码 - 打印Client端获取的工具列表详细信息
"""
import asyncio
import json
from mcp_client import GraphMCPClient, create_client_and_connect


async def print_tools_detailed():
    """详细打印所有工具的信息"""
    client = await create_client_and_connect()
    
    try:
        # 获取工具列表
        tools = await client.list_tools()
        print(f"🎯 共获取到 {len(tools)} 个工具\n")
        
        for i, tool in enumerate(tools, 1):
            print("=" * 80)
            print(f"🔧 工具 #{i}: {tool['name']}")
            print("=" * 80)
            
            # 打印基本信息
            print(f"📝 描述: {tool['description']}")
            
            # 打印input_schema详细信息
            schema = tool['input_schema']
            print(f"\n📋 Input Schema:")
            print(f"   类型: {schema.get('type', 'N/A')}")
            
            # 处理properties
            properties = schema.get('properties', {})
            if properties:
                print(f"\n   参数详情:")
                required_params = schema.get('required', [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '无描述')
                    default_value = param_info.get('default', '无默认值')
                    
                    # 标记必需参数
                    required_mark = "🔴 [必需]" if param_name in required_params else "🟢 [可选]"
                    
                    print(f"     • {param_name} {required_mark}")
                    print(f"       类型: {param_type}")
                    print(f"       描述: {param_desc}")
                    print(f"       默认值: {default_value}")
                    
                    # 如果有枚举值，打印枚举
                    if 'enum' in param_info:
                        print(f"       可选值: {param_info['enum']}")
                    
                    print()
            else:
                print("   无参数")
            
            # 打印完整的原始schema（JSON格式）
            print(f"\n📄 完整Schema (JSON):")
            print(json.dumps(schema, indent=2, ensure_ascii=False))
            
            print("\n" + "─" * 80 + "\n")
        
        # 统计信息
        print("📊 统计信息:")
        total_params = sum(len(tool['input_schema'].get('properties', {})) for tool in tools)
        required_params = sum(len(tool['input_schema'].get('required', [])) for tool in tools)
        print(f"   工具总数: {len(tools)}")
        print(f"   参数总数: {total_params}")
        print(f"   必需参数: {required_params}")
        print(f"   可选参数: {total_params - required_params}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


async def test_specific_tool():
    """测试特定工具的参数验证功能"""
    client = await create_client_and_connect()
    
    try:
        # 测试initialize_graph工具
        tool_name = "initialize_graph"
        schema = client.get_tool_schema(tool_name)
        
        if schema:
            print(f"\n🎯 测试工具: {tool_name}")
            print(f"📋 Schema: {json.dumps(schema, indent=2, ensure_ascii=False)}")
            
            # 测试参数验证
            test_cases = [
                {
                    "name": "有效参数",
                    "args": {
                        "vertices": [{"vid": "A", "properties": {}}],
                        "edges": [{"src": "A", "dst": "B"}],
                        "directed": True
                    }
                },
                {
                    "name": "缺少必需参数",
                    "args": {
                        "vertices": [{"vid": "A", "properties": {}}]
                        # 缺少edges参数
                    }
                },
                {
                    "name": "类型错误参数",
                    "args": {
                        "vertices": "invalid_string",  # 应该是list
                        "edges": []
                    }
                }
            ]
            
            for test_case in test_cases:
                is_valid, error = client.validate_arguments(tool_name, test_case["args"])
                status = "✅ 通过" if is_valid else f"❌ 失败: {error}"
                print(f"   测试 '{test_case['name']}': {status}")
                
        else:
            print(f"❌ 未找到工具: {tool_name}")
            
    finally:
        await client.disconnect()


async def compare_tool_extraction():
    """对比Server定义和Client提取的结果"""
    client = await create_client_and_connect()
    
    try:
        tools = await client.list_tools()
        
        print("🔄 Server工具 vs Client提取对比")
        print("=" * 60)
        
        for tool in tools:
            print(f"\n🔧 {tool['name']}:")
            print(f"   描述长度: {len(tool['description'])} 字符")
            
            schema = tool['input_schema']
            
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            print(f"   Schema类型: {schema}")
            print(f"   参数数量: {len(properties)}")
            print(f"   必需参数: {required}")
            
            # 显示参数类型分布
            type_count = {}
            for param_info in properties.values():
                param_type = param_info.get('type', 'unknown')
                type_count[param_type] = type_count.get(param_type, 0) + 1
            
            print(f"   参数类型分布: {type_count}")
            
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("🚀 开始测试Client端工具提取功能...\n")
    
    # 运行详细打印
    asyncio.run(print_tools_detailed())
    
    print("\n" + "=" * 80)
    print("🧪 运行参数验证测试...")
    
    # 运行参数验证测试
    asyncio.run(test_specific_tool())
    
    print("\n" + "=" * 80)
    print("📊 运行对比分析...")
    
    # 运行对比分析
    asyncio.run(compare_tool_extraction())
    
    print("\n🎉 所有测试完成！")