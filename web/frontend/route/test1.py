def format_string_to_list(input_str):
    """
    将逗号分隔的字符串转换为列表格式的字符串
    
    参数:
    input_str: 逗号分隔的字符串，例如 "name,age"
    
    返回:
    字符串格式的列表表示，例如 "['name','age']"
    """
    # 分割字符串并去除每个元素两端的空格
    items = [item.strip() for item in input_str.split(',') if item.strip()]
    
    # 构建格式化的字符串
    formatted_items = [f"'{item}'" for item in items]
    result = f"[{','.join(formatted_items)}]"
    
    return result


# 更简洁的版本
def format_string_to_list_simple(input_str):
    """
    更简洁的实现版本
    """
    items = [item.strip() for item in input_str.split(',') if item.strip()]
    return f"[{','.join(f'{item!r}' for item in items)}]"


# 测试函数
if __name__ == "__main__":
    # 测试用例1
    test_input1 = "name,age"
    expected_output1 = "['name','age']"
    result1 = format_string_to_list(test_input1)
    print(f"输入: '{test_input1}'")
    print(f"预期输出: {expected_output1}")
    print(f"实际输出: {result1}")
    print(f"测试通过: {result1 == expected_output1}")
    print()
    
    # 测试用例2 - 包含空格
    test_input2 = "name, age, city"
    result2 = format_string_to_list(test_input2)
    print(f"输入: '{test_input2}'")
    print(f"输出: {result2}")
    print()
    
    # 测试用例3 - 单个元素
    test_input3 = "name"
    result3 = format_string_to_list(test_input3)
    print(f"输入: '{test_input3}'")
    print(f"输出: {result3}")
    print()
    
    # 测试用例4 - 使用简洁版本
    test_input4 = "name,age,city,country"
    result4 = format_string_to_list_simple(test_input4)
    print(f"输入: '{test_input4}'")
    print(f"输出(简洁版): {result4}")
    print()
    
    # 测试用例5 - 空字符串或只有逗号的情况
    test_input5 = "name,,age,"
    result5 = format_string_to_list(test_input5)
    print(f"输入: '{test_input5}'")
    print(f"输出: {result5}")