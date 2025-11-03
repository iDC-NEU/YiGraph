import json
import re

def extract_json_from_response(text: str):
    """
    从任意模型输出中提取第一个有效 JSON 对象。
    自动忽略 <think>、``` 等格式。
    """
    cleaned = text.strip()

    # remove markdown mark
    cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned, flags=re.MULTILINE).strip()

    # match {} content
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        raise ValueError(f"❌ 未找到 JSON 内容，原始响应:\n{cleaned}")

    json_str = match.group(0)

    # parse json
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON 解析失败: {e}\n提取内容:\n{json_str}")




def test():

    response_text = """
```python
<think>
Some reasoning...
</think>
```python
    {
    "id": "centrality_importance"
}
```
"""
    result = extract_json_from_response(response_text)
    print(result)




if __name__ == '__main__':

    test()
    