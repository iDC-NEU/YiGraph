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


def parse_openai_json_response(response_text: str, method_name: str = "unknown") -> dict:
    """
    Parse JSON response from OpenAI API, handling errors gracefully.
    
    Args:
        response_text: The response text from OpenAI API
        method_name: Name of the calling method (for error logging)
        
    Returns:
        dict: Parsed JSON object, or error dict if parsing fails
    """
    if not response_text:
        return {"error": "Empty response from OpenAI API"}
    
    # Remove markdown code block markers
    result_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
    result_text = re.sub(r'\s*```$', '', result_text, flags=re.MULTILINE)
    result_text = result_text.strip()
    
    try:
        result_json = json.loads(result_text)
        return result_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {method_name}: {e}")
        print(f"Response text: {result_text}")
        return {"error": "JSONDecodeError", "message": str(e), "raw_response": result_text}



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
    