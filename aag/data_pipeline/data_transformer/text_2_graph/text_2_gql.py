import re
import time

PROMPT_TEMPLATE = """
You are an expert in graph databases and should produce a valid {query_lang} query.
Important: DO NOT invent or output any relationship/edge type names.
Use anonymous/unnamed relationships in the query using the syntax `-[]->` or `-[]-` (i.e., empty bracket for edge).
The schema below only contains vertex/node definitions; relationship types are intentionally omitted.

{schema}

Guidelines:
- Only output the query, no explanation or commentary.
- Do not include any edge labels (no ':EDGE_TYPE' or similar).
- Use anonymous relationships `-[]->` (directed) or `-[]-` (undirected) between nodes.
- Prefer MATCH ... RETURN style (or the closest equivalent in {query_lang}).
- Keep the query concise and executable if possible.

Question: "{question}"

Output:
```{query_lang}
"""

def extract_gql(content: str, verbose: bool = False) -> str | None:
    """
    更稳健的 GQL 提取器：
      - 优先提取 Markdown/三引号代码块内的内容（如果有）
      - 如果代码块存在但开头不是 GQL 关键字，则从第一个关键字处切出
      - 如果没有代码块，则在全文中寻找第一个 GQL 关键字并截取
      - 返回 None 表示未发现可用的 GQL 片段
    """
    if verbose:
        print("=== raw LLM output ===")
        print(content)
        print("======================")

    # 1) 尝试从代码块中提取（优先）
    code_patterns = [
        r"```(?:nGQL|cypher|sparql)?\s*\n([\s\S]*?)```",
        r"```([\s\S]*?)```",
        r'"""([\s\S]*?)"""',
    ]
    for p in code_patterns:
        m = re.search(p, content, re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            # 如果代码块前面有注释性文字，也可保留，但要确保从第一个关键字开始
            kw_m = re.search(r"\b(MATCH|GO|SELECT|CONSTRUCT|WHERE|RETURN|YIELD)\b", code, re.IGNORECASE)
            if kw_m:
                code = code[kw_m.start():].strip()
            # 返回代码块内容（已经从第一个关键字开始）
            if verbose:
                print("=== extracted from code block ===")
                print(code)
                print("=================================")
            # 基本验证：包含至少一个关键字才算有效
            if re.search(r"\b(MATCH|GO|SELECT|CONSTRUCT|WHERE|RETURN|YIELD)\b", code, re.IGNORECASE):
                return code
            # 如果代码块里没有关键字，继续后续策略（不立即返回）

    # 2) 没有合适的代码块，或代码块不包含关键字 —— 在全文中寻找第一个关键字并截取
    full = content.strip()
    kw_m = re.search(r"\b(MATCH|GO|SELECT|CONSTRUCT|WHERE|RETURN|YIELD)\b", full, re.IGNORECASE)
    if kw_m:
        candidate = full[kw_m.start():].strip()
        # 删除大模型常见的前缀句（例如 "Here is the query:"）但不要删掉关键字
        candidate = re.sub(r"^(Here (is|are)|Sure[:,]?)\s*", "", candidate, flags=re.IGNORECASE)
        if verbose:
            print("=== extracted by keyword fallback ===")
            print(candidate)
            print("====================================")
        return candidate

    # 3) 最后兜底：尝试移除明显的自然语言前缀，然后返回剩余（风险较高）
    #    例如模型可能直接返回 "The nGQL query is: MATCH ...", 所以可以删除 up to colon
    m2 = re.search(r":\s*(MATCH|GO|SELECT|CONSTRUCT|WHERE|RETURN|YIELD)\b", full, re.IGNORECASE)
    if m2:
        candidate = full[m2.start() + 1:].strip()
        if verbose:
            print("=== extracted by colon fallback ===")
            print(candidate)
            print("==================================")
        return candidate

    # 若都没找到，返回 None
    if verbose:
        print("No GQL-like fragment found in LLM output.")
    return None


def text_to_gql(question: str, llm, graph_schema, query_lang: str = "nGQL", max_retries: int = 5) -> str:
    """
    将自然语言转换为图查询语句 (nGQL / Cypher / SPARQL)
    支持多轮重试 + 智能提取 + 关键字验证
    """

    for attempt in range(1, max_retries + 1):
        try:
            # 调用大模型
            content = llm.complete(PROMPT_TEMPLATE.format(query_lang=query_lang, schema=graph_schema, question=question))
            print(content)

            gql_query = extract_gql(content)

            if gql_query:
                # print(f"✅ GQL生成成功 (第 {attempt} 次尝试)")
                return gql_query
            else:
                print(f"⚠️ 第 {attempt} 次输出无效，重试中...")

        except Exception as e:
            print(f"❌ 调用失败 (第 {attempt} 次): {e}")

        time.sleep(1.5)  # 防止频繁请求

    # 全部失败
    raise RuntimeError("生成失败：多次尝试后仍未生成有效 GQL 查询。")

if __name__ == '__main__':
    from aag.reasoner.model_deployment import OllamaEnv

    llm = OllamaEnv(llm_mode_name = "llama3:8b")

    question = "Find all persons who work at companies in the technology industry and participated in projects after 2020."

    schema = """
    Graph Schema:
    - Node types:
    * Person
    * Company
    * Project
    """

    gql_query = text_to_gql(question, llm, schema, query_lang="nGQL")

    print(f"Generated nGQL Query:\n{gql_query}")
