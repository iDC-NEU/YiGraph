from typing import Literal, List
import re
import json
import openai
import os

test_promt = """
Extract relation triplets from the following text.
Output format requirements:
1. One triplet per line
2. Each triplet format: (head_entity,relation,tail_entity)
3. Entities and relations should be separated by English commas
4. If no valid triplets found, output "None"

Input text:
{text}
"""

prompt_template_str = """
#### Process
**Identify Named Entities**: Extract entities based on the given entity types, ensuring they appear in the order they are mentioned in the text.
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text.

Your final response should follow this format:

**Output:**
```json
{{
    "entities": # type: Dict
    {{
        "Entity Type": ["entity_name"]
    }},
    "triples": # type: List
    [
        ["subject", "predicate", "object"]
    ],
}}
```

### Example:
**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
OTHER
PERSON
YEAR
MONTH
DAY
OTHER
QUANTITY
EVENT

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

**Output:**
```json
{{
"entities": {{
    "COMPANY": ["Walmart Inc.", "Sam's Club", "Flipkart Wholesale"],
    "PERSON": ["Sam Walton", "James 'Bud' Walton"],
    "COUNTRY": ["United States", "Canada", "Mexico", "Central America", "India"],
    "CITY": ["Bentonville", "Rogers"],
    "STATE": ["Arkansas"],
    "DATE": ["1962", "October 31, 1969", "October 31, 2022"],
    "ORGANIZATION": ["Delaware General Corporation Law"]
}},
"triples": [
    ["Walmart Inc.", "FOUNDED_BY", "Sam Walton"],
    ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"],
    ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"],
    ["Walmart Inc.", "FOUNDED_IN", "1962"],
    ["Walmart Inc.", "INCORPORATED", "October 31, 1969"],
    ["Sam Walton", "FOUNDED", "Walmart Inc."],
    ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."],
    ["Walmart Inc.", "OWNS", "Sam's Club"],
    ["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."],
    ["Walmart Inc.", "OPERATES_IN", "United States"],
    ["Walmart Inc.", "OPERATES_IN", "Canada"],
    ["Walmart Inc.", "OPERATES_IN", "Mexico"],
    ["Walmart Inc.", "OPERATES_IN", "Central America"],
    ["Walmart Inc.", "OPERATES_IN", "India"]
]
}}
```

### Task:
Your task is to perform Named Entity Recognition (NER) and knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""


prompt_extract_triplest_str = """
#### Process
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text. Each triplet follows this format: `["Subject Entity", "Relation", "Object Entity"]`.

Your final response should follow this format:

**Output:**
```json
{{
    "triples": # type: List
    [
        ["subject", "predicate", "object"]
    ],
}}
```

### Example:

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

The 2022 Kentucky Derby took place on Saturday, May 7, 2022, at Churchill Downs in Louisville, Kentucky. It was the 148th running of the Kentucky Derby, ...

**Output:**
```json
{{
"triples": [
    ["Walmart Inc.", "FOUNDED_BY", "Sam Walton"],
    ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"],
    ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"],
    ["Walmart Inc.", "FOUNDED_IN", "1962"],
    ["Walmart Inc.", "INCORPORATED", "October 31, 1969"],
    ["Sam Walton", "FOUNDED", "Walmart Inc."],
    ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."],
    ["Walmart Inc.", "OWNS", "Sam's Club"],
    ["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."],
    ["Walmart Inc.", "OPERATES_IN", "United States"],
    ["Walmart Inc.", "OPERATES_IN", "Canada"],
    ["Walmart Inc.", "OPERATES_IN", "Mexico"],
    ["Walmart Inc.", "OPERATES_IN", "Central America"],
    ["Walmart Inc.", "OPERATES_IN", "India"]
    ["2022 Kentucky Derby", "TOOK_PLACE_ON", "Saturday, May 7, 2022"],
    ["2022 Kentucky Derby", "TOOK_PLACE_AT", "Churchill Downs"],
    ["Churchill Downs", "LOCATED_IN", "Louisville, Kentucky"],
    ["2022 Kentucky Derby", "IS", "the 148th running of the Kentucky Derby"]
]
}}
```

### Task:
Your task is to perform knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""


prompt_extract_triplest_without_predicate_limitation_str = """
#### Process
**Establish Triplets**: Extract triples from the provided context in the order they appear in the text. Each triple must follow the format: `["Subject Entity", "Relation", "Object Entity"]`. When constructing triples, avoid using pronouns (e.g., it, she, they, this game, this song) as Subject Entity or Object Entity. Instead, resolve pronouns based on the context and replace them with explicit entities. Additionally, the “Relation” should have a specific and clear semantic meaning to clearly reflect the connection between the subject entity and the object entity.  If the context provides time-related, number-related, or location-specific information, ensure it is included in the triples for completeness.

Your final response should follow this format:

**Output:**
```json
{{
    "triples": # type: List
    [
        ["subject", "predicate", "object"]
    ],
}}
```

### Example:

**Input:**
The 2022 and 2023 Citrus Bowl showcased exciting matchups, with the Iowa Hawkeyes defeating the Kentucky Wildcats 20–17 in 2022, and the Purdue Boilermakers dominating the Tennessee Volunteers 63–7 in 2023.

Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

The 2022 Kentucky Derby took place on Saturday, May 7, 2022, at Churchill Downs in Louisville, Kentucky. It was the 148th running of the Kentucky Derby, ...

**Output:**
```json
{{
"triples": [
  ["2022 Citrus Bowl", "featured", "an exciting matchup"],
  ["2023 Citrus Bowl", "featured", "an exciting matchup"],
  ["Iowa Hawkeyes", "defeated with a score of 20-17", "Kentucky Wildcats"],
  ["Iowa Hawkeyes", "won", "2022 Citrus Bowl"],
  ["Purdue Boilermakers", "defeated with a score of 63-7", "Tennessee Volunteers"],
  ["Purdue Boilermakers", "won", "2023 Citrus Bowl"]
  ["Walmart Inc.", "is", "an American multinational retail corporation"],
  ["Walmart Inc.", "operates", "a series of large supermarkets"],
  ["Walmart Inc.", "operates", "discount department stores"],
  ["Walmart Inc.", "operates", "grocery stores in the United States"],
  ["Walmart Inc.", "is headquartered in", "Bentonville, Arkansas"],
  ["Walmart Inc.", "was founded by", "Sam Walton"],
  ["Walmart Inc.", "was founded by", "James 'Bud' Walton"],
  ["Walmart Inc.", "place of founding is", "Rogers, Arkansas"],
  ["Walmart Inc.", "time of founding is", "1962"],
  ["Walmart Inc.", "was incorporated under the", "Delaware General Corporation Law"],
  ["Walmart Inc.", "was incorporated on", "October 31, 1969"],
  ["Walmart Inc.", "owns and operates", "Sam's Club retail warehouses"]
  ["Walmart", "has as of October 31, 2022", "10,586 stores and clubs"],
  ["Walmart", "as of October 31 2022 has stores and clubs in", "24 countries"],
  ["Walmart", "as of October 31 2022 operates under", "46 different names"],
  ["Walmart", "operates under the name", "Walmart in the United States"],
  ["Walmart", "operates under the name", "Walmart in Canada"],
  ["Walmart", "operates under the name", "Walmart de México y Centroamérica in Mexico"],
  ["Walmart", "operates under the name", "Walmart de México y Centroamérica in Central America"],
  ["Walmart", "operates under the name", "Flipkart Wholesale in India"]
  ["The 2022 Kentucky Derby", "took place on", "Saturday, May 7, 2022"],
  ["The 2022 Kentucky Derby", "took place at", "Churchill Downs"],
  ["Churchill Downs", "is located in", "Louisville, Kentucky"],
  ["The 2022 Kentucky Derby", "was the", "148th running of the Kentucky Derby"]
]
}}
```

### Task:
Your task is to perform knowledge graph triplet extraction on the following text.

**Input:**
{context}

**Output:**
"""


prompt_extract_triplest_without_predicate_limitation_chine_str = """
#### Process
**提取三元组**: 从提供的上下文中按文本出现的顺序提取三元组。每个三元组必须遵循以下格式：`["主语实体", "关系", "宾语实体"]`。在构造三元组时，避免使用代词（例如：它、她、他们、这场比赛、这首歌）作为主语实体或宾语实体，而是根据上下文解析代词并用明确的实体替代。此外，“关系”应具有特定且清晰的语义意义，以清楚地反映主语实体与宾语实体之间的联系。如果上下文提供了与时间、数字或位置相关的信息，请确保它们包含在三元组中以保证完整性。

最终的输出格式应如下所示：

**Output:**
```json
{{
    "triples": # 类型: 列表
    [
        ["主语实体", "关系", "宾语实体"]
    ],
}}

### Example:

**Input:**
2022年和2023年的Citrus Bowl展现了激动人心的对决，2022年爱荷华鹰队以20–17击败肯塔基野猫队，2023年普渡锅炉工队以63–7战胜田纳西志愿者队。

沃尔玛公司（前称沃尔玛商店公司）是一家美国跨国零售公司，经营一系列大型超市（又称超级中心）、折扣百货商店和美国的杂货店，总部位于阿肯色州本顿维尔。[10] 该公司由山姆·沃尔顿和詹姆斯“巴德”沃尔顿兄弟于1962年在附近的阿肯色州罗杰斯创立，并于1969年10月31日根据特拉华州普通公司法成立。此外，它还拥有并运营山姆会员店零售仓库。[11][12]

截至2022年10月31日，沃尔玛在24个国家/地区拥有10,586家商店和俱乐部，使用46种不同的名称运营。[2][3][4] 该公司在美国和加拿大使用沃尔玛名称，在墨西哥和中美洲使用Walmart de México y Centroamérica名称，在印度使用Flipkart Wholesale名称。

2022年肯塔基德比赛于2022年5月7日星期六在肯塔基州路易斯维尔的丘吉尔唐斯赛马场举行。这是肯塔基德比的第148次比赛，…

**Output:**
```json
{{
"triples": [
  ["2022年Citrus Bowl", "展现", "一场激动人心的对决"],
  ["2023年Citrus Bowl", "展现", "一次激动人心的对决"],
  ["爱荷华鹰队", "以20-17的比分击败", "肯塔基野猫队"],
  ["爱荷华鹰队", "赢得了", "2022年Citrus Bowl"],
  ["普渡锅炉工队", "以63-7的比分击败", "田纳西志愿者队"],
  ["普渡锅炉工队", "赢得了", "2023年Citrus Bowl"],
  ["沃尔玛公司", "前称是", "前称沃尔玛商店公司"],
  ["沃尔玛公司", "是", "一家美国跨国零售公司"],
  ["沃尔玛公司", "经营", "一系列大型超市"],
  ["沃尔玛公司", "经营", "折扣百货商店"],
  ["沃尔玛公司", "经营", "美国的杂货店"],
  ["沃尔玛公司", "总部位于", "阿肯色州本顿维尔"],
  ["沃尔玛公司", "创立者之一是", "山姆·沃尔顿"],
  ["沃尔玛公司", "创立者之一是", "詹姆斯“巴德”沃尔顿"],
  ["沃尔玛公司", "创立地点是", "阿肯色州罗杰斯"],
  ["沃尔玛公司", "创立时间是", "1962年"],
  ["沃尔玛公司", "成立法律的根据是", "特拉华州普通公司法"],
  ["沃尔玛公司", "成立于", "1969年10月31日"],
  ["沃尔玛公司", "拥有并运营", "山姆会员店零售仓库"],
  ["沃尔玛", "截至2022年10月31日拥有", "10,586家商店和俱乐部"],
  ["沃尔玛", "截至2022年10月31日在以下地区拥有商店和俱乐部", "24个国家"],
  ["沃尔玛", "截至2022年10月31日运营名称使用", "46种不同的名称运营"],
  ["沃尔玛", "在美国使用的运营名称是", "沃尔玛"],
  ["沃尔玛", "在加拿大使用的运营名称是", "沃尔玛"],
  ["沃尔玛", "在墨西哥使用的运营名称是", "Walmart de México y Centroamérica"],
  ["沃尔玛", "在中美洲使用的运营名称是", "Walmart de México y Centroamérica"],
  ["沃尔玛", "在印度使用的运营名称是", "Flipkart Wholesale"],
  ["2022年肯塔基德比", "举办时间", "2022年5月7日星期六"],
  ["2022年肯塔基德比", "举办地点", "丘吉尔唐斯赛马场"],
  ["丘吉尔唐斯赛马场", "位于", "肯塔基州路易斯维尔"],
  ["2022年肯塔基德比", "是", "第148次肯塔基德比赛"]
]
}}
```

### Task:
你的任务是对以下输入的文本进行知识图谱三元组提取。

**Input:**
{context}

**Output:**
"""

prompt_question_unit_template_str = """
#### Process
**Identify Named Entities**: Extract entities based on the given entity types, ensuring they appear in the order they are mentioned in the text.
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text.

Your final response should follow this format:

**Output:**
```json
{{
    "entities": # type: Dict
    {{
        "Entity Type": ["entity_name"]
    }},
    "triples": # type: Dict
    {{
        "context_1":[["subject", "predicate", "object"],["subject", "predicate", "object"]]
        "context_2":[["subject", "predicate", "object"],["subject", "predicate", "object"]]
    }},
}}
```

### Example:
**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
OTHER
PERSON
YEAR
MONTH
DAY
OTHER
QUANTITY
EVENT

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

**Output:**
```json
{{
"entities": {{
    "COMPANY": ["Walmart Inc.", "Sam's Club", "Flipkart Wholesale"],
    "PERSON": ["Sam Walton", "James 'Bud' Walton"],
    "COUNTRY": ["United States", "Canada", "Mexico", "Central America", "India"],
    "CITY": ["Bentonville", "Rogers"],
    "STATE": ["Arkansas"],
    "DATE": ["1962", "October 31, 1969", "October 31, 2022"],
    "ORGANIZATION": ["Delaware General Corporation Law"]
}},
"triples": {{
    "Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]":[["Walmart Inc.", "FOUNDED_BY", "Sam Walton"], ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"], ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"], ["Walmart Inc.", "FOUNDED_IN", "1962"], ["Walmart Inc.", "INCORPORATED", "October 31, 1969"], ["Sam Walton", "FOUNDED", "Walmart Inc."], ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."], ["Walmart Inc.", "OWNS", "Sam's Club"]],
    "As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.": [["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."], ["Walmart Inc.", "OPERATES_IN", "United States"], ["Walmart Inc.", "OPERATES_IN", "Canada"], ["Walmart Inc.", "OPERATES_IN", "Mexico"], ["Walmart Inc.", "OPERATES_IN", "Central America"], ["Walmart Inc.", "OPERATES_IN", "India"]]
}},
}}
```

### Task:
Your task is to perform Named Entity Recognition (NER) and knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""


alignment_prompt = """
#### Task

Perform **Entity Alignment** and **Relation Alignment** across all collections of triplets provided in the nested list. Ensure that identical entities and relationships are standardized consistently across the entire input, not just within individual collections.

- **Entity Alignment**: Identify and merge different expressions of the same entity. Ensure all variations of an entity are unified into a single, standardized name.
- **Relation Alignment**: Identify and merge different expressions of the same relationship. Ensure all variations of a relationship are unified into a single, standardized form.

**Input Format**:
The input is a nested list, where each sub-list is a collection of triplets. Each triplet follows this format:
`["Subject Entity", "Relation", "Object Entity"]`

Example **Input**:
[
    [
        ["International Business Machines Corporation","headquartered_in", "New York"],
        ["IBM", "based_in", "New York"],
        ["International Business Machines Corporation", "owns", "Red Hat"]
    ],
    [
        ["IBM", "acquired", "Red Hat"],
        ["IBM", "has_location", "Armonk"]
    ]
]


**Expected Output Format**:
- Each collection has aligned entities and relationships, ensuring consistency across all collections.
- All variations of the same **entity** are unified into a single standardized name across all collections.
- All variations of the same **relationship** are unified into a single standardized form across all collections.
- **The number of triplets in the output must match the number in the input, and the order of triplets in the output must align exactly with the order in the input.**
- Ensure that the output is **valid JSON** format.

Example **Output**:
```json
{{
"aligned_triples": [
    [
        ["IBM", "headquartered_in", "New York"],
        ["IBM", "headquartered_in", "New York"],
        ["IBM", "owns", "Red Hat"]
    ],
    [
        ["IBM", "owns", "Red Hat"],
        ["IBM", "has_location", "Armonk"]
    ]
]
}}
```


### Task Instructions:
1. Identify different expressions for the same entity (e.g., "IBM" and "International Business Machines Corporation") across all collections, and standardize them to one name.
2. Identify different expressions of the same relationship (e.g., "headquartered_in" and "based_in") across all collections, and standardize them to one form.
3. Return the aligned result as a **valid JSON object** with the key `aligned_triples` containing the aligned triplets, ensuring entities and relationships are consistent across all collections
4. **The output must exactly match the input in terms of the number and order of triplets**. Provide only the final aligned result—no explanation or analysis.

**Input**:
{input_data}

**Output**:
"""


alignment_prompt_chine_str = """
#### Task

对嵌套列表中提供的所有三元组集合执行**实体对齐**和**关系对齐**。确保在整个输入中（不仅限于单个集合内）相同的实体和关系被统一标准化。

- **实体对齐**：识别并合并同一实体的不同表达。确保实体的所有变体都统一为一个标准化名称。
- **关系对齐**：识别并合并同一关系的不同表达。确保关系的所有变体都统一为一个标准化形式。

**Input Format**:
输入是嵌套列表，其中每个子列表都是一个三元组集合。每个三元组都遵循以下格式：
`[“主语实体”，“关系”，“宾语实体”]`

Example **Input**:
[
    [
        ["阿里巴巴集团", "位于", "杭州"],
        ["Alibaba", "总部在", "杭州"],
        ["阿里", "持有", "饿了么"]
    ],
    [
        ["阿里巴巴", "拥有", "饿了么"],
        ["Alibaba", "运营于", "中国"]
    ]
]


**Expected Output Format**:
- 每个集合的实体和关系经过对齐，确保在所有集合中保持一致性。
- 相同的**实体**的所有变体在所有集合中统一为一个标准化名称。
- 相同的**关系**的所有变体在所有集合中统一为一个标准化形式。
- **输出中的三元组数量必须与输入中的数量匹配，并且输出中的三元组顺序必须与输入中的顺序完全一致。**
- 确保输出为**有效的 JSON**格式。

Example **Output**:
```json
{{
"aligned_triples": [
    [
        ["阿里巴巴", "总部在", "杭州"],
        ["阿里巴巴", "总部在", "杭州"],
        ["阿里巴巴", "持有", "饿了么"]
    ],
    [
        ["阿里巴巴", "持有", "饿了么"],
        ["阿里巴巴", "运营于", "中国"]
    ]
]
}}
```


### Task Instructions:
1. 识别同一实体的不同表达方式（例如，“阿里巴巴集团”和“Alibaba”），并在所有集合中统一为一个名称。
2. 识别同一关系的不同表达方式（例如，“位于”和“总部在”），并在所有集合中统一为一个形式。
3. 将对齐结果作为**有效的 JSON 对象**返回，其中键是 `aligned_triples` ，对应包含的是对齐后的三元组，确保实体和关系在所有集合中保持一致。
4. **输出必须在三元组的数量和顺序方面与输入完全匹配**。仅提供最终的对齐结果，不提供解释或分析。

**Input**:
{input_data}

**Output**:
"""


# 指定查询个数
prompt_extract_keywords_str = """
### Task: Extract Keywords

#### Objective:
Your task is to extract **up to {max_keywords} keywords** from the provided question. Focus on extracting **meaningful terms** that we can use to best lookup answers to the question. And **avoid stopwords** (e.g., "the," "is," "at").

#### Response Format:
Your final response should follow this format:

```json
{{
    "keywords": # type: List
    [
        "keyword1",
        "keyword2",
        "keyword3"
    ],
}}
```

#### Example 1:
**Input Question:**
What position did Jason Semore hold at Valdosta State before returning to Georgia Tech?

**max_keywords:** 3

**Output:**
```json
{{
    "keywords": [
        "Jason Semore",
        "Valdosta State",
        "Georgia Tech",
    ]
}}
```

#### Example 2:
**Input Question:**
Which team won the 2024 NBA championship?

**max_keywords:** 2

**Output:**
```json
{{
    "keywords": [
        "2024 NBA championship",
        "team"
    ]
}}
```

#### Instructions:
**Your task is to extract keywords from the following inputs:

**Input Question:**
{input_question}

**max_keywords:**
{max_keywords}

**Output:**

"""


# 没有指定查询个数
prompt_extract_qury_keywords_str = """
### Task: Extract Query Subject

#### Objective:
Your task is to extract the focus query subject and expand the max 2 synonym entities of the query subject based on your knowledge.

#### Response Format:
Your final response should follow this format:

```json
{{
    "keywords": # type: List
    [
        "keyword1",
        "keyword2",
        "keyword3"
    ],
}}
```

#### Example 1:
**Input Question:**
What position did Jason Semore hold at Valdosta State before returning to Georgia Tech?

**Output:**
```json
{{
    "keywords": [
        "Jason Semore",
        "Valdosta State",
        "Georgia Tech",
    ]
}}
```

#### Example 2:
Super Bowl 2022 date and location?

**Output:**
```json
{{
    "keywords": [
        "Super Bowl 2022",
        "Super Bowl LVI",
        "Super Bowl 56"
    ]
}}
```

#### Example 3:
What films won the 2022 and 2023 Academy Awards for Best Picture?

**Output:**
```json
{{
    "keywords": [
        "2022 Academy Awards",
        "2023 Academy Awards",
        "94th Academy Awards",
        "95th Academy Awards",
        "Best Picture"
        "Oscar for Best Picture"
    ]
}}
```

#### Instructions:
**Your task is to extract the focus query subject from the following inputs:

**Input Question:**
{input_question}

**Output:**

"""


prompt_filter_entities_from_entities_collection = """
### Task: Identify Query Entity

#### Objective:
Your task is to select the **query object** thatare most relevant to  the given question from the given entities collection.

Selection Principles:
1.	Relevant Entity Selection:
•Choose entities that appear in the question.
•Include synonyms and sub-entities of these entities.
2.	Exclusion of Irrelevant Entities:
•Do not select time-related entities or location-related entities that do not appear in the question.
•Exclude ambiguous pronoun entities (e.g., “The game,” “A film”).

#### Response Format:
Your final response should follow this format:

```json
{{
    "keywords": # type: List
    [
        "keyword1",
        "keyword2",
        "keyword3"
    ],
}}
```

#### Example 1:
**Input Question:**
Who is the director of 'Carole King & James Taylor: Just Call Out My Name' and when is its premiere?

**Entity Collection:**
[
  'Carole king & james taylor: just call out my name',
  'Carole king & james taylor',
  'Carole king',
  'James taylor',
  'Regina king',
  'Defining voice in music and pop culture since 1952',
  'Lin-manuel miranda',
  'Miranda lambert',
  'Taylor swift',
  'Song of the year',
  'Carrie underwood',
  'Goldie hawn',
  'Beverly hilton',
  'Adele',
  'Jimmy connors',
  'James vanderbilt',
  'Taylor fritz',
  'Luke bryan',
  'Director',
  'Best director',
  'Director and producer',
  'Director and actress',
  'Executive director',
  'Director and playwright',
  'Oscars for best director',
  'Best director motion picture',
  'Best director (the shape of water)',
  'Ceo',
  'Release date',
  'Final release date',
  "Film's release",
  'Late february date',
  'New movie',
  'Just weeks after the successful debut of scream (2022)',
  '8pm et',
  'Oct 20, 2023',
  '7:15 p.m. et thursday',
  'Early 2021',
  'Screenwriter',
  'A film',
  'Full-day coverage of the opening ceremony',
  'Initial reveal',
  "Film's title",
  'Filming',
  'Feature-length film',
  'This two-part exhibition',
  '3d showings',
  'The ceremony'
]

**Output:**
```json
{{
    "keywords": [
        "Carole king & james taylor: just call out my name",
        "Carole king & james taylor",
        "Carole king",
        "James taylor",
        "Director",
        "Executive director",
        "Release date",
        "Final release date"
    ]
}}
```

#### Instructions:
**Your task is to select the query entities from the following inputs:

**Input Question:**
{input_question}

**Entity Collection:**
{entity_collection}

**Output:**

"""


prompt_identify_supporting_triples = """
### Task: Identify Supporting Triples

#### Objective:
Your task is to identify **supporting triples** from the provided **triples collection** that can accurately generate the given **answer** to the **input question**. When presenting the output, group different representations of the **same triple** into a single list. 

#### Response Format:
Your final response should follow this format:

```json
{{
    "supporting_triples": # type: List[List]
    [
        [
            ["Subject", "Predicate", "Object"],
            "Subject -Predicate-> Object"
        ],
        [
            ["Another_Subject", "Another_Predicate", "Another_Object"],
            "Another_Subject -Another_Predicate-> Another_Object"
        ]
    ]
}}
```
#### Example 1:

**Input Question:**
When did Neal Lemlein pass away?

**Answer:**
July 22 2022

**Triples Collection:**
[
    [“Neal”, “Died on”, “July 22, 2022”],
    [“Neal”, “Died of”, “Kidney cancer”],
    [“Neal”, “Was”, “71 years old”],
    ["His death <-Led to- Abe's fabricated tweet"],
    ["Date of death -Is-> July 8, 2022"],
    ["His passing <-Was 39 years old at the time of- Shad gaspard -Passed away on-> May 17, 2020"],
    ["Neal lemlein -Is a friend of-> Gray -Died on-> March 5, 2019"],
    ["Neal -Died on-> July 22, 2022"]
]

**Output::**
```json
{{
    "supporting_triples": # type: List[List]
    [
        ["Neal", "Died on", "July 22, 2022"],
        "Neal -Died on-> July 22, 2022"
    ]
}}
```

#### Example 2:

**Input Question:**
Who finished second in the 2023 Tour de France?

**Answer:**
Tadej Pogacar

**Triples Collection:**
[
    [“Jonas Vingegaard”, “Won”, “The yellow jersey”],
    [“Tadej Pogacar”, “Finished”, “Second place”],
    [“Tadej Pogacar -Finished-> Second place”]
]

**Output::**
```json
{{
    "supporting_triples": # type: List[List]
    [
        [
            ["Tadej Pogacar", "Finished", "Second place"],
            "Tadej Pogacar -Finished-> Second place"
        ]
    ]
}}

#### Instructions:
**Your task is to identify supporting triples from the following inputs:

**Input Question:**
{input_question}

**Answer:**
{answer}

**Triples Collection**:
{triples_collection}

**Output:**

"""


prompt_identify_supporting_triples_separately = """
### Task: Identify Supporting Triples

#### Background:
This task is part of the **GraphRAG framework**, where the goal is to generate **ground-truth triples** for each question. These triples are critical for ensuring the retrieval process accurately supports the given answer. The triples are extracted separately from **from_evidences_triplets** and **from_retrieve_results**.

#### Objective:
Your task is to identify **ground-truth triples** from the provided triples collection that can generate the **given answer** to the **input question**. Extract the relevant triples from both **from_evidences_triplets** and **from_retrieve_results**, ensuring accurate coverage. Only include triples that explicitly support the given answer. Do not include triples that are indirectly related or provide partial context.

#### Response Format:
Your final response should follow this format:

```json
{{
    "supporting_triples": # type: List[Dict]
    [
        {{
            "from_evidences_triplets":
            [
                ["Subject", "Predicate", "Object"],
            ]
        }},
        {{
            "from_retrieve_results":
            [
                "Subject -Predicate-> Object",
                "Subject -Predicate-> Object -Next Predicate ->Next Object"
            ]
        }}
    ]
}}
```
#### Example 1:

**Input Question:**
When did Neal Lemlein pass away?

**Answer:**
July 22 2022

**Triples Collection:**
[
    {{
        "from_evidences_triplets":
        [
            [“Neal”, “Died on”, “July 22, 2022”],
            [“Neal”, “Died in”, “July 22, 2022”],
            [“Neal”, “Died of”, “Kidney cancer”],
            [“Neal”, “Was”, “71 years old”]
        ]
    }},
    {{
        "from_retrieve_results":
        [
            "His death <-Led to- Abe's fabricated tweet",
            "Date of death -Is-> July 8, 2022",
            "His passing <-Was 39 years old at the time of- Shad gaspard -Passed away on-> May 17, 2020",
            "Neal lemlein -Is a friend of-> Gray -Died on-> March 5, 2019",
            "Neal -Died on-> July 22, 2022"
        ]
    }},
]


**Output::**
```json
{{
    "supporting_triples":
    [
        {{
            "from_evidences_triplets": 
            [
                [“Neal”, “Died on”, “July 22, 2022”],
                [“Neal”, “Died in”, “July 22, 2022”]
            ]
        }},
        {{
            "from_retrieve_results":
            [
                "Neal -Died on-> July 22, 2022"
            ]
        }}
    ]
}}
```


#### Example 2:

**Input Question:**
Who won the 2022 Tour de France?

**Answer:**
Jonas vingegaard

**Triples Collection:**
[
    {{
        "from_evidences_triplets":
        [
            ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"],
            ["Jonas vingegaard", "Won", "The yellow jersey"],
            ["Jonas vingegaard", "Won the tour by", "3:34 over tadej pogačar"]
        ]
    }},
    {{
        "from_retrieve_results":
        [
            "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard",
            "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar",
            "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won-> The yellow jersey"
        ]
    }},
]


**Output::**
```json
{{
    "supporting_triples":
    [
        {{
            "from_evidences_triplets": 
            [
                ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"]
            ]
        }},
        {{
            "from_retrieve_results":
            [
                "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard",
                "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar",
                "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won-> The yellow jersey"
            ]
        }}
    ]
}}
```


#### Instructions:
**Using the inputs below, identify **only the ground-truth triples** that directly and explicitly answer the question. Avoid including any triples that are tangentially related or provide only partial context.

**Input Question:**
{input_question}

**Answer:**
{answer}

**Triples Collection**:
{triples_collection}

**Output:**

"""

prompt_identify_supporting_triples_from_evidences_triplets = """
### Task: Identify triplets that support answering questions

#### Background:
This task is part of the **GraphRAG framework**, where the goal is to identify **ground-truth triplets** for input question. These triplets are critical for ensuring the retrieval process accurately supports the given answer. 

#### Objective:
Your task is to identify **ground-truth triplets** from the provided triplets collection that can be used to generate the answer to the given input question. Based on the input question, answer, and triplets collection, identify which triplets can support returning the correct answer to the question.


#### Response Format:
Your final response should follow this format:
```json
{{
    "supporting_triplets": # type: List[List]
    [
        ["Subject", "Predicate", "Object"],
        ["Another subject", "Predicate", "Another object"],
    ]
}}
```

#### Example 1: Type 1 (Direct single-hop answer)
**Input Question:**
Who won the 2022 Tour de France?

**Answer:**
Jonas vingegaard

**Triplets Collection:**
[
    ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"],
    ["Jonas vingegaard", "Won the tour by", "3:34 over tadej pogačar"]
]


**Output::**
```json
{{
    "supporting_triplets":
    [
        ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"]
    ]
}}
```

#### Example 2: Type 2 (Multi-hop reasoning)
**Input Question:**
The King of Hollywood starred in what 1932 American pre-Code dram film?

**Answer:**
Strange Interlude

**Triplets Collection:**
[
    ["Strange interlude", "Is", "A 1932 american pre-code drama film"],
    ["Strange interlude", "Stars", "Clark gable"],
    ["Clark gable",  "Was often referred to as", "The king of hollywood"],
    ["The stage production", "Lasts", "Six hours"],
    ["The stage production", "Is sometimes performed over", "Two evenings"]
]


**Output::**
```json
{{
    "supporting_triplets":
    [
        ["Strange interlude", "Is", "A 1932 american pre-code drama film"],
        ["Strange interlude", "Stars", "Clark gable"],
        ["Clark gable",  "Was often referred to as", "The king of hollywood"],
    ]
}}
```


#### Example 3:  Type 3 (Different expressions of the same fact)
**Input Question:**
When did Neal Lemlein pass away?

**Answer:**
July 22 2022

**Triplets Collection:**
[
    [“Neal”, “Died on”, “July 22, 2022”],
    [“Neal”, “Died in”, “July 22, 2022”],
    [“Neal”, “Died of”, “Kidney cancer”],
    [“Neal”, “Was”, “71 years old”]
]

**Output::**
```json
{{
    "supporting_triplets":
    [
        [“Neal”, “Died on”, “July 22, 2022”],
        [“Neal”, “Died in”, “July 22, 2022”]
    ]
}}
```


#### Instructions:
**Your task is to identify supporting triplets from the following inputs:

**Input Question:**
{input_question}

**Answer:**
{answer}

**Triplets Collection**:
{triplets_collection}

**Output:**

"""

prompt_identify_supporting_triples_from_evidences_triplets_type1_single_hop_triplets = """
### Task: Identify triplets that support answering questions
#### Objective:
Your task is to identify **ground-truth triplets** from the provided triplets collection that can generate the **given answer** to the **input question**. Identify the relevant triplets from **triplets_collection**, ensuring accurate coverage. Only include triplets that explicitly support the given answer. Do not include triplets that are indirectly related or provide partial context.

#### Guidelines:
1. Select the **most relevant single triple** that directly and explicitly answers the question.  Ignore other triplets that provide additional or background information, even if related to the subject.
2. If multiple triplets express the same fact using different words or phrases, include **all such triplets** in the response.
3. Do not include triplets that are only indirectly related or provide incomplete support for the answer.

#### Response Format:
Your final response should follow this format:
```json
{{
    "supporting_triplets": # type: List[List]
    [
        ["Subject", "Predicate", "Object"]
    ]
}}
```

#### Example 1: Type 1  (Explicit single-hop answer, select only the most relevant triple)
**Input Question:**
Who won the 2022 Tour de France?

**Answer:**
Jonas vingegaard

**Triplets Collection:**
[
    ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"],
    ["Jonas vingegaard", "Won", "The yellow jersey"],
    ["Jonas vingegaard", "Won the tour by", "3:34 over tadej pogačar"]
]

**Output::**
```json
{{
    "supporting_triplets":
    [
        ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"]
    ]
}}
```

#### Example 2:  Type 2 (Redundant expressions of the same fact, include all)
**Input Question:**
When did Neal Lemlein pass away?

**Answer:**
July 22 2022

**Triplets Collection:**
[
    [“Neal”, “Died on”, “July 22, 2022”],
    [“Neal”, “Died in”, “July 22, 2022”],
    [“Neal”, “Died of”, “Kidney cancer”],
    [“Neal”, “Was”, “71 years old”]
]

**Output::**
```json
{{
    "supporting_triplets":
    [
        [“Neal”, “Died on”, “July 22, 2022”],
        [“Neal”, “Died in”, “July 22, 2022”]
    ]
}}
```

#### Instructions:
**Using the above guidelines, identify the supporting triplets from the inputs below:

**Input Question:**
{input_question}

**Answer:**
{answer}

**Triplets Collection**:
{triplets_collection}

**Output:**

"""

prompt_identify_supporting_triples_from_evidences_triplets_type2_multi_hop_triplets = """
### Task: Identify Supporting Multihop Triplets for Answering Questions

#### Goal:
The objective of this task is to identify **relevant multihop triplets** from the provided triplets collection that form a reasoning chain to generate the correct answer to the input question. These triplets should enable a step-by-step logical deduction starting from entities in the question.

#### Objective:
Your task is to identify **ground-truth triplets** from the provided triplets collection that can be used to generate the answer to the given input question. Based on the input question, answer, and triplets collection, identify which triplets can support returning the correct answer to the question.
- Analyze the **input question** and the ** answer**.
- Identify and select **ground-truth triplets** from the triplets collection that can establish a valid reasoning chain from the question's entities to the answer.


#### Response Format:
Your final response should follow this format:
```json
{{
    "supporting_triplets": # type: List[List]
    [
        ["Subject", "Predicate", "Object"],
        ["Another subject", "Predicate", "Another object"],
    ]
}}
```

#### Example:
**Input Question:**
The King of Hollywood starred in what 1932 American pre-Code dram film?

**Answer:**
Strange Interlude

**Triplets Collection:**
[
    ["Strange interlude", "Is", "A 1932 american pre-code drama film"],
    ["Clark gable", "Stars", "Strange interlude"],
    ["The king of hollywood", "Was often referred to as", "Clark gable"],
    ["The stage production", "Lasts", "Six hours"],
    ["The stage production", "Is sometimes performed over", "Two evenings"]
]


**Output::**
```json
{{
    "supporting_triplets":
    [
        ["The king of hollywood", "Was often referred to as", "Clark gable"],
        ["Clark gable", "Stars", "Strange interlude"],
        ["Strange interlude", "Is", "A 1932 american pre-code drama film"]
    ]
}}
```

#### Instructions:
**Your task is to identify supporting triplets from the following inputs.

**Input Question:**
{input_question}

**Answer:**
{answer}

**Triplets Collection**:
{triplets_collection}

**Output:**

"""


prompt_identify_supporting_triples_from_retrieve_results = """
### Task: Identify retrieved paths that support answering questions

#### Background:
This task is part of the **GraphRAG framework**, where the goal is to identify **ground-truth retrieved paths** for input question. These retrieved paths are critical for ensuring the retrieval process accurately supports the given answer. 

#### Objective:
Your task is to identify **ground-truth retrieved paths** from the provided retrieved paths collection that can be used to generate the answer to the given input question. Based on the input question, answer, and retrieved paths collection, identify which paths can support returning the correct answer to the question.


#### Response Format:
Your final response should follow this format:

```json
{{
    "supporting_paths": # type: List
    [
        "Subject -Predicate-> Object",
        "Subject -Predicate-> Object -Next Predicate ->Next Object"
    ]
}}
```

#### Example 1:  Type 1 (Different expressions of the same fact)
**Input Question:**
When did Neal Lemlein pass away?

**Answer:**
July 22 2022

**Retrieved Paths Collection:**
[
    "His death <-Led to- Abe's fabricated tweet",
    "Date of death -Is-> July 8, 2022",
    "His passing <-Was 39 years old at the time of- Shad gaspard -Passed away on-> May 17, 2020",
    "Neal lemlein -Is a friend of-> Gray -Died on-> March 5, 2019",
    "Neal -Died on-> July 22, 2022",
    "Neal -Died in-> July 22, 2022"
]

**Output::**
```json
{{
    "supporting_paths":
    [
        "Neal -Died on-> July 22, 2022",
        "Neal -Died in-> July 22, 2022"
    ]
}}
```


#### Example 2: Type 2 (Direct single-hop answer)
**Input Question:**
Who won the 2022 Tour de France?

**Answer:**
Jonas vingegaard

**Retrieved Paths Collection:**
[
    "2022 Tour de France -Started in-> Denmark",
    “2022 Tour de France -Finished in-> Paris”,
    "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard",
    "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar",
    "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won-> The yellow jersey"
]

**Output::**
```json
{{
    "supporting_paths":
    [
        "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard"
    ]
}}
```

#### Example 3: Type 3 (Multi-hop reasoning)
**Input Question:**
The King of Hollywood starred in what 1932 American pre-Code dram film?

**Answer:**
Strange Interlude

**Retrieved Paths Collection:**
[
    "A 1932 American pre-code drama film <-Is- Strange interlude",
    "1932 american pre-code gangster film <-Is a- Scarface -Also stars-> Karen morley",
    "A 1932 american pre-code mystery comedy thriller film <-Is- The thirteenth guest",
    "The king of hollywood <-Was_often_referred_to_as- Clark gable <-Stars- Strange interlude",
    "A 1932 american pre-code mystery comedy thriller film <-Is- The thirteenth guest -Is also known as-> Lady beware in the united kingdom"
]

**Output::**
```json
{{
    "supporting_paths":
    [
        "A 1932 American pre-code drama film <-Is- Strange interlude",
        "The king of hollywood <-Was_often_referred_to_as- Clark gable <-Stars- Strange interlude"
    ]
}}
```

#### Instructions:
**Your task is to identify supporting retrieved paths from the following inputs:

**Input Question:**
{input_question}

**Answer:**
{answer}

**Retrieved Paths Collection:**
{paths_collection}

**Output:**

"""


prompt_merged_triplets = """
### Task: Match Triplets and Paths Referring to the Same Fact

#### Objective:
Your task is to identify **matches between triplets and paths** that refer to the same fact. From the **provided triplets_collection**  and ** paths_collection**, merge triplets and paths that are aligned with each other. For multi-hop paths, split them into multiple single-hop paths before matching. If a triplet or path does not have any match, include it in the merged_triplets section at the end, leaving the unmatched entry empty for the corresponding field.

#### Response Format:
Your final response should follow this format:

```json
{{
    "merged_triplets": # type: List[List]
    [
        [
            ["Subject", "Predicate", "Object"],
            "Subject -Predicate-> Object",
        ],
        [
            ["Another subject", "Another predicate", "Another object"],
            ""
        ],
        [
            "",
            "Another subject -Another predicate-> Another object"
        ]
    ]
}}
```

#### Example 1:
**Input Triplets Collection:**
[
    [“Neal”, “Died on”, “July 22, 2022”],
    [“Neal”, “Died in”, “July 22, 2022”],
    [“Neal”, “Was”, “71 years old”]
]

**Input Paths Collection:**
[
    "Neal -Died on-> July 22, 2022",
    "Neal -Died in-> July 22, 2022",
    “Neal -Was-> 71 years old”,
    “Neal -Born in-> America”
]

**Output::**
```json
{{
    "merged_triplets":
    [
        [
            [“Neal”, “Died on”, “July 22, 2022”],
            [“Neal”, “Died in”, “July 22, 2022”],
            "Neal -Died on-> July 22, 2022",
            "Neal -Died in-> July 22, 2022"
        ],
        [
            [“Neal”, “Was”, “71 years old”],
            “Neal -Was-> 71 years old”
        ],
        [
            "",
            “Neal -Born in-> America”
        ]
    ]
}}
```

#### Example 2:
**Input Triplets Collection:**
[
    ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"]
]

**Input Paths Collection:**
[
    "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar"
]

**Split Paths Collection (After Splitting Multi-hop Paths):**
[
    “The overall winner of the 2022 Tour de France <-Is- Jonas Vingegaard”,
    “Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar"
]

**Output::**
```json
{{
    "merged_triplets":
    [
        [
            ["Jonas vingegaard", "Is", "The overall winner of the 2022 tour de france"],
            "The overall winner of the 2022 tour de france <-Is- Jonas vingegaard"
        ],
        [   
            "",
            “Jonas vingegaard -Won the tour by-> 3:34 over tadej pogačar"
        ]
    ]
}}
```

#### Example 3: 
**Input Triplets Collection:**
[
    ["Strange interlude", "Is", "A 1932 american pre-code drama film"],
    ["Strange interlude", "Stars", "Clark gable"],
    ["Clark gable",  "Was often referred to as", "The king of hollywood"],
]

**Input Paths Collection:**
[
    "A 1932 American pre-code drama film <-Is- Strange interlude",
    "The king of hollywood <-Was_often_referred_to_as- Clark gable <-Stars- Strange interlude"
]

**Split Paths Collection (After Splitting Multi-hop Paths):**
[
    “A 1932 American pre-Code drama film <-Is- Strange Interlude”,
    “The King of Hollywood <-Was_often_referred_to_as- Clark Gable”,
    “Clark Gable <-Stars- Strange Interlude”
]

**Output::**
```json
{{
    "merged_triplets":
    [
        [
            ["Strange interlude", "Is", "A 1932 american pre-code drama film"],
            "A 1932 American pre-code drama film <-Is- Strange interlude",
        ],
        [
            ["Strange interlude", "Stars", "Clark gable"],
            "Clark gable <-Stars- Strange interlude"
        ],
        [
            ["Clark gable",  "Was often referred to as", "The king of hollywood"],
            ""The king of hollywood <-Was_often_referred_to_as- Clark gable"
        ]
    ]
}}
```

#### Instructions:
1.Split multi-hop paths into single-hop paths.
2.Match each triplet with its corresponding single-hop path.
3.Include unmatched triplets or paths in the merged_triplets section at the end, with the unmatched field left as an empty string ("").
4.Provide the output in the specified format.

**Input Triplets Collection:**
{triplets_collection}

**Input Paths Collection:**
{paths_collection}

**Output:**

"""


prompt_extract_startnodes_and_retrieval_dependency = """
### Task: Extract Start Nodes and Retrieval Dependency Triplets

#### Objective:
Given a natural language question, extract:
1. The **retrieval_start_nodes**: the known entities from which retrieval should begin (typically the head entities of triplets).
2. The **retrieval requirement triplets** that capture the retrieval dependencies within the question.

#### Rules:
- A head entity is considered a **retrieval start node** .
- Use the triplet format: ["head_entity", "relation", "tail_entity"], where head_entity represents a known entity and tail_entity is the **concrete retrieval target**, not a variable.

#### Output Format:
Return a strict JSON object in the following format:

```json
{{
  "retrieval_start_nodes": ["Tom ZHANG", "Maria"],
  "triplets": [
    ["Tom ZHANG", "collaborated_with", "scholars"],
    ["Maria", "collaborated_with", "scholars"],
    ["scholars", "nationality", "American"]
  ]
}}
```

#### Example Input:
Question: Which American scholars have Tom ZHANG and Maria both collaborated with?

#### Output:
```json
{{
  "retrieval_start_nodes": ["Tom ZHANG", "Maria"],
  "triplets": [
    ["Tom ZHANG", "collaborated_with", "scholars"],
    ["Maria", "collaborated_with", "scholars"],
    ["scholars", "nationality", "American"]
  ]
}}
```

**Input Question:**
Question: {question}

**Output:**

"""


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(),
                      re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


class OpenAIExtractor:
    def __init__(self, config_path="/home/chency/NeutronRAG/neutronrag/dataprocess/openai_extractor/openai_config.json"):
        # home_dir = os.path.expanduser("~")
        # print(home_dir)
        # config_path = os.path.join(home_dir,"NeutronRAG/openai_config.json")

        # initial OpenAI configuration
        self.config = self._load_config(config_path)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get(
            "base_url", "https://api.openai.com/v1")
        self.model = self.config.get("model", "gpt-3.5-turbo")

        openai.api_key = self.api_key
        openai.base_url = self.base_url
        self.client = openai

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Config file {config_path} has invalid format")

    def get_triplets(self, context, temperature=0, language="en"):
        try:
            prompt_mapping = {
                "en": prompt_extract_triplest_without_predicate_limitation_str,
                "zh": prompt_extract_triplest_without_predicate_limitation_chine_str,
            }
            prompt_extract_triplest = prompt_mapping.get(language.lower())
            if not prompt_extract_triplest:
                raise NotImplementedError(
                    f"Language '{language}' is not supported.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_extract_triplest.format(
                    context=context)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return response.strip()
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def align_entity_relation(self, context, chunk_size=20, temperature=0, language="en"):
        align_prompt_mapping = {
            "en": alignment_prompt,
            "zh": alignment_prompt_chine_str,
        }
        prompt_align = align_prompt_mapping.get(language.lower())
        if not prompt_align:
            raise NotImplementedError(
                f"Language '{language}' is not supported.")

        aligned_results = []
        for chunk in self.split_list(context, chunk_size):
            chunk = [chunk]

            retry = 3
            while retry > 0:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt_align.format(input_data=chunk)}],
                        temperature=temperature
                    )
                    response = response.choices[0].message.content.strip()
                    partial_aligned_result = self.get_pasrse_output(
                        response, field="aligned_triples")

                    aligned_results.extend(partial_aligned_result)
                    break

                except json.JSONDecodeError as e:
                    print(f"JSON decoding error in chunk: {chunk}")
                    print(f"Error message: {e}")
                    retry -= 1
                except Exception as e:
                    print(f"Error in OpenAI API call: {e}")
                    retry -= 1
        return aligned_results

    def get_keyword_from_question(self, question_str, max_keywords=2, temperature=0) -> List[str]:
        try:
            # 指定个数
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[{"role": "user", "content": prompt_extract_keywords_str.format(
            #         input_question=question_str, max_keywords=max_keywords)}],
            #     temperature=temperature
            # )

            # 不指定个数 + 扩展同义词
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_extract_qury_keywords_str.format(
                    input_question=question_str)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return self.get_pasrse_output(response.strip(), field="keywords")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_filter_keyword_from_question(self, question_str, entities, temperature=0) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_filter_entities_from_entities_collection.format(
                    input_question=question_str, entity_collection=entities)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return self.get_pasrse_output(response.strip(), field="keywords")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_pasrse_output(self, output_str, field=Literal["aligned_triples", "keywords", "supporting_triplets", "supporting_paths", "merged_triplets"]):
        retry = 3
        while retry > 0:
            try:
                output_data = json.loads(extract_json_str(output_str))
                assert field in output_data
                if field == "aligned_triples":
                    aligned_triples = output_data[field]
                    capitalized_triplets = [
                        [
                            [
                                phrase.capitalize() if isinstance(phrase, str) else phrase
                                for phrase in triplet
                            ]
                            for triplet in each_item
                        ]
                        for each_item in aligned_triples
                    ]
                    return capitalized_triplets
                elif field == "keywords":
                    keywords = output_data[field]
                    assert isinstance(keywords, list)
                    return keywords
                elif field == "supporting_triplets":
                    supporting_triples = output_data[field]
                    assert isinstance(supporting_triples, list)
                    return supporting_triples
                elif field == "supporting_paths":
                    supporting_paths = output_data[field]
                    assert isinstance(supporting_paths, list)
                    return supporting_paths
                elif field == "merged_triplets":
                    merged_triplets = output_data[field]
                    assert isinstance(merged_triplets, list)
                    return merged_triplets
                # print("Converted output to list:")
                # print(output_data)
                # print(type(output_data))
            except json.JSONDecodeError as e:
                retry -= 1
                if retry == 0:
                    return {"error": "JSONDecodeError", "message": str(e), "output": output_str}

    def get_align_output(self, output_str):
        retry = 3
        while retry > 0:
            try:
                output_data = json.loads(extract_json_str(output_str))
                assert 'aligned_triples' in output_data
                aligned_triples = output_data["aligned_triples"]
                capitalized_triplets = [
                    [
                        [
                            phrase.capitalize() if isinstance(phrase, str) else phrase
                            for phrase in triplet
                        ]
                        for triplet in each_item
                    ]
                    for each_item in aligned_triples
                ]
                return capitalized_triplets
                # print("Converted output to list:")
                # print(output_data)
                # print(type(output_data))
            except json.JSONDecodeError as e:
                print(output_str)
                print("Failed to decode JSON:", e)
                retry -= 1

    def split_list(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def get_supporting_triples(self, question_str, answer, triples_collection, temperature=0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_identify_supporting_triples.format(
                    input_question=question_str, answer=answer, triples_collection=triples_collection)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return self.get_pasrse_output(response.strip(), field="supporting_triples")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_supporting_triples_from_evidences_and_retrieval_separately(self, question_str, answer, triples_collection, temperature=0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_identify_supporting_triples_separately.format(
                    input_question=question_str, answer=answer, triples_collection=triples_collection)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return self.get_pasrse_output(response.strip(), field="supporting_triples")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_supporting_triples_from_evidences_or_retrieval_separately(self, question_str, answer, triplets_collection, data_prompt_type=Literal["onehop", "multihop"], field=Literal["evidences", "retrieval"], temperature=0):
        retry = 3
        while retry > 0:
            try:
                if field == "evidences":
                    if data_prompt_type == "onehop":
                        prompt_str = prompt_identify_supporting_triples_from_evidences_triplets_type1_single_hop_triplets
                    elif data_prompt_type == "multihop":
                        prompt_str = prompt_identify_supporting_triples_from_evidences_triplets_type2_multi_hop_triplets
                    else:
                        raise ValueError("not exist prompt")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt_str.format(
                            input_question=question_str, answer=answer, triplets_collection=triplets_collection)}],
                        temperature=temperature
                    )
                    response = response.choices[0].message.content
                    # print(response)
                    return self.get_pasrse_output(response.strip(), field="supporting_triplets")
                elif field == "retrieval":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt_identify_supporting_triples_from_retrieve_results.format(
                            input_question=question_str, answer=answer, paths_collection=triplets_collection)}],
                        temperature=temperature
                    )
                    response = response.choices[0].message.content
                    # print(response)
                    return self.get_pasrse_output(response.strip(), field="supporting_paths")
            except Exception as e:
                retry -= 1
                if retry == 0:
                    print(f"Error in OpenAI API call: {e}")
                    return None

    def get_merged_triplets(self, triplets_collection, paths_collection, temperature=0):
        retry = 3
        while retry > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_merged_triplets.format(
                        triplets_collection=triplets_collection, paths_collection=paths_collection)}],
                    temperature=temperature
                )
                response = response.choices[0].message.content
                return self.get_pasrse_output(response.strip(), field="merged_triplets")
            except Exception as e:
                retry -= 1
                if retry == 0:
                    print(f"Error in OpenAI API call: {e}")
                    return None

    def extract_start_nodes_and_retrieval_triplets(self, question_str, temperature=0.2):
        retry = 3
        while retry > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_extract_startnodes_and_retrieval_dependency.format(
                        question=question_str)}],
                    temperature=temperature
                )
                response = response.choices[0].message.content
                json_obj = json.loads(extract_json_str(response))
                startnodes = json_obj.get("retrieval_start_nodes", [])
                dependency = json_obj.get("triplets", [])
                return startnodes, dependency
            except Exception as e:
                retry -= 1
                if retry == 0:
                    print(f"[extract_triplets] OpenAI error: {e}")
                    return [], []


if __name__ == '__main__':
    connector = OpenAIExtractor()
    question = "Who is the director of 'Carole King & James Taylor: Just Call Out My Name' and when is its premiere?"
    startnodes, dependency = connector.extract_start_nodes_and_retrieval_triplets(
        question)
    print(f"startnodes:{startnodes}")
    print(f"dependency:{dependency}")

    # context = "I am a student. Xiaoming play basektball. He also play football. The 2022 Met Gala takes place on May 2, 2022. 2022 Met Gala will take place on May 2, 2022. Met Gala 2022 takes place on May 2, 2022. It held on paris."
    # context_list = ["I am a student. Xiaoming play basektball. He also play football. ",
    #                 "The 2022 Met Gala takes place on May 2, 2022. 2022 Met Gala will take place on May 2, 2022. Met Gala 2022 takes place on May 2, 2022. It held on paris."]

    # context1 = "Best Supporting Actor winner Troy Kotsur became the first deaf man and second deaf individual overall to win an acting award.[a] Best Supporting Actress winner Ariana DeBose was the first Afro-Latina person and first openly queer woman of color to win an acting Oscar.[17] Furthermore, as a result of her win for portraying Anita in the 2021 film adaptation of the Broadway musical West Side Story, she and Rita Moreno, who previously won for playing the same character in the 1961 film adaptation, became the third pair of actors to win for portraying the same character in two different films.[b] Nominated for their performances as Leda Caruso in The Lost Daughter, Best Actress nominee Olivia Colman and Best Supporting Actress nominee Jessie Buckley were the third pair of actresses nominated for portraying the same character in the same film.[c] Flee became the first film to be nominated for Best Animated Feature, Best International Feature Film, and Best Documentary Feature in the same year.[19]  Winners are listed first, highlighted in boldface, and indicated with a double dagger (‡).[20]  The Academy held its 12th annual Governors Awards ceremony on March 25, 2022, during which the following awards were presented:[6]  The following individuals, listed in order of appearance, presented awards or performed musical numbers.[21]"
    # context2 = "[Excerpt from document]\ntitle: West Indies clinch T20 series after Shai Hope edges hosts past England total\npublished_at: 2023-12-21T23:45:58+00:00\nsource: The Guardian\nExcerpt:\n-----\nskip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion\n\nQuick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android by searching for 'The Guardian'.\n\nIf you already have the Guardian app, make sure you’re on the most recent version.\n\nIn the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.\n\nTurn on sport notifications. Was this helpful?\n-----"
    # print(connector.get_triplets(context2))

    # test_context = ["Nov 16, 2022 ... Carlos Alcaraz is set to become the youngest year-end No. 1 in the history of men's tennis after Rafael Nadal was knocked out of the ATP ..."]
    # # print(connector.get_triplets(test_context))

    # triplets = [
    #     [["2022 Met Gala", "ANNOUNCED", "May 2, 2022"],
    #      ["Met Gala 2022", "HELD_IN", "Paris"],
    #      ["Met Gala 2022", "Took_place_in", "Paris"]],
    #     [["Xiaoming", "IS", "student"]],
    #     [["Met Gala of 2022", "IN", "Paris"]],
    #     [["Met Gala 2022", "Ticket_at", "500rmb"],
    #      ["Xiaoming", "PLAYS", "basketball"],
    #      ["ming", "PLAY", "football"]],
    #     [["2022 Met Gala", "ANNOUNCED", "May 2, 2022"],
    #      ["Met Gala 2022", "HELD_IN", "Paris"],
    #      ["Met Gala 2022", "Took_place_in", "Paris"]]]
    # # print(connector.align_entity_relation(triplets))
    # # output = connector.align_entity_relation(triplets)
    # # connector.get_output(output)

    # question = "Who won the 2022 and 2023 Citrus Bowl?",

    # entities = [
    #     "2022 vrbo citrus bowl",
    #     "2022 peach bowl",
    #     "2022 bowl lvi",
    #     "Super bowl 2022",
    #     "2022 college football playoff's peach bowl",
    #     "2022 tournament",
    #     "2022 season",
    #     "2022 super bowl",
    #     "Cheez-it citrus bowl",
    #     "2022 edition",
    #     "2022 vrbo citrus bowl",
    #     "2022 peach bowl",
    #     "2022 bowl lvi",
    #     "Cheez-it citrus bowl",
    #     "2023",
    #     "2022 college football playoff's peach bowl",
    #     "2023 us open",
    #     "Super bowl 2022",
    #     "2023 event",
    #     "June 2023",
    #     "Cheez-it citrus bowl champions",
    #     "Florida citrus sports",
    #     "Super bowl lvi champions",
    #     "Cheez-it citrus bowl",
    #     "2022 vrbo citrus bowl",
    #     "Peach bowl",
    #     "Titles won",
    #     "Two touchdown passes",
    #     "Champion of the nfl",
    #     "Rose bowl",
    #     "Cheez-it citrus bowl champions",
    #     "Super bowl lvi champions",
    #     "Florida citrus sports",
    #     "Champion of the nfl",
    #     "Cheez-it citrus bowl",
    #     "2022 vrbo citrus bowl",
    #     "Reigning world champions",
    #     "Titles won",
    #     "Fcs championship",
    #     "Defending champion",
    #     "Cheez-it citrus bowl",
    #     "Peach bowl",
    #     "2022 vrbo citrus bowl",
    #     "Rose bowl",
    #     "Tangerine bowl",
    #     "Florida citrus sports",
    #     "2022 college football playoff's peach bowl",
    #     "2022 peach bowl",
    #     "American football game",
    #     "Super bowl lvi"
    # ]
    # print(connector.get_filter_keyword_from_question(question, entities))
    # print(connector.get_keyword_from_question(question, max_keywords=2))

    # "Mar 14, 2023 ... The Pixel Watch release date was October 13, 2022, and since then we're continuing to track the best Pixel Watch deals. Note that the Pixel ..."
