import os

from sympy import false
from volcenginesdkarkruntime import Ark
import json # 添加 json 模块用于解析 LLM 的响应
import re
from py2neo import Graph
import pypinyin
import numpy as np
from findd import find_closest_pattern
import torch
from sentence_transformers import SentenceTransformer

def init_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        # 使用多语言模型以支持中文
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
        print("SentenceTransformer 模型加载成功。")
        return model, device
    except Exception as e:
        print(f"加载 SentenceTransformer 模型失败: {e}")
        return None, 'cpu'

# --- Neo4j 配置 (需要替换为你的实际配置) ---
NEO4J_URI = "bolt://localhost:7687" # 示例 URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "1234" # 你的密码
try:
    driver = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.run("RETURN 1") # 尝试连接
    print("成功连接到 Neo4j 数据库。")
except Exception as e:
    print(f"连接 Neo4j 数据库失败: {e}")
    driver = None # 连接失败则设置 driver 为 None

# --- 方舟 API Key ---
# 从环境变量中读取您的方舟API Key
client = Ark(api_key=os.environ.get("ARK_API_KEY"))

def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def to_pinyin_if_chinese(text):
    """如果包含中文，转为无声调拼音，否则转小写"""
    if not text:
        return ""
    if contains_chinese(text):
        pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.NORMAL, heteronym=False)
        return " ".join([item[0] for item in pinyin_list if item])
    else:
        return text.lower()

# --- 修改 load_patterns ---
def load_patterns(file_path):
    """
    读取词条和Embedding，返回 (原始词条, 用于匹配的表示, embedding向量) 的元组列表.
    文件格式: "词条名称|数字 数字 数字 ..."
    """
    patterns_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|', 1)
                if len(parts) == 2:
                    original_pattern = parts[0].strip()
                    embedding_str = parts[1].strip()
                    try:
                        # 解析 embedding 字符串为 numpy 数组
                        embedding_vector = np.fromstring(embedding_str, sep=' ', dtype=np.float32)
                        if original_pattern:
                            matching_representation = to_pinyin_if_chinese(original_pattern)
                            patterns_data.append((original_pattern, matching_representation, embedding_vector))
                        else:
                            print(f"警告：文件 {file_path} 第 {line_num + 1} 行词条名称为空。")
                    except ValueError:
                        print(f"警告：无法解析文件 {file_path} 第 {line_num + 1} 行的Embedding: '{embedding_str}'")
                else:
                    print(f"警告：文件 {file_path} 第 {line_num + 1} 行格式错误，缺少 '|' 分隔符。")
    except FileNotFoundError:
        print(f"警告：词典文件未找到: {file_path}")
    return patterns_data

# --- 修改执行Neo4j查询的函数，确保正确获取labels ---
def execute_neo4j_query(query, parameters):
    """执行 Neo4j 查询并返回结果 (使用 py2neo)"""
    if not driver: # 检查数据库连接
        print("错误：Neo4j 数据库未连接。")
        return None
    try:
        # 使用 py2neo 的 driver.run 直接执行查询
        results = driver.run(query, parameters)
        # 将 py2neo 的结果转换为字典列表，以便后续处理
        # 注意：这里直接返回 py2neo 的 Record 对象列表，方便后续提取 .labels 等属性
        return list(results) # 返回 Record 对象列表

    except Exception as e:
        print(f"Neo4j 查询错误 (py2neo): {e}")
        return None # 查询出错返回 None

# --- 获取节点社区信息的函数 ---
def get_community_info(node_names):
    """
    获取给定节点名称列表所属的社区信息
    返回社区信息的字典 {community_id: description}
    """
    if not node_names:
        return {}
        
    community_info = {}
    community_ids_map = {} # 存储 node_name -> community_id

    try:
        # 1. 获取节点的社区ID
        query_node_ids = """
        MATCH (n)
        WHERE n.name IN $node_names
        RETURN n.name AS node_name, n.community_id AS community_id
        """
        node_community_results = execute_neo4j_query(query_node_ids, {"node_names": node_names})
        
        if not node_community_results:
            return {}, {} # 返回空的社区信息和节点映射

        community_ids_to_query = set()
        for record in node_community_results:
            node_name = record['node_name']
            community_id = record['community_id']
            if community_id is not None:
                community_ids_map[node_name] = community_id
                community_ids_to_query.add(community_id)

        if not community_ids_to_query:
            return {}, community_ids_map # 返回空的社区信息和节点映射

        # 2. 获取社区节点的描述信息
        community_query = """
        MATCH (c:Community)
        WHERE c.name IN $community_ids // 假设 Community 节点的 name 是 ID
        RETURN c.name AS community_id, c.description AS description
        """
        # py2neo 的参数应该是列表/集合
        community_results = execute_neo4j_query(community_query, {"community_ids": list(community_ids_to_query)})
        
        if community_results:
            for record in community_results:
                community_id = record['community_id'] # 直接使用 ID
                description = record.get('description')
                if community_id is not None and description: # ID 不为空且描述存在
                    community_info[community_id] = description # 键是社区 ID (int or str?)

        return community_info, community_ids_map # 返回社区信息和节点到社区ID的映射
        
    except Exception as e:
        print(f"获取社区信息时出错: {e}")
        return {}, {}

# --- 获取节点一跳邻居的函数 ---
def get_node_neighbors(node_name):
    """
    获取给定节点的一跳邻居
    返回邻居节点信息的原始 Record 对象列表
    """
    try:
        # 返回 n, r, m 以便后续处理
        cypher_query = "MATCH (n {name: $node_name}) -[r]- (m) RETURN n, type(r) as r, m"
        query_params = {"node_name": node_name}
        return execute_neo4j_query(cypher_query, query_params) # 返回 Record 列表
    except Exception as e:
        print(f"获取节点 '{node_name}' 的邻居时出错: {e}")
        return []

# --- 格式化一跳邻居信息，避免冗余 ---
# 修改：此函数现在接收原始 Record 列表
def format_neighbor_results_for_llm(node_records, center_node_name):
    """
    格式化原始查询结果 (Record列表) 为LLM易于理解的JSON友好格式。
    返回格式化后的列表，其中包含节点和关系信息。
    """
    if not node_records:
        return []
        
    formatted_results = []
    center_node_added = False

    for record in node_records:
        n_node = record.get('n')
        r_rel = record.get('r')
        m_node = record.get('m')

        # 处理中心节点 (只添加一次)
        if not center_node_added:
            center_node_data = dict(n_node)
            center_node_data['labels'] = list(n_node.labels)
            is_cyber_attack = 'CyberAttackPattern' in center_node_data['labels']
        if not is_cyber_attack:
            filtered_center = {
                    'name': center_node_data.get('name', center_node_name),
                    'community_id': center_node_data.get('community_id'),
                    'labels': center_node_data['labels'][0]
                }
        else:
            filtered_center = {
                    'name': center_node_data.get('name', center_node_name),
                    'community_id': center_node_data.get('community_id'),
                    'description': center_node_data.get('description', ''),
                    'labels': center_node_data['labels'][0]
                }
            formatted_results.append({"center_node": filtered_center})

            if r_rel and m_node:
                relationship_info = {}
                relationship_info['start_node'] = {'name': n_node.get('name')}  # 起点
                relationship_info['type'] = r_rel # 关系类型

                # 邻居节点
                neighbor_node_data = dict(m_node)
                neighbor_node_data['labels'] = list(m_node.labels)
                is_neighbor_cyber_attack = 'CyberAttackPattern' in neighbor_node_data['labels']

                if not is_neighbor_cyber_attack:
                    filtered_neighbor = {
                        'name': neighbor_node_data.get('name', '未知'),
                        'community_id': neighbor_node_data.get('community_id'),
                        'labels': neighbor_node_data['labels'][0]
                    }
                else:
                    filtered_neighbor = {
                        'name': neighbor_node_data.get('name', '未知'),
                        'community_id': neighbor_node_data.get('community_id'),
                        'description': neighbor_node_data.get('description', ''),
                        'labels': neighbor_node_data['labels'][0]
                    }
                relationship_info['end_node'] = filtered_neighbor  # 终点
                formatted_results.append({"relationship": relationship_info})
                center_node_added = True
                continue

        # 处理关系和邻居节点
        if r_rel and m_node:
            relationship_info = {}
            relationship_info['start_node'] = {'name': n_node.get('name')} # 起点
            relationship_info['type'] = r_rel # 关系类型
            # 邻居节点
            neighbor_node_data = dict(m_node)
            neighbor_node_data['labels'] = list(m_node.labels)
            is_neighbor_cyber_attack = 'CyberAttackPattern' in neighbor_node_data['labels']
            if not is_neighbor_cyber_attack:
                filtered_neighbor = {
                    'name': neighbor_node_data.get('name', '未知'),
                    'community_id': neighbor_node_data.get('community_id'),
                    'labels': neighbor_node_data['labels'][0]
                }
            else:
                filtered_neighbor = {
                    'name': neighbor_node_data.get('name', '未知'),
                    'community_id': neighbor_node_data.get('community_id'),
                    'description': neighbor_node_data.get('description', ''),
                    'labels': neighbor_node_data['labels'][0]
                }
            relationship_info['end_node'] = filtered_neighbor # 终点

            formatted_results.append({"relationship": relationship_info})
    
    return formatted_results

# --- 生成节点摘要的函数 ---
# 修改：接收 py2neo Node 对象
def generate_node_summary(node_obj, parent_node_name=None, relationship_type=None):
    """
    生成节点摘要信息
    node_obj: py2neo Node 对象
    parent_node_name: 父节点名称（从哪个节点查询到的）
    relationship_type: 与父节点的关系类型
    """
    if not node_obj:
        return None
        
    node_data = dict(node_obj) # 获取属性
    node_name = node_data.get('name')
    if not node_name:
        return None
        
    node_label_list = list(node_obj.labels) # 获取标签
    node_type_str = node_label_list[0] if node_label_list else "未知类型"
    community_id = node_data.get('community_id')
    
    summary = {
        "name": node_name,
        "labels": node_type_str, # 主要标签作为 labels 字符串 (原 type 字段)
        "parent_node": parent_node_name if parent_node_name else node_name,
        "relationship": relationship_type if relationship_type else "null",
        "community_id": community_id
        # 移除 labels 列表字段，is_valuable 将在 node_cache 外部管理
    }
    return summary

# --- 生成社区摘要的函数 ---
def generate_community_summary(community_id, description):
    """
    生成社区摘要信息
    """
    return {
        "community_id": community_id,
        "description": description
    }

# --- 从缓存中生成节点选择列表 ---
def generate_node_pick_list(node_cache):
    """
    从节点缓存中生成可选节点列表 (未被标记为valuable的节点)
    """
    pick_list = []
    for node_name, node_info in node_cache.items():
        # node_info 现在是 {"summary": {...}, "is_valuable": 0/1}
        if node_info.get("is_valuable", 0) == 0:
            if "summary" in node_info and "name" in node_info["summary"]:
               pick_list.append({
                    "name": node_info["summary"]["name"],
                    "labels": node_info["summary"].get("labels", "未知类型") # 从摘要中获取 labels 字段(原 type)
            })
    return pick_list

# --- 修改LLM判断函数，引入意向子问题和回答 ---
def llm_judge_and_select_with_subq(
    original_question,
    node_cache_info, # {name: {"summary": {...}, "is_valuable": 0/1}}
    community_summaries, # {id: description}
    qa_cache, # {name: {"question": q, "answer": a}}
    previous_sub_q_and_results, # {name: {"question": q, "results": formatted_neighbor_results}}
    node_pick_list,
    max_nodes=3
):
    """
    使用LLM判断信息是否足够，生成子问题回答，并选择下一批节点及子问题。
    """
    # 预处理节点缓存，移除 is_valuable 标志和 name 字段
    filtered_node_cache = {}
    for node_name, node_info in node_cache_info.items():
        if "summary" in node_info:
            # 创建摘要的副本，以免修改原始数据
            filtered_summary = {k: v for k, v in node_info["summary"].items() if k != "name"}
            filtered_node_cache[node_name] = {"summary": filtered_summary}
    
    prompt = f"""你是一个专业的图数据库分析和问答助手。请根据以下信息，判断是否足够回答最终用户问题，并按要求进行操作。

用户问题: "{original_question}"

已有信息：

1.  节点摘要缓存 (node_cache_info):
    包含已探索节点的摘要信息（标签、父节点名称、与父节点的关系、社区ID）。
{json.dumps(filtered_node_cache, ensure_ascii=False, indent=2)}

2.  社区信息(community_summaries):
    已识别社区的ID及其LLM生成的概要描述，希望这能更好地帮助你分析复杂关系。
    {json.dumps(community_summaries, ensure_ascii=False, indent=2)}

3.  历史问答对 (qa_cache):
    记录了之前为了探索特定节点而提出的"意向子问题"及其对应的（可能已生成的）回答。
    {json.dumps(qa_cache, ensure_ascii=False, indent=2)}

4.  上一轮查询结果 (previous_sub_q_and_results):
    包含上一轮选出的节点、对应的意向子问题、以及查询这些节点得到的一跳邻居详细信息。你需要根据这些信息为上一轮的意向子问题生成回答。
    {json.dumps(previous_sub_q_and_results, ensure_ascii=False, indent=2)}

5.  可选节点列表 (node_pick_list):
    当前未深入查询的、可供选择的邻居节点列表。
{json.dumps(node_pick_list, ensure_ascii=False, indent=2)}

任务：

请仔细分析以上所有信息，并执行以下步骤：

A. 判断信息充分性:
   判断当前所有累积信息（节点缓存、社区信息、历史问答、上一轮结果）是否足以直接、完整地回答最初的"用户问题"？

B. 如果信息足够 (is_sufficient: 1):
   1.  根据所有已有信息生成对"最终用户问题"的最终回答。
   2.  同时，仍然需要根据`previous_sub_q_and_results`中的信息，为上一轮的每个意向子问题生成简短回答。

C. 如果信息不足 (is_sufficient: 0):
   1.  根据`previous_sub_q_and_results`中的信息，为上一轮的每个意向子问题生成简短回答。
   2.  从`node_pick_list`中选择最多 {max_nodes} 个与最终用户问题最相关、最有希望提供缺失信息的节点。
   3.  为每个选定的新节点，生成一个清晰的新意向子问题，解释"为什么选择这个节点进行查询？期望从中获得什么信息来帮助回答最终用户问题？"。

输出格式：

请严格按照以下 JSON 格式返回结果：

如果信息足够:
```json
{{
  "is_sufficient": 1,
  "previous_answers": {{ // 对上一轮子问题的回答 (即使信息足够也要生成)
    "node_name1": "对上一轮节点1子问题的回答...",
    "node_name2": "对上一轮节点2子问题的回答..."
    // ...
  }},
  "response": "针对最终用户问题的完整回答..." // 最终答案
}}
```

**如果信息不足:**
```json
{{
  "is_sufficient": 0,
  "previous_answers": {{ // 对上一轮子问题的回答
    "node_name1": "对上一轮节点1子问题的回答...",
    "node_name2": "对上一轮节点2子问题的回答..."
  }},
  "next_nodes": [ // 选择的新节点及对应的新意向子问题
    {{"name": "new_node_A", "sub_question": "选择节点A是因为..."}},
    {{"name": "new_node_B", "sub_question": "选择节点B是因为..."}}
    // 节点最多 {max_nodes} 个
  ]
}}
```

重要提示:
1.所有回答和子问题都应基于提供的上下文信息。
2.在生成 `previous_answers` 时，请务必参考 `previous_sub_q_and_results` 中对应节点的查询结果。
3.确保 JSON 格式严格正确。
"""
    print("------ LLM Judge Prompt ------")
    print(prompt) # 调试时取消注释
    print("...") # 避免打印过长信息
    print("-----------------------------")

    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-250324", # 使用你的模型
            messages=[
                {"role": "system", "content": "你是一个专业的图数据库分析和问答助手。请严格按JSON格式输出。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # 稍微调低温度可能有助于生成结构化输出
        )
        llm_response_content = completion.choices[0].message.content
        print("------ LLM Judge Response ------")
        print(llm_response_content)
        print("------------------------------")
        
        # 尝试解析 JSON
        try:
            # 提取 JSON 部分 (更鲁棒的方式)
            json_match = re.search(r'```json\n(.*?)\n```', llm_response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有 ```json 包裹，尝试直接解析
                json_str = llm_response_content
                # 尝试去除可能的解释性文字
                if '{' in json_str:
                    json_str = json_str[json_str.find('{'):]
                if '}' in json_str:
                    json_str = json_str[:json_str.rfind('}')+1]

            result = json.loads(json_str)
            
            # 验证和提取结果
            is_sufficient = result.get('is_sufficient')
            previous_answers = result.get('previous_answers', {})
            
            if is_sufficient not in [0, 1]:
                raise ValueError("Invalid 'is_sufficient' value.")
            if not isinstance(previous_answers, dict):
                 # 如果previous_answers不是字典，尝试从其他地方获取或置空
                 print(f"警告：'previous_answers' 格式错误，尝试修正或置空。原始值: {previous_answers}")
                 # 尝试查找可能的嵌入字典
                 found_dict = False
                 if isinstance(result.get('response'), dict): # 有时可能误放在 response 里
                      maybe_answers = result['response'].get('previous_answers')
                      if isinstance(maybe_answers, dict):
                           previous_answers = maybe_answers
                           found_dict = True
                 if not found_dict:
                     previous_answers = {} # 无法修正则置空


            if is_sufficient == 1:
                final_response = result.get('response')
                if not isinstance(final_response, str):
                     # 如果 final_response 不是字符串，尝试从其他地方获取
                     print(f"警告：最终回答 'response' 格式错误，尝试修正。原始值: {final_response}")
                     # 可能 LLM 把回答放在了别处，或者嵌套了
                     if isinstance(result.get('response'), dict) and 'response' in result['response']:
                          final_response = result['response']['response'] # 尝试解套
                     elif isinstance(result.get('final_answer'), str): # 检查是否有 final_answer 字段
                          final_response = result['final_answer']
                     else:
                           final_response = str(final_response) # 最后尝试转为字符串

                return is_sufficient, previous_answers, final_response # response 是最终答案
            else: # is_sufficient == 0
                next_nodes_raw = result.get('next_nodes', [])
                if not isinstance(next_nodes_raw, list):
                    print(f"警告：'next_nodes' 格式错误，置为空列表。原始值: {next_nodes_raw}")
                    next_nodes_raw = []

                # 进一步验证 next_nodes 内容
                next_nodes_with_q = []
                for item in next_nodes_raw:
                    if isinstance(item, dict) and 'name' in item and 'sub_question' in item:
                        next_nodes_with_q.append({
                            "name": str(item['name']),
                            "sub_question": str(item['sub_question'])
                        })
                    else:
                        print(f"警告：'next_nodes' 中的项目格式错误，已跳过: {item}")

                if not next_nodes_with_q:
                     print("警告：LLM未选择任何有效的新节点进行下一步查询。")
                     # 这里可以考虑返回一个特殊状态或空列表，让主循环知道停止

                return is_sufficient, previous_answers, next_nodes_with_q # response 是下一步节点和问题

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"LLM 响应解析或验证失败: {e}")
            print(f"原始响应: {llm_response_content}")
            # 出错时，保守地认为信息不足，不选择新节点，也不生成回答
            return 0, {}, [] # 返回不足状态、空回答、空节点列表
            
    except Exception as e:
        print(f"调用 LLM 进行判断和选择时出错: {e}")
        return 0, {}, [] # 返回不足状态、空回答、空节点列表


# --- 更新节点缓存的函数 ---
def update_node_cache_valuable(node_cache, node_name):
    """标记节点为已查询 (valuable=1)"""
    if node_name in node_cache:
        node_cache[node_name]["is_valuable"] = 1
        return True
    return False

def add_node_to_cache(node_cache, node_obj, parent_node_name=None, relationship_type=None, is_valuable=0):
    """添加新节点到缓存或更新信息（如果已存在但未标记为valuable）"""
    node_name = node_obj.get('name')
    if not node_name:
        return False

    # 只有当节点不存在，或者存在但未被查询过时，才添加/更新摘要
    if node_name not in node_cache or node_cache[node_name].get("is_valuable", 0) == 0:
        summary = generate_node_summary(node_obj, parent_node_name, relationship_type)
        if summary:
            # 如果节点已存在但未查询，保留其 is_valuable 状态 (应为0)
            existing_valuable_status = node_cache.get(node_name, {}).get("is_valuable", 0)
            node_cache[node_name] = {
                "summary": summary,
                "is_valuable": is_valuable if node_name not in node_cache else existing_valuable_status
            }
            return True
    return False

# --- 修改扩展handle函数为 handle5，实现新 RAG 流程 ---
def handle5(question: str, model=None, fuzzy_threshold=81, embedding_threshold=0.66, max_nodes=3, max_cypher=4):
    """
    使用 LLM、社区感知、意向子问题跟踪的图谱 RAG 流程。
    """
    if not driver:
        return "数据库未连接，无法处理请求。"
    if not client:
        return "LLM 客户端未初始化，无法处理请求。"

    print(f"\n--- 开始处理问题 (handle5): {question} ---")
    print(f"参数: fuzzy={fuzzy_threshold}, embed={embedding_threshold}, max_nodes={max_nodes}, max_cypher={max_cypher}")

    # 初始化缓存
    node_cache = {} # {node_name: {"summary": {...}, "is_valuable": 0/1}}
    community_cache = {} # {community_id: description}
    qa_cache = {} # {node_name: {"question": str, "answer": str/None}}

    # --- 1. LLM 实体识别 (与 handle4 相同) ---
    possible_entity_types = ["attackpattern", "skill", "consequences", "indicator", "prerequisite"]
    entity_type_map_zh = { "攻击模式": "attackpattern", "技能": "skill", "后果": "consequences", "指标": "indicator", "先决条件": "prerequisite"}
    entity_prompt = f"""请从以下问题中识别出提及的实体及其最可能的类型。
问题: "{question}"
可能的实体类型: {', '.join(possible_entity_types)}。请以JSON格式返回结果，键是实体类型，值是实体名称列表。如果你认为识别出的实体名称有错别字，拼写错误或符号上的问题，请在原实体名称后加上你认为正确的实体名称。
如：{{"attackpattern":["sql 注人|sql 注入","Signture Spof|Signature Spoof","Byp@ss!ng@Phy$ica1#L0cks|Bypassing Physical Locks","Fuzzing')]|Fuzzing"]}}。
如果找不到明确的实体或类型，请返回空的JSON对象 {{}}。""" # 简化提示

    try:
        completion_entity = client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[
                {"role": "system", "content": "你是一个帮助识别网络安全问题文本中的实体的助手，实体类型限制在给定列表中。严格按JSON格式返回。"},
                {"role": "user", "content": entity_prompt}
            ],
        )
        llm_entity_response = completion_entity.choices[0].message.content
        # 尝试解析JSON
        try:
            # 提取 JSON 部分
            json_match = re.search(r'```json\n(.*?)\n```', llm_entity_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_entity_response
                if '{' in json_str: json_str = json_str[json_str.find('{'):]
                if '}' in json_str: json_str = json_str[:json_str.rfind('}')+1]
            entities = json.loads(json_str)
            if not isinstance(entities, dict): entities = {}
        except json.JSONDecodeError:
            print(f"LLM实体识别响应解析JSON失败: {llm_entity_response}")
            entities = {}
    except Exception as e:
        print(f"调用LLM进行实体识别时出错: {e}")
        return "抱歉，我在理解问题中的实体时遇到了错误。"

    if not entities:
        return "抱歉，我无法从您的问题中识别出明确的实体。请尝试换一种问法。"
    print(f"识别到的实体: {entities}")

    # --- 2. 第一次实体链接和 Neo4j 查询 ---
    initial_nodes_with_q = [] # 存储第一次查询的节点及其"虚拟"子问题

    for entity_type, entity_values in entities.items():
        normalized_entity_type = entity_type_map_zh.get(entity_type, entity_type).lower()
        if normalized_entity_type not in possible_entity_types: continue
        if not isinstance(entity_values, list): entity_values = [entity_values] # 确保是列表

        dict_file = f"{normalized_entity_type}_embeddings.txt"
        patterns = load_patterns(dict_file)
        if not patterns: continue # 如果词典加载失败则跳过

        for entity_name_raw in entity_values:
            if not isinstance(entity_name_raw, str): continue
            # 处理可能的修正: "原始|修正"
            entity_name_to_match = entity_name_raw.split('|')[-1].strip() # 取修正后的或原始的

            matched_name = find_closest_pattern(entity_name_to_match, patterns, model, fuzzy_threshold=fuzzy_threshold, embedding_threshold=embedding_threshold)
            if matched_name:
                print(f"实体 '{entity_name_to_match}' 链接到图谱节点: '{matched_name}'")
                # 为初始节点创建虚拟子问题并加入 qa_cache
                initial_sub_question = f"查询初始实体节点 '{matched_name}' 以了解其直接相关的上下文信息，帮助回答用户问题。"
                initial_nodes_with_q.append({"name": matched_name, "sub_question": initial_sub_question})
                if matched_name not in qa_cache:
                    qa_cache[matched_name] = {"question": initial_sub_question, "answer": None}

                # 获取邻居 (但不立即处理，在循环中统一处理)
            else:
                print(f"实体 '{entity_name_to_match}' 未能在类型 '{normalized_entity_type}' 的词典中找到匹配节点。")

    if not initial_nodes_with_q:
        return "抱歉，识别到的实体无法在知识库中找到对应节点。"

    # --- 3. 迭代查询流程 ---
    current_iteration = 0
    final_answer = None
    latest_valuable_nodes_with_q = initial_nodes_with_q # 第一轮要查询的节点
    previous_sub_q_and_results = {} # 第一轮此项为空
    
    while current_iteration < max_cypher:
        print(f"\n--- 第 {current_iteration + 1} / {max_cypher} 轮查询 ---")
        print(f"本轮待查询节点及子问题: {latest_valuable_nodes_with_q}")

        # -- a. 执行查询并格式化结果 --
        current_neighbors_results_map = {} # 存储本轮查询结果 {node_name: formatted_results}
        newly_discovered_nodes = [] # 存储本轮新发现的邻居节点对象

        for node_info in latest_valuable_nodes_with_q:
            node_name = node_info['name']
            # 标记当前节点为已查询
            update_node_cache_valuable(node_cache, node_name)

            raw_neighbor_records = get_node_neighbors(node_name) # 获取原始 Record 列表
            if raw_neighbor_records:
                formatted_results = format_neighbor_results_for_llm(raw_neighbor_records, node_name)
                current_neighbors_results_map[node_name] = formatted_results
                print(f"节点 '{node_name}' 查询到 {len(raw_neighbor_records)} 条邻居关系。")

                # -- b. 更新 Node Cache --
                # 处理中心节点 (确保在缓存中)
                center_node_obj = raw_neighbor_records[0].get('n')
                if center_node_obj:
                     # 添加或更新中心节点摘要，标记为 valuable
                     if add_node_to_cache(node_cache, center_node_obj):
                          update_node_cache_valuable(node_cache, node_name) # 确保标记

                # 处理邻居节点
                for record in raw_neighbor_records:
                    neighbor_node_obj = record.get('m')
                    relationship_obj = record.get('r')
                    if neighbor_node_obj:
                        neighbor_name = neighbor_node_obj.get('name')
                        rel_type = relationship_obj
                        # 添加邻居到缓存 (如果需要)，标记为 not valuable (is_valuable=0)
                        if add_node_to_cache(node_cache, neighbor_node_obj, node_name, rel_type, is_valuable=0):
                            if neighbor_name: newly_discovered_nodes.append(neighbor_node_obj) # 收集新节点对象
            else:
                print(f"节点 '{node_name}' 未查询到邻居。")
                current_neighbors_results_map[node_name] = [] # 即使没结果也要记录

        # -- c. 更新 Community Cache --
        if newly_discovered_nodes:
             new_node_names = [n.get('name') for n in newly_discovered_nodes if n.get('name')]
             if new_node_names:
                  print(f"发现新节点: {new_node_names}，正在获取社区信息...")
                  new_community_info, node_to_community_map = get_community_info(new_node_names)
                  # 更新节点缓存中的 community_id
                  for n_name, c_id in node_to_community_map.items():
                       if n_name in node_cache and 'summary' in node_cache[n_name]:
                            node_cache[n_name]['summary']['community_id'] = c_id
                  # 更新社区缓存
                  for c_id, desc in new_community_info.items():
                       if c_id not in community_cache:
                            community_cache[c_id] = desc
                            print(f"添加新社区 {c_id} 到缓存。")


        # -- d. 准备 LLM 判断所需信息 --
        node_pick_list = generate_node_pick_list(node_cache)
        # 准备上一轮 (现在是本轮完成查询) 的子问题和结果
        previous_sub_q_and_results = {}
        for node_info in latest_valuable_nodes_with_q:
             node_name = node_info['name']
             previous_sub_q_and_results[node_name] = {
                  "question": node_info['sub_question'],
                  "results": current_neighbors_results_map.get(node_name, []) # 获取本轮查询结果
             }

        # -- e. 调用 LLM 判断 --
        is_sufficient, previous_answers, llm_response_data = llm_judge_and_select_with_subq(
            question, 
            node_cache, 
            community_cache,
            qa_cache, # 传递完整的 QA 历史
            previous_sub_q_and_results, # 传递本轮查询的问题和结果
            node_pick_list,
            max_nodes
        )
        
        # -- f. 处理 LLM 判断结果 --
        # 更新 qa_cache 中上一轮子问题的答案
        if previous_answers:
            print("更新 QA 缓存中的答案:")
            for node_name, answer in previous_answers.items():
                if node_name in qa_cache:
                    qa_cache[node_name]["answer"] = answer
                    print(f"  - {node_name}: 已更新")
                else:
                     # 如果初始节点不在 qa_cache 中（理论上不应发生），添加它
                     print(f"  - 警告：节点 {node_name} 不在 qa_cache 中，尝试添加问答对。")
                     # 需要找到对应的子问题，它在 previous_sub_q_and_results 里
                     original_sub_q = previous_sub_q_and_results.get(node_name, {}).get("question", "未知原始子问题")
                     qa_cache[node_name] = {"question": original_sub_q, "answer": answer}


        if is_sufficient == 1:
            final_answer = llm_response_data # llm_response_data 是最终答案字符串
            print("\nLLM 判断信息已足够，准备生成最终答案。")
            break # 结束迭代
        else: # is_sufficient == 0
            latest_valuable_nodes_with_q = llm_response_data # llm_response_data 是 [{"name":..,"sub_question":..}]
            if not latest_valuable_nodes_with_q:
                 print("\nLLM 未选择任何新节点，且信息不足，终止查询。")
                 # 设置 final_answer 为信息不足的消息
                 final_answer = "抱歉，根据现有探索路径和信息，数据库中的信息不足以回答您的问题。"
                 break # 只有当没有选择新节点时才中断循环
            
            # 将新的子问题添加到 qa_cache (answer 为 None)
            for node_info in latest_valuable_nodes_with_q:
                 node_name = node_info['name']
                 sub_question = node_info['sub_question']
                 if node_name not in qa_cache: # 避免覆盖已有条目（理论上不应发生）
                      qa_cache[node_name] = {"question": sub_question, "answer": None}

            current_iteration += 1 # 准备下一轮

    # --- 4. 迭代结束，生成最终答案 (如果需要) ---
    if final_answer is None: # 如果循环正常结束或因 max_cypher 退出但未得到答案
        print("\n达到最大查询次数或循环结束，最后尝试生成答案...")
        
        # 预处理节点缓存，移除 is_valuable 标志和 name 字段
        filtered_node_cache = {}
        for node_name, node_info in node_cache.items():
            if "summary" in node_info:
                # 创建摘要的副本，以免修改原始数据
                filtered_summary = {k: v for k, v in node_info["summary"].items() if k != "name"}
                filtered_node_cache[node_name] = {"summary": filtered_summary}
        
        # 使用最后一次查询的 previous_sub_q_and_results
        answer_prompt = f"""请根据以下所有累积信息，为用户生成最终回答。

最终用户问题: "{question}"

已有信息：

1.  节点摘要缓存 (node_cache_info):
    {json.dumps(filtered_node_cache, ensure_ascii=False, indent=2)}

2.  社区信息 (community_summaries):
    {json.dumps(community_cache, ensure_ascii=False, indent=2)}

3.  历史问答对 (qa_cache): (包含了探索过程中的意向子问题及其回答)
    {json.dumps(qa_cache, ensure_ascii=False, indent=2)}

4.  最后一轮查询的问题与结果 (last_query_info):
    {json.dumps(previous_sub_q_and_results, ensure_ascii=False, indent=2)}

任务：
请整合以上所有信息，用流畅的中文回答最初的"最终用户问题"。对于一些关键术语将原文以（xxx）的形式给出。
请完全根据已有的资料与信息回答，不要使用先验知识。如果信息不足以支持回答问题，则回复："抱歉，数据库中的信息不足以回答这个问题。"
"""
        print("------ Final Answer Prompt ------")
        print(answer_prompt) # 调试时取消注释
        print("...")
        print("-----------------------------")
        try:
            completion_answer = client.chat.completions.create(
                model="deepseek-v3-250324", # 使用你的模型
                messages=[
                    {"role": "system", "content": "你是一个网络安全问答助手，根据提供的问题和多轮图谱探索信息生成最终答案。请严格基于提供的信息回答。"},
                    {"role": "user", "content": answer_prompt}
                ]
            )
            final_answer = completion_answer.choices[0].message.content.strip()

            
        except Exception as e:
            print(f"调用LLM进行最终答案生成时出错: {e}")
            final_answer = "抱歉，我在组织最终答案时遇到了问题。"
    
    print("\n--- 处理结束 (handle5) ---")
    return final_answer

# --- 主程序示例 (测试用) ---
if __name__ == "__main__":
    # model, device = init_model() # 如果需要 embedding
    model = None # 假设暂时不需要 embedding

    test_question = ("SQL Injection through SOAP Parameter Tampering导致的Consequences类型的结果可以被其他cyberattackpattern类型的节点所导致吗？如果有，节点名字叫什么")
    test_question2=("sql injection是什么时候被发现的")
    print(f"测试问题: {test_question}")

    final_response = handle5(
        test_question2,
        model,
        fuzzy_threshold=81,
        embedding_threshold=0.66,
        max_nodes=2, # 每次最多选2个节点
        max_cypher=3 # 最多迭代3次
    )
    print("\n------ 最终回答 ------")
    print(final_response)
    print("----------------------")


