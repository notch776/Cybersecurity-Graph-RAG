import os
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
        data = []
        for record in results:
            record_data = {}
            # record.keys() 包含查询返回的变量名，如 'n', 'relationship', 'm'
            for key in record.keys():
                node_or_rel = record[key]
                if hasattr(node_or_rel, 'properties'): # 检查是否是节点或关系
                    record_data[key] = dict(node_or_rel.properties)
                    # 参考neo4jtest.py的方法获取labels
                    if hasattr(node_or_rel, 'labels'):
                        record_data[key]['labels'] = list(node_or_rel.labels)
                    elif hasattr(node_or_rel, 'type'): # 对于关系
                        record_data[key]['type'] = type(node_or_rel).__name__
                else:
                    # 处理非节点/关系的返回值（例如纯字符串、数字等）
                    record_data[key] = node_or_rel
            if record_data: # 确保添加了数据
                data.append(record_data)

        return data # 返回实际查询结果处理后的列表

    except Exception as e:
        print(f"Neo4j 查询错误 (py2neo): {e}")
        return None # 查询出错返回 None

# --- 获取节点社区信息的函数 ---
def get_community_info(node_names):
    """
    获取给定节点名称列表所属的社区信息
    返回社区信息的字典
    """
    if not node_names:
        return {}
        
    community_info = {}
    try:
        # 构建查询，获取节点的社区ID
        query = """
        MATCH (n)
        WHERE n.name IN $node_names
        RETURN n.name AS node_name, n.community_id AS community_id
        """
        node_community_results = execute_neo4j_query(query, {"node_names": node_names})
        
        if not node_community_results:
            return {}
            
        community_ids = []
        for result in node_community_results:
            if 'community_id' in result and result['community_id'] is not None:
                community_ids.append(result['community_id'])
                
        # 去重社区ID
        unique_community_ids = list(set(community_ids))
        
        if not unique_community_ids:
            return {}
            
        # 获取社区节点的描述信息
        community_query = """
        MATCH (c:Community)
        WHERE c.name IN $community_ids
        RETURN c.name AS community_id, c.description AS description
        """
        community_results = execute_neo4j_query(community_query, {"community_ids": unique_community_ids})
        
        if community_results:
            for result in community_results:
                community_id = result.get('community_id')
                description = result.get('description')
                if community_id and description:
                    community_info[community_id] = description
                    
        return community_info
        
    except Exception as e:
        print(f"获取社区信息时出错: {e}")
        return {}

# --- 获取节点一跳邻居的函数 ---
def get_node_neighbors(node_name):
    """
    获取给定节点的一跳邻居
    返回邻居节点信息的列表
    """
    try:
        cypher_query = "MATCH (n {name: $node_name}) -[r]- (m) RETURN n, type(r) as relationship, m"
        query_params = {"node_name": node_name}
        
        return execute_neo4j_query(cypher_query, query_params)
    except Exception as e:
        print(f"获取节点 '{node_name}' 的邻居时出错: {e}")
        return []

# --- 格式化一跳邻居信息，避免冗余 ---
def format_neighbor_results(node_results, center_node_name):
    """
    格式化一跳邻居查询结果，避免中心节点信息冗余
    返回格式化后的列表
    """
    if not node_results or len(node_results) == 0:
        return []
        
    formatted_results = []

    first_record = node_results[0]
    if 'n' in first_record:
        center_node = first_record.get('n')
        # 检查是否为CyberAttackPattern类型
        center_node_data = dict(center_node)
        center_node_data['labels'] = list(center_node.labels)
        is_cyber_attack = False
        if 'CyberAttackPattern' in center_node_data['labels']:
            is_cyber_attack = True

        # 如果不是CyberAttackPattern类型，只保留name和community_id
        if not is_cyber_attack:
            filtered_center = {
                'name': center_node_data['name'],
                'community_id': center_node.get('community_id', None),
                'labels': center_node_data['labels']
            }
            formatted_results.append(filtered_center)
        else:
            # 是CyberAttackPattern类型，保留所有信息，确保包含description和labels
            filtered_center = {
                'name': center_node_data['name'],
                'community_id': center_node.get('community_id', None),
                'description': center_node.get('description', ''),
                'labels': center_node_data['labels']
            }
            # 可以添加其他需要保留的属性
            formatted_results.append(filtered_center)
    
    # 2. 遍历所有结果，添加关系信息，中心节点只保留name
    for record in node_results:
        if 'm' in record and 'relationship' in record:
            relationship_info = {}
            
            # 中心节点只保留名称和labels
            relationship_info['n'] = {'name': center_node_name,'labels': center_node_data['labels']}
            
            # 关系类型
            relationship_info['relationship'] = record.get('relationship', '未知关系')
            
            # 邻居节点
            neighbor_node = record.get('m')
            neighbor_node_data = dict(neighbor_node)
            neighbor_node_data['labels'] = list(neighbor_node.labels)
            # 检查邻居节点是否为CyberAttackPattern类型
            is_neighbor_cyber_attack = False
            if 'CyberAttackPattern' in neighbor_node_data['labels']:
                is_neighbor_cyber_attack = True
            
            # 如果不是CyberAttackPattern类型，只保留name和community_id和labels
            if not is_neighbor_cyber_attack:
                filtered_neighbor = {
                    'name': neighbor_node.get('name', '未知'),
                    'community_id': neighbor_node.get('community_id', None),
                    'labels': neighbor_node_data['labels'] # 保留labels
                }
                relationship_info['m'] = filtered_neighbor
            else:
                # 是CyberAttackPattern类型，保留description和labels等信息
                filtered_neighbor = {
                    'name': neighbor_node.get('name', '未知'),
                    'community_id': neighbor_node.get('community_id', None),
                    'description': neighbor_node.get('description', ''),
                    'labels': neighbor_node_data['labels']
                }
                # 可以添加其他需要保留的属性
                relationship_info['m'] = filtered_neighbor
                
            formatted_results.append(relationship_info)
    
    return formatted_results

# --- 生成节点摘要的函数 ---
def generate_node_summary(node_data, parent_node=None, relationship=None, is_valuable=0):
    """
    生成节点摘要信息
    node_data: 节点数据
    parent_node: 父节点名称（从哪个节点查询到的）
    relationship: 与父节点的关系
    is_valuable: 是否被标记为valuable节点 (0:未标记, 1:已标记且查询过)
    """
    if not node_data:
        return None
    node_data_data = dict(node_data)
    node_data_data['labels'] = list(node_data.labels)
    node_name = node_data.get('name')
    if not node_name:
        return None
        
    # 获取节点类型（标签），从节点的labels属性获取
    node_labels = node_data_data['labels']
    # 将 type 设置为第一个标签（如果存在）或默认值
    node_type_str = node_labels[0] if node_labels else "未知类型"

    # 获取社区ID
    community_id = node_data.get('community_id', None)

    summary = {
        "labels": node_type_str, # 使用第一个标签作为label 字符串
        "parent_node": parent_node if parent_node else node_name,  # 如果没有父节点，自己就是父节点
        "relationship": relationship if relationship else "null",   # 如果没有关系，设为null
        "community_id": community_id
        # 移除is_valuable字段
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
    从节点缓存中生成可选节点列表
    只返回未被标记为valuable的节点，包含节点名称和labels
    """
    pick_list = []
    for node_name, node_info in node_cache.items():
        if node_info.get("is_valuable", 0) == 0:  # 使用get方法安全访问
            pick_list.append({
                "name": node_name,
                "labels": node_info.get("labels", [])  # 从缓存中获取 labels 列表
            })
    return pick_list

# --- 修改LLM判断函数，调整节点列表展示 ---
def llm_judge_and_select(question, latest_neighbors_info, node_cache_info, community_info, node_pick_list, max_nodes=3):
    """
    使用LLM判断当前信息是否足够回答问题，并选择下一批valuable节点
    
    参数:
    - question: 用户问题
    - latest_neighbors_info: 最新查询的一跳邻居详细信息
    - node_cache_info: 节点摘要信息缓存
    - community_info: 社区信息
    - node_pick_list: 可选节点列表
    - max_nodes: 最多选择的节点数
    
    返回(是否足够, 回答或下一批节点列表)
    """
    try:
        prompt = f"""下面有一点Neo4j图数据库中的信息，请认真判断它们是否足够回答用户问题，并选择下一步操作。

用户问题: "{question}"

下面是缓存的节点信息。因为这个api没有多轮对话功能，所以需要缓存之前对话里已经了解到的图结构。
现在这里包括可能和答案相关的图结构中的节点的名称，类型(labels)，父类节点（是从什么节点那里一跳查询得到的），父类关系（与父类节点的关系）以及该节点所在的社区id。
{json.dumps(node_cache_info, ensure_ascii=False, indent=2)}

下面是社区结构信息。我预先使用社区算法将整个图分成了多个社区，上面你不是知道了社区id吗，下面就可以根据社区id了解对应的社区描述。
希望这能更好地帮助你分析复杂关系。

{json.dumps(community_info, ensure_ascii=False, indent=2)}

下面是最近查询到的邻居节点详细信息:
{json.dumps(latest_neighbors_info, ensure_ascii=False, indent=2)}

可选择的节点列表:
{json.dumps(node_pick_list, ensure_ascii=False, indent=2)}

请分析上述信息，并按照以下步骤操作：
1. 请仔细判断当前的一点信息是否足以回答用户问题
2. 如果足够，请使用已有信息生成回答
3. 如果不够，请从可选节点列表中选择最多{max_nodes}个与问题最相关的节点进行深入查询

请以JSON格式返回：
{{
  "is_sufficient": 0或1,  // 0表示信息不足，1表示信息足够
  "response": 如果信息足够，这里是回答；如果不足，这里是选择的节点名称列表
}}"""
        print(prompt)
        completion = client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[
                {"role": "system", "content": "你是一个专业的图数据库分析助手，擅长分析图结构和选择关键节点，或是根据图结构生成回答。请严格按照要求以JSON格式返回结果。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        llm_response = completion.choices[0].message.content
        print(llm_response)
        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\n(.*?)\n```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_response
                
            # 解析JSON
            result = json.loads(json_str)
            
            is_sufficient = result.get('is_sufficient', 0)
            response = result.get('response', [])
            
            # 验证响应格式
            if is_sufficient not in [0, 1]:
                print(f"LLM返回的is_sufficient值无效: {is_sufficient}")
                is_sufficient = 0
                
            if is_sufficient == 0 and not isinstance(response, list):
                print(f"LLM返回的节点列表格式无效: {response}")
                response = []
                
            if is_sufficient == 1 and not isinstance(response, str):
                print(f"LLM返回的回答格式无效: {response}")
                response = "抱歉，我无法根据当前信息提供有效回答。"
                
            # 如果返回的是节点列表，提取节点名称
            if is_sufficient == 0 and isinstance(response, list):
                # 处理可能包含字典的情况
                node_names = []
                for item in response:
                    if isinstance(item, dict) and 'name' in item:
                        node_names.append(item['name'])
                    elif isinstance(item, str):
                        node_names.append(item)
                response = node_names
                
            return is_sufficient, response
                
        except json.JSONDecodeError as e:
            print(f"LLM响应解析JSON失败: {e}")
            print(f"原始响应: {llm_response}")
            return 0, []
            
    except Exception as e:
        print(f"调用LLM进行判断和选择时出错: {e}")
        return 0, []

# --- 更新节点缓存的函数 ---
def update_node_cache(node_cache, node_name, is_valuable=1):
    """
    在节点缓存中更新节点的valuable标志
    """
    if node_name in node_cache:
        node_cache[node_name]["is_valuable"] = is_valuable
        return True
    return False

# --- 修改扩展handle3函数为完整的RAG流程 ---
def handle3(question: str, model=None, fuzzy_threshold=81, embedding_threshold=0.66, max_nodes=3, max_cypher=4):
    """
    使用 LLM 和社区感知的图谱RAG流程处理自然语言问题
    最多进行max_cypher次查询迭代，每次选择最多max_nodes个节点

    Args:
        question (str): 用户的问题。
        model: SentenceTransformer 模型实例。
        fuzzy_threshold (int, optional): 模糊匹配的阈值. Defaults to 81.
        embedding_threshold (float, optional): Embedding相似度的阈值. Defaults to 0.66.
        max_nodes (int, optional): 每次迭代LLM最多选择的节点数. Defaults to 3.
        max_cypher (int, optional): 最多进行Cypher查询的迭代次数. Defaults to 4.
    """
    # max_cypher = 4  # 最大查询迭代次数 - 使用函数参数
    # max_nodes = 3   # 每次最多选择的节点数 - 使用函数参数

    possible_entity_types = ["attackpattern", "skill", "consequences", "indicator", "prerequisite"]
    entity_type_map_zh = { # 如果需要，可以添加中文到英文的映射
        "攻击模式": "attackpattern",
        "技能": "skill",
        "后果": "consequences",
        "指标": "indicator",
        "先决条件": "prerequisite"
    }

    # 初始化缓存
    node_cache = {}  # 节点摘要缓存，使用字典结构，键为节点名称
    community_cache = []  # 社区摘要缓存
    latest_neighbors_results = []  # 存储最新一次查询的一跳邻居结果(格式化后)
    raw_neighbors_results = []  # 存储原始查询结果，用于节点摘要生成

    # 1. LLM 实体识别
    entity_prompt = f"""请从以下问题中识别出提及的实体及其最可能的类型。
问题: "{question}"
可能的实体类型: {', '.join(possible_entity_types)}。
请以JSON格式返回结果，键是实体类型，值是实体名称。例如：{{"attackpattern": ["SQL Injection","XSS Using Alternate Syntax"], "skill": ["Commercial tools are available"]}}。
如果找不到明确的实体或类型，请返回空的JSON对象 {{}}。如果你认为识别出的实体名称有错别字，拼写错误或符号上的问题，请在原实体名称后加上你认为正确的实体名称。
如：{{"attackpattern":["sql 注人|sql 注入","Signture Spof|Signature Spoof","Byp@ss!ng@Phy$ica1#L0cks|Bypassing Physical Locks","Fuzzing')]|Fuzzing"]}}。"""

    try:
        completion_entity = client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[
                {"role": "system", "content": "你是一个帮助识别网络安全问题文本中的实体的助手，实体类型限制在给定列表中。"},
                {"role": "user", "content": entity_prompt}
            ],
        )
        llm_entity_response = completion_entity.choices[0].message.content
        # 尝试解析JSON
        try:
            json_str = re.search(r'```json\n(.*?)\n```', llm_entity_response, re.DOTALL)
            if json_str:
                json_str = json_str.group(1)
            else:
                json_str = llm_entity_response
            entities = json.loads(json_str)
            if not isinstance(entities, dict):
                print("LLM实体识别返回格式非字典，置为空。")
                entities = {}
        except json.JSONDecodeError:
            print(f"LLM实体识别响应解析JSON失败: {llm_entity_response}")
            entities = {} # 如果解析失败，则认为没有识别到实体

    except Exception as e:
        print(f"调用LLM进行实体识别时出错: {e}")
        return "抱歉，我在理解问题中的实体时遇到了错误。"

    if not entities:
        return "抱歉，我无法从您的问题中识别出明确的实体。请尝试换一种问法。"

    # 2. 第一次实体链接和Neo4j查询 - 从识别的实体开始
    initial_matched_entities = {}  # 存储最初匹配到的实体

    for entity_type, entity_values in entities.items():
        # 标准化实体类型
        normalized_entity_type = entity_type_map_zh.get(entity_type, entity_type).lower()
        
        if normalized_entity_type not in possible_entity_types:
            print(f"警告：LLM返回了未知的实体类型 '{entity_type}'，跳过。")
            continue
            
        # 确保entity_values是列表
        if isinstance(entity_values, str):
            entity_names_list = [entity_values]
        elif isinstance(entity_values, list):
            entity_names_list = entity_values
        else:
            continue
            
        # 加载对应的词典
        dict_file = f"{normalized_entity_type}_embeddings.txt"
        patterns = load_patterns(dict_file)
        
        # 处理每个实体
        for entity_name in entity_names_list:
            if not isinstance(entity_name, str):
                continue
                
            # 使用传入的阈值进行匹配
            matched_name = find_closest_pattern(entity_name, patterns, model, fuzzy_threshold=fuzzy_threshold, embedding_threshold=embedding_threshold)
            
            if matched_name:
                initial_matched_entities[matched_name] = normalized_entity_type
                
                # 对每个实体进行一跳查询
                node_results = get_node_neighbors(matched_name)
                
                # 将原始查询结果添加到raw_neighbors_results
                if node_results:
                    raw_neighbors_results.extend(node_results)
                
                # 将查询结果格式化，避免冗余
                formatted_results = format_neighbor_results(node_results, matched_name)
                
                # 将格式化后的结果加入到最新一跳邻居结果中
                if formatted_results:
                    latest_neighbors_results.extend(formatted_results)
                
                if node_results:
                    # 处理中心节点
                    first_record = node_results[0]
                    if 'n' in first_record:
                        center_node = first_record.get('n')
                        # 将中心节点标记为valuable且已处理
                        center_summary = generate_node_summary(center_node, None, None, is_valuable=1)
                        if center_summary and matched_name not in node_cache:
                            # 确保is_valuable添加到缓存条目，而不是摘要本身
                            node_cache[matched_name] = center_summary
                            node_cache[matched_name]['is_valuable'] = 1
                    
                    # 处理邻居节点
                    for record in node_results:
                        if 'm' in record and 'relationship' in record:
                            neighbor_node = record['m']
                            neighbor_name = neighbor_node.get('name')
                            relationship = record['relationship']
                            
                            if neighbor_name and neighbor_name not in node_cache:
                                neighbor_summary = generate_node_summary(
                                    neighbor_node, 
                                    matched_name, 
                                    relationship,
                                    is_valuable=0
                                )
                                if neighbor_summary:
                                    # 确保is_valuable添加到缓存条目，而不是摘要本身
                                    node_cache[neighbor_name] = neighbor_summary
                                    node_cache[neighbor_name]['is_valuable'] = 0
    
    if not node_cache:
        return "抱歉，虽然识别到了实体，但无法在知识库中找到相关信息。"
        
    # 3. 获取社区信息
    # 提取所有节点名称
    all_node_names = list(node_cache.keys())
    community_info = get_community_info(all_node_names)
    
    # 更新社区缓存和节点社区ID
    for node_name, node_data in node_cache.items():
        # 从查询结果中获取社区ID
        for result in raw_neighbors_results:
            if 'n' in result and result['n'].get('name') == node_name:
                if 'community_id' in result['n']:
                    node_data['community_id'] = result['n']['community_id']
            elif 'm' in result and result['m'].get('name') == node_name:
                if 'community_id' in result['m']:
                    node_data['community_id'] = result['m']['community_id']
    
    # 更新社区缓存
    for community_id, description in community_info.items():
        community_summary = generate_community_summary(community_id, description)
        if community_summary and all(c['community_id'] != community_id for c in community_cache):
            community_cache.append(community_summary)
    
    # 4. 开始迭代查询流程
    current_iteration = 0
    final_answer = None
    
    while current_iteration < max_cypher:
        # 生成节点选择列表
        node_pick_list = generate_node_pick_list(node_cache)
        
        if not node_pick_list:
            print("没有更多可选节点，结束查询。")
            break
            
        # 将社区信息格式化
        formatted_community_info = {}
        for community in community_cache:
            formatted_community_info[community['community_id']] = community['description']
        
        # 让LLM判断信息是否足够，并选择valuable节点
        is_sufficient, response = llm_judge_and_select(
            question, 
            latest_neighbors_results,  # 传递格式化后的一跳邻居结果
            node_cache, 
            formatted_community_info, 
            node_pick_list,
            max_nodes
        )
        
        # 如果信息足够，获取最终答案
        if is_sufficient == 1:
            final_answer = response
            break
            
        # 信息不足，处理选择的节点
        valuable_nodes = response
        if not valuable_nodes or len(valuable_nodes) == 0:
            print("LLM未选择任何节点，自动结束查询。")
            break
            
        # 更新节点的valuable标志
        for node_name in valuable_nodes:
            if node_name in node_cache:
                node_cache[node_name]['is_valuable'] = 1
            
        # 清空最新邻居结果列表，准备存储新的查询结果
        latest_neighbors_results = []
        raw_neighbors_results = []
            
        # 对每个valuable节点进行一跳查询
        new_node_names = []
        for node_name in valuable_nodes:
            neighbor_results = get_node_neighbors(node_name)
            
            # 将原始查询结果添加到raw_neighbors_results
            if neighbor_results:
                raw_neighbors_results.extend(neighbor_results)
            
            # 将查询结果格式化，避免冗余
            formatted_results = format_neighbor_results(neighbor_results, node_name)
            
            # 将格式化后的结果加入到最新一跳邻居结果中
            if formatted_results:
                latest_neighbors_results.extend(formatted_results)
            
            if neighbor_results:
                for record in neighbor_results:
                    if 'm' in record and 'relationship' in record:
                        neighbor_node = record['m']
                        neighbor_name = neighbor_node.get('name')
                        if neighbor_name and neighbor_name not in node_cache:
                            relationship = record['relationship']
                            neighbor_summary = generate_node_summary(
                                neighbor_node,
                                node_name,
                                relationship,
                                is_valuable=0
                            )
                            if neighbor_summary:
                                # 确保is_valuable添加到缓存条目，而不是摘要本身
                                node_cache[neighbor_name] = neighbor_summary
                                node_cache[neighbor_name]['is_valuable'] = 0
                                new_node_names.append(neighbor_name)
        
        # 获取新节点的社区信息
        if new_node_names:
            new_community_info = get_community_info(new_node_names)
            
            # 更新节点社区ID
            for node_name in new_node_names:
                # 从查询结果中获取社区ID
                for result in raw_neighbors_results:
                    if 'm' in result and result['m'].get('name') == node_name:
                        if 'community_id' in result['m']:
                            node_cache[node_name]['community_id'] = result['m']['community_id']
            
            # 更新社区缓存
            for community_id, description in new_community_info.items():
                if all(c['community_id'] != community_id for c in community_cache):
                    community_cache.append(generate_community_summary(community_id, description))
        
        # 增加迭代计数
        current_iteration += 1
    
    # 5. 如果迭代结束后仍未得到答案，生成最终答案
    if not final_answer:
        # 格式化社区信息
        formatted_community_info = {}
        for community in community_cache:
            formatted_community_info[community['community_id']] = community['description']
            
        answer_prompt = f"""
用户问题: "{question}"

下面是缓存的Neo4j知识图谱节点信息。因为这个api没有多轮对话功能，所以需要缓存之前对话里已经了解到的图结构。
现在这里包括可能和答案相关的图结构中的节点的名称，类型，父类节点（是从什么节点那里一跳查询得到的），父类关系（与父类节点的关系）以及该节点所在的社区id。
{json.dumps(node_cache, ensure_ascii=False, indent=2)}

下面是社区结构信息。我预先使用社区算法将整个图分成了多个社区，上面你不是知道了社区id吗，下面就可以根据社区id了解对应的社区描述。
希望这能更好地帮助你分析复杂关系。
{json.dumps(formatted_community_info, ensure_ascii=False, indent=2)}

下面是最近一次查询到的邻居节点的详细信息。
{json.dumps(latest_neighbors_results, ensure_ascii=False, indent=2)}

请整合以上信息，用流畅的中文回答用户问题。对于一些关键术语将原文以（xxx）的形式给出。
请完全根据已有的资料与信息回答，不要使用先验知识。如果信息不足以支持回答问题，则回复抱歉，数据库中的信息不足以回答这个问题。"""

        try:
            completion_answer = client.chat.completions.create(
                model="deepseek-v3-250324",
                messages=[
                    {"role": "system", "content": "你是一个网络安全问答助手，根据提供的问题和知识图谱信息生成答案。"},
                    {"role": "user", "content": answer_prompt}
                ]
            )
            final_answer = completion_answer.choices[0].message.content
            
        except Exception as e:
            print(f"调用LLM进行最终答案生成时出错: {e}")
            final_answer = "抱歉，我在组织最终答案时遇到了问题。"
    
    return final_answer

# --- 主程序示例 (测试用) ---
if __name__ == "__main__":
    test_question = ("SQL Injection through SOAP Parameter Tampering导致的Consequences类型的结果可以被其他cyberattackpattern类型的节点所导致吗？如果有，节点名字叫什么")
    print(f"测试问题: {test_question}")
    # model,_= init_model()
    model=None
    final_response = handle3(
        test_question, 
        model, 
        fuzzy_threshold=81, # 自定义模糊匹配阈值
        embedding_threshold=0.66, # 自定义Embedding阈值
        max_nodes=2, # 每次最多选2个节点
        max_cypher=3 # 最多迭代3次
    )
    print("\n最终回答:")
    print(final_response)


