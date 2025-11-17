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

# --- Neo4j 查询函数 (修正为使用 py2neo) ---
def execute_neo4j_query(query, parameters):
    """执行 Neo4j 查询并返回结果 (使用 py2neo)"""
    if not driver: # 检查数据库连接
        print("错误：Neo4j 数据库未连接。")
        return None
    try:
        # 使用 py2neo 的 driver.run 直接执行查询
        results = driver.run(query, parameters)
        # 将 py2neo 的结果转换为字典列表，以便后续处理
        # 注意：这假设你的查询返回节点和关系。
        # 你需要根据你的具体查询返回格式来调整这部分的数据提取逻辑。
        data = []
        for record in results:
            record_data = {}
            # record.keys() 包含查询返回的变量名，如 'n', 'relationship', 'm'
            for key in record.keys():
                node_or_rel = record[key]
                if hasattr(node_or_rel, 'properties'): # 检查是否是节点或关系
                    record_data[key] = dict(node_or_rel.properties)
                    # 可以选择性地添加标签或类型信息
                    if hasattr(node_or_rel, 'labels'):
                         record_data[key]['labels'] = list(node_or_rel.labels)
                    elif hasattr(node_or_rel, 'type'): # 对于关系
                         record_data[key]['type'] = type(node_or_rel).__name__
                else:
                    # 处理非节点/关系的返回值（例如纯字符串、数字等）
                    record_data[key] = node_or_rel
            if record_data: # 确保添加了数据
                data.append(record_data)


        # print(f"执行查询: {query} 参数: {parameters} -> 结果数: {len(data)}")
        return data # 返回实际查询结果处理后的列表

    except Exception as e:
        print(f"Neo4j 查询错误 (py2neo): {e}")
        return None # 查询出错返回 None



# --- 新的 handle2 函数 ---
def handle2(question: str,model=None):
    """
    使用 LLM 处理自然语言问题，结合知识图谱进行问答。
    流程：LLM实体识别 -> 模糊匹配 -> Neo4j查询 -> LLM答案生成
    """
    possible_entity_types = ["attackpattern", "skill", "consequences", "indicator", "prerequisite"]
    entity_type_map_zh = { # 如果需要，可以添加中文到英文的映射
        "攻击模式": "attackpattern",
        "技能": "skill",
        "后果": "consequences",
        "指标": "indicator",
        "先决条件": "prerequisite"
    }

    # 1. LLM 实体识别
    entity_prompt = f"""请从以下问题中识别出提及的实体及其最可能的类型。
问题: "{question}"
可能的实体类型: {', '.join(possible_entity_types)}。
请以JSON格式返回结果，键是实体类型，值是实体名称。例如：{{"attackpattern": ["SQL Injection","XSS Using Alternate Syntax"], "skill": ["Commercial tools are available"]}}。
如果找不到明确的实体或类型，请返回空的JSON对象 {{}}。如果你认为识别出的实体名称有错别字，拼写错误或符号上的问题，请在原实体名称后加上你认为正确的实体名称。
如：{{"attackpattern":["sql 注人|sql 注入","Signture Spof|Signature Spoof","Byp@ss!ng@Phy$ica1#L0cks|Bypassing Physical Locks","Fuzzing')]|Fuzzing"]}}。"""

    try:
        completion_entity = client.chat.completions.create(
            model="deepseek-v3-250324", # 或你选择的模型
    messages=[
                {"role": "system", "content": "你是一个帮助识别网络安全问题文本中的实体的助手，实体类型限制在给定列表中。"},
                {"role": "user", "content": entity_prompt}
            ],
            # temperature=0.2, # 较低的温度可能使输出更稳定和结构化
        )
        llm_entity_response = completion_entity.choices[0].message.content
        # 尝试解析JSON
        try:
            json_str = re.search(r'```json\n(.*?)\n```', llm_entity_response, re.DOTALL).group(1)
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

    # 2. 实体链接（模糊匹配）和 3. Neo4j 查询
    neo4j_results = {}
    matched_entities_info = {} # 存储匹配到的实体及其类型

    for entity_type, entity_values in entities.items(): # 重命名 entity_name 为 entity_values
        # 标准化实体类型（例如，如果LLM返回中文名）
        normalized_entity_type = entity_type_map_zh.get(entity_type, entity_type).lower()

        if normalized_entity_type not in possible_entity_types:
            print(f"警告：LLM返回了未知的实体类型 '{entity_type}'，跳过。")
            continue

        # 确保 entity_values 是一个列表，以便统一处理
        if isinstance(entity_values, str):
            entity_names_list = [entity_values] # 如果是字符串，转为单元素列表
        elif isinstance(entity_values, list):
            entity_names_list = entity_values # 如果是列表，直接使用
        else:
            print(f"警告：实体类型 '{entity_type}' 的值既不是字符串也不是列表: {entity_values}，跳过。")
            continue

        # 加载对应的词典
        dict_file = f"{normalized_entity_type}_embeddings.txt"
        patterns = load_patterns(dict_file)

        # 遍历识别出的该类型的所有实体名称
        for entity_name in entity_names_list:
            if not isinstance(entity_name, str): # 添加一层检查，确保列表内是字符串
                 print(f"警告：实体列表内包含非字符串元素: {entity_name}，跳过。")
                 continue

            matched_name = find_closest_pattern(entity_name, patterns, model, fuzzy_threshold=81, embedding_threshold=.66)

            if matched_name:
                # print(f"实体 '{entity_name}' (类型: {normalized_entity_type}) 匹配到词典条目: '{matched_name}'")
                # 检查是否已查询过此精确匹配名称，避免重复查询同一实体
                if f"关于'{matched_name}'({normalized_entity_type})的信息" in neo4j_results:
                    print(f"实体 '{matched_name}' 的信息已查询，跳过重复查询。")
                    continue

                matched_entities_info[matched_name] = normalized_entity_type # 记录匹配到的标准名称和类型

                # 构建 Neo4j 查询语句 (示例：查询一跳邻居)
                cypher_query = f"MATCH (n {{name: $entity_name}}) -[r]- (m) RETURN n, type(r) as relationship, m" # 限制返回数量避免过大
                query_params = {"entity_name": matched_name}

                # 执行查询 (使用 py2neo)
                result_data = execute_neo4j_query(cypher_query, query_params)


                # 将结果格式化为字符串以便传递给 LLM
                if result_data:
                    # 优化：格式化结果以减少冗余信息
                    formatted_results = []
                    if len(result_data) > 0:
                        # 1. 添加第一个结果中 'n' 节点的完整信息作为列表的第一个元素
                        first_n_node = result_data[0].get('n', None)
                        if first_n_node:
                            formatted_results.append(first_n_node)
                        else:
                            formatted_results.append({"name": matched_name, "info": "完整节点信息缺失"})

                        # 2. 遍历所有结果，添加关系信息，'n' 只保留 name
                        for record in result_data:
                            relationship_info = {}
                            n_node = record.get('n', {})
                            n_name = n_node.get('name', matched_name)
                            relationship_info['n'] = {'name': n_name}
                            relationship_info['relationship'] = record.get('relationship', '未知关系')
                            relationship_info['m'] = record.get('m', {}) # 保持 m 节点的完整信息
                            formatted_results.append(relationship_info)

                    result_str = json.dumps(formatted_results, ensure_ascii=False, indent=2)
                    print(result_str)
                    neo4j_results[f"关于'{matched_name}'({normalized_entity_type})的信息"] = result_str
                else:
                    neo4j_results[f"关于'{matched_name}'({normalized_entity_type})的信息"] = "查询无结果或出错。"
            else:
                print(f"实体 '{entity_name}' (类型: {normalized_entity_type}) 未能在 {dict_file} 中找到足够相似的匹配。")
                # 即使未匹配成功，也为原始识别出的名称添加一条记录
                if f"关于'{entity_name}'({normalized_entity_type})的信息" not in neo4j_results:
                    neo4j_results[f"关于'{entity_name}'({normalized_entity_type})的信息"] = "无法在知识库中精确定位此实体。"

    # 4. LLM 答案生成
    if not neo4j_results:
         return "抱歉，虽然识别到了实体，但无法在知识库中找到相关信息。"

    answer_prompt = f"""根据以下问题和相关的知识图谱查询结果，生成一个简短精炼的回答。
原始问题: "{question}"

知识图谱信息:
{json.dumps(neo4j_results, ensure_ascii=False, indent=2)}

请整合信息，用流畅的中文回答原始问题。对于一些关键术语将原文以（xxx）的形式给出。"""

    try:
        completion_answer = client.chat.completions.create(
            model="deepseek-v3-250324", # 或你选择的模型
            messages=[
                {"role": "system", "content": "你是一个网络安全问答助手，根据提供的问题和知识图谱信息生成答案。"},
                {"role": "user", "content": answer_prompt}
            ]
        )
        final_answer = completion_answer.choices[0].message.content
        return final_answer

    except Exception as e:
        print(f"调用LLM进行答案生成时出错: {e}")
        return "抱歉，我在组织答案时遇到了问题。"


# --- 主程序示例 (测试用) ---
if __name__ == "__main__":
    test_question = ("sqi injectionn的攻击技巧是什么？")
    print(f"测试问题: {test_question}")
    # model,_= init_model()
    model=None
    final_response = handle2(test_question, model)
    print("\n最终回答:")
    print(final_response)


