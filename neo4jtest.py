import os
import json
from py2neo import Graph
from volcenginesdkarkruntime import Ark
from collections import Counter # 用于计数

# --- 配置 ---
# Neo4j 连接信息 (请替换为你的实际配置)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "1234" # 你的密码

# 方舟 API Key (从环境变量读取)
ARK_API_KEY = os.environ.get("ARK_API_KEY")
if not ARK_API_KEY:
    print("错误：请设置环境变量 ARK_API_KEY")
    exit()

# 要处理的社区 ID 列表
COMMUNITY_IDS_TO_PROCESS = [13, 26, 15, 33, 20, 19, 9, 16, 2, 18, 40, 17, 24, 12, 31, 7, 22, 27, 30, 32, 29, 39, 3, 8, 23, 43, 4, 10, 35, 44]
# LLM 模型
LLM_MODEL = "doubao-1-5-thinking-pro-250415" # 或者你选择的其他模型

# --- 初始化 ---
try:
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph.run("RETURN 1") # 测试连接
    print("成功连接到 Neo4j 数据库。")
except Exception as e:
    print(f"连接 Neo4j 数据库失败: {e}")
    exit()

try:
    client = Ark(api_key=ARK_API_KEY)
    print("方舟 Ark 客户端初始化成功。")
except Exception as e:
    print(f"初始化方舟 Ark 客户端失败: {e}")
    exit()

# --- 辅助函数：计算占比前N ---
def get_top_n_percentage(counts, total, n=3):
    """计算计数字典中占比前N的项及其百分比"""
    if not total:
        return []
    # 计算每个项的百分比
    percentages = {item: (count / total) * 100 for item, count in counts.items()}
    # 按百分比降序排序
    sorted_items = sorted(percentages.items(), key=lambda item: item[1], reverse=True)
    # 取前 N 个，格式化输出
    top_n = [(item, f"{percentage:.1f}%") for item, percentage in sorted_items[:n]]
    return top_n

# --- 主处理循环 ---
for community_id in COMMUNITY_IDS_TO_PROCESS:
    print(f"\n--- 正在处理社区 ID: {community_id} ---")

    community_nodes_data = []
    node_type_counts = Counter()
    total_nodes = 0

    try:
        # 1. 查询社区节点信息并统计节点类型
        query_nodes = """
        MATCH (n)
        WHERE n.community_id = $community_id
        RETURN n.name AS name, labels(n) AS labels, n.description AS description
        """
        results = graph.run(query_nodes, parameters={"community_id": community_id})

        for record in results:
            name = record["name"]
            labels = record["labels"]
            description = record["description"] # 可能为 None

            if not name: continue

            primary_type = labels[0] if labels else "未知类型"
            node_type_counts[primary_type] += 1
            total_nodes += 1

            node_info = {
                "name": name,
                "type": primary_type,
                "labels": labels
            }
            if description is not None:
                node_info["description"] = description
            community_nodes_data.append(node_info)

        if not community_nodes_data:
            print(f"社区 {community_id} 中没有找到任何节点。")
            continue

        print(f"社区 {community_id} 找到 {total_nodes} 个节点。")

        # 2. 查询社区内部边信息并统计关系类型
        relationship_type_counts = Counter()
        total_relationships = 0
        query_edges = """
        MATCH (n)-[r]->(m)
        WHERE n.community_id = $community_id AND m.community_id = $community_id
        RETURN type(r) AS relationship_type
        """
        edge_results = graph.run(query_edges, parameters={"community_id": community_id})

        for record in edge_results:
            rel_type = record["relationship_type"]
            relationship_type_counts[rel_type] += 1
            total_relationships += 1

        print(f"社区 {community_id} 找到 {total_relationships} 条内部关系。")

        # 3. 计算占比
        top_node_types = get_top_n_percentage(node_type_counts, total_nodes, n=3)
        top_relationship_types = get_top_n_percentage(relationship_type_counts, total_relationships, n=3)

        print(f"社区 {community_id} 节点类型占比前三: {top_node_types}")
        print(f"社区 {community_id} 关系类型占比前三: {top_relationship_types}")

        # 4. 准备数据并调用 LLM API 获取概要
        nodes_json = json.dumps(community_nodes_data, ensure_ascii=False, indent=2)

        # 构建包含统计信息和节点数据的 Prompt
        stats_summary = f"该社区包含 {total_nodes} 个节点和 {total_relationships} 条内部关系。\n"
        stats_summary += "节点类型占比前三： " + ", ".join([f"{t}: {p}" for t, p in top_node_types]) + "\n"
        stats_summary += "关系类型占比前三： " + ", ".join([f"{t}: {p}" for t, p in top_relationship_types])

        llm_prompt = f"""以下信息描述了ID为{community_id}的知识图谱社区：

统计概要:
{stats_summary}

社区节点详细信息 (JSON 格式，包含节点名称 name, 主要类型 type, 类型标签 labels, 以及可能的节点描述 description):
```json
{nodes_json}
```

请基于上述所有信息（统计概要和节点详情），总结这个社区的主要特征和概要。用一段话进行描述，力求简洁明了地涵盖社区的主要构成，联系特点与领域方向。一些专业术语将原文以（xxx）的形式给出。
例如：社区A包含100个节点，主要由 AttackPattern（60%）和 Vulnerability（30%）组成，大多与数据库注入（SQL Injection）有关，另外还涉及了数据泄露（Data Leakage）和恶意软件（Malware）。
主要关系为 MITIGATED_BY(50%),DEPENDENCY_OF(15%)和IS_A(10%)。

请生成社区概要："""
        print(llm_prompt) # 可选：打印完整的 prompt
        try:
            print(f"正在为社区 {community_id} 调用 LLM 生成概要...")
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个网络安全领域的知识图谱分析助手，擅长根据节点、关系统计和详细信息总结社区特征。"},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            community_summary = completion.choices[0].message.content.strip()
            print(f"社区 {community_id} 概要生成成功。")

        except Exception as e:
            print(f"调用 LLM API 为社区 {community_id} 生成概要时出错: {e}")
            continue

        # 5. 将概要写入 Neo4j
        community_id_int = int(community_id)
        try:
            query_update = """
            MATCH (c:Community {name: $community_id_int})
            SET c.description = $summary
            RETURN c.name AS updated_community_name
            """
            update_result = graph.run(query_update, parameters={
                "community_id_int": community_id_int,
                "summary": community_summary
            }).data()

            if update_result:
                print(f"成功将概要写入社区 {update_result[0]['updated_community_name']} 的 description 属性。")
            else:
                 print(f"警告：未找到名为 '{community_id_int}' 的 Community 节点，尝试创建...")
                 query_create_update = """
                 MERGE (c:Community {name: $community_id_int})
                 ON CREATE SET c.description = $summary
                 ON MATCH SET c.description = $summary
                 RETURN c.name AS community_name
                 """
                 create_update_result = graph.run(query_create_update, parameters={
                     "community_id_int": community_id_int,
                     "summary": community_summary
                 }).data()
                 if create_update_result:
                     print(f"成功创建/更新社区 {create_update_result[0]['community_name']} 并写入概要。")
                 else:
                      print(f"错误：无法更新或创建社区 {community_id_int} 的概要。")

        except Exception as e:
            print(f"更新社区 {community_id_int} 的 description 时出错: {e}")

    except Exception as e:
        print(f"处理社区 {community_id} 时发生意外错误: {e}")

print("\n--- 所有指定社区处理完毕 ---")