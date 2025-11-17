from py2neo import Graph

# 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))

# 示例 Cypher 查询，获取 CyberAttackPattern 和 Consequences 之间的关系
query = """
MATCH (a:CyberAttackPattern)-[r]->(b:Consequences)
RETURN a, type(r) as r, b
LIMIT 5
"""

# 执行查询
results = graph.run(query)

# 遍历结果
for record in results:
    # 获取节点和关系
    node_a = record.get('a')  # 节点 a
    relationship_obj = record.get('r')  # 关系对象 r
    node_b = record.get('b')  # 节点 b

    # 检查 relationship_obj 是否能读取到东西
    if relationship_obj:
        print("Relationship found:")
        print(f"Type: {relationship_obj}")  # 输出关系类型
        print(f"Properties: {relationship_obj}")  # 输出关系属性
    else:
        print("No relationship found in this record.")

    # 输出节点的信息
    print(f"Node A: {node_a}")
    print(f"Node B: {node_b}")
    print("-" * 40)
