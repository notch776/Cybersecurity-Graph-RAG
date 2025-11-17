import pypinyin
from fuzzywuzzy import process, fuzz
import numpy as np
from numpy.linalg import norm
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

# 中文判断与转换函数
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

# 分隔检查是否有corrected实体的函数
def parse_input_entity(input_entity_str):
    """
    解析输入的实体字符串，支持两种格式:
    1. "原始实体|纠正实体" - 返回(原始实体, 纠正实体)
    2. "原始实体" - 返回(原始实体, None)
    
    Args:
        input_entity_str: 输入的实体字符串
        
    Returns:
        tuple: (原始实体, 纠正实体)
    """
    if not input_entity_str:
        return "", None
        
    parts = input_entity_str.split('|', 1)
    if len(parts) == 2:
        original = parts[0].strip()
        corrected = parts[1].strip()
        return original, corrected
    else:
        return input_entity_str.strip(), None

# 加载词典数据
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

# 字符串匹配函数
def levenshtein_matching(input_str, patterns_data, fuzzy_threshold=80, corrected_str=None):
    """
    使用Levenshtein距离进行字符串匹配
    
    Args:
        input_str: 输入的原始实体字符串
        patterns_data: 词典数据 [(original, matching_repr, embedding),...]
        fuzzy_threshold: 模糊匹配阈值 (0-100)
        corrected_str: 纠正后的实体字符串，如果有
    
    Returns:
        matched_entity: 匹配的实体名称，如果分数低于阈值则返回None
        highest_score: 最高匹配分数
    """
    if not patterns_data:
        return None, 0
    
    # 提取所有用于模糊匹配的表示形式
    matching_representations = [item[1] for item in patterns_data]
    
    # 准备输入字符串的匹配表示
    input_matching_repr = to_pinyin_if_chinese(input_str)
    
    # 获取原始实体的匹配结果
    original_match_result = process.extractOne(input_matching_repr, 
                                            matching_representations, 
                                            scorer=fuzz.ratio)
    
    best_match_repr = None
    highest_score = 0
    
    # 如果有纠正后的实体，也进行匹配并比较分数
    if corrected_str:
        corrected_repr = to_pinyin_if_chinese(corrected_str)
        corrected_match_result = process.extractOne(corrected_repr, 
                                                matching_representations, 
                                                scorer=fuzz.ratio)
        
        # 比较原始实体和纠正后实体的匹配分数，取较高者
        if corrected_match_result and original_match_result:
            if corrected_match_result[1] > original_match_result[1]:
                best_match_repr = corrected_match_result[0]
                highest_score = corrected_match_result[1]
                print(f"使用纠正实体 '{corrected_str}' 的Levenshtein匹配分数更高: {highest_score:.2f}")
            else:
                best_match_repr = original_match_result[0]
                highest_score = original_match_result[1]
                print(f"使用原始实体 '{input_str}' 的Levenshtein匹配分数更高: {highest_score:.2f}")
        elif corrected_match_result:
            best_match_repr = corrected_match_result[0]
            highest_score = corrected_match_result[1]
            print(f"只有纠正实体 '{corrected_str}' 有Levenshtein匹配结果: {highest_score:.2f}")
        elif original_match_result:
            best_match_repr = original_match_result[0]
            highest_score = original_match_result[1]
            print(f"只有原始实体 '{input_str}' 有Levenshtein匹配结果: {highest_score:.2f}")
    else:
        # 仅使用原始实体
        if original_match_result:
            best_match_repr = original_match_result[0]
            highest_score = original_match_result[1]
            print(f"原始实体 '{input_str}' 的Levenshtein匹配分数: {highest_score:.2f}")
    
    # 检查是否有高分匹配结果
    if best_match_repr and highest_score >= fuzzy_threshold:
        # 查找对应的原始实体
        for data in patterns_data:
            if data[1] == best_match_repr:
                print(f"Levenshtein匹配成功: '{data[0]}' (分数: {highest_score:.2f} >= {fuzzy_threshold})")
                return data[0], highest_score
    
    return None, highest_score

# Embedding函数
def get_embedding(text, model):
    """
    使用SentenceTransformer模型生成文本的embedding向量
    
    Args:
        text: 输入文本
        model: SentenceTransformer模型
    
    Returns:
        embedding: numpy数组形式的embedding向量
    """
    try:
        if not text:
            return None
        # 生成embedding
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"生成embedding时出错: {e}")
        return None

# 余弦相似度函数
def cosine_similarity(vec_a, vec_b):
    """计算两个向量之间的余弦相似度"""
    # 检查向量是否有效
    if vec_a is None or vec_b is None or vec_a.size == 0 or vec_b.size == 0:
        return 0.0

    # 确保向量维度一致
    if vec_a.shape != vec_b.shape:
        print(f"警告：向量维度不匹配 {vec_a.shape} vs {vec_b.shape}")
        return 0.0

    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 避免除以零
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

# 余弦相似度匹配函数
def embedding_matching(input_str, patterns_data, embedding_model, embedding_threshold=0.7, corrected_str=None):
    """
    使用embedding余弦相似度进行匹配
    
    Args:
        input_str: 输入的原始实体字符串
        patterns_data: 词典数据 [(original, matching_repr, embedding),...]
        embedding_model: SentenceTransformer模型
        embedding_threshold: 余弦相似度阈值 (0-1)
        corrected_str: 纠正后的实体字符串，如果有
    
    Returns:
        matched_entity: 最佳匹配的实体名称，如果无匹配则返回None
    """
    # 确定要生成embedding的文本
    if corrected_str:
        # 使用组合方式
        entity_description = f"{corrected_str} | {input_str}"
        print(f"使用组合输入进行余弦相似度计算: '{entity_description}'")
    else:
        entity_description = input_str
    
    # 生成输入实体的embedding
    input_embedding = get_embedding(entity_description, embedding_model)
    
    if input_embedding is None:
        print("无法生成输入实体的embedding")
        return None
    
    # 计算余弦相似度
    highest_cosine_sim = -1.0
    best_match_entity = None
    
    for original, _, embedding in patterns_data:
        cosine_sim = cosine_similarity(input_embedding, embedding)
        if cosine_sim > highest_cosine_sim:
            highest_cosine_sim = cosine_sim
            best_match_entity = original
    
    # 检查余弦相似度是否高于阈值
    if highest_cosine_sim >= embedding_threshold:
        print(f"余弦相似度匹配成功: '{best_match_entity}' (相似度: {highest_cosine_sim:.4f} >= {embedding_threshold})")
        return best_match_entity
    else:
        print(f"余弦相似度低于阈值 ({highest_cosine_sim:.4f} < {embedding_threshold})，无匹配结果")
        return None

# 总处理函数 - 作为模块化调用的入口
def find_closest_pattern(input_entity_str, patterns_data, embedding_model=None, fuzzy_threshold=81, embedding_threshold=0.66):
    """
    查找最接近的匹配实体，综合使用字符串匹配和嵌入向量匹配方法
    
    Args:
        input_entity_str: 输入实体字符串，格式为"原始实体"或"原始实体|纠正实体"
        patterns_data: 词典数据 [(original, matching_repr, embedding),...]
        embedding_model: 已初始化的SentenceTransformer模型，如果为None则只使用Levenshtein匹配
        fuzzy_threshold: Levenshtein匹配的阈值 (0-100)
        embedding_threshold: 余弦相似度匹配的阈值 (0-1)
    
    Returns:
        matched_entity: 匹配的实体名称，如果无匹配则返回None
    """
    # 解析输入实体
    original_entity, corrected_entity = parse_input_entity(input_entity_str)
    print(f"解析输入: 原始实体='{original_entity}', 纠正实体='{corrected_entity or '无'}'")
    
    # 阶段1: Levenshtein距离匹配
    levenshtein_result, highest_score = levenshtein_matching(
        original_entity, patterns_data, fuzzy_threshold, corrected_entity
    )
    
    # 如果Levenshtein匹配成功，直接返回结果
    if levenshtein_result:
        return levenshtein_result
    
    # 如果embedding模型不可用，则在Levenshtein匹配失败后返回None
    if embedding_model is None:
        print("无可用的embedding模型，无法进行余弦相似度匹配")
        return None
    
    # 阶段2: 使用embedding余弦相似度匹配
    print(f"Levenshtein匹配低于阈值 ({highest_score:.2f} < {fuzzy_threshold})，进行Embedding匹配")
    embedding_result = embedding_matching(
        original_entity, patterns_data, embedding_model, embedding_threshold, corrected_entity
    )
    
    return embedding_result

if __name__ == "__main__":
    # 加载词典数据
    entity_type = "attackpattern"
    dict_file = f"{entity_type}_embeddings.txt"  # 确保这个文件存在且格式正确
    patterns_data = load_patterns(dict_file)
    
    if not patterns_data:
        print("错误：未能加载词典数据。")
        exit(1)
    
    # 设置阈值参数
    fuzzy_threshold = 81
    embedding_threshold = 0.66
    
    # 在外部初始化模型，避免重复加载

    embedding_model, _ =init_model()
    
    print(f"使用参数：fuzzy_threshold={fuzzy_threshold}, embedding_threshold={embedding_threshold}")
    print(f"成功加载词典，共 {len(patterns_data)} 个实体\n")

    # 测试用例 - 使用新的输入格式
    test_cases = [
        "sqlzhuru|sql_injection",  # 有纠正实体
        "someRandomEntity",        # 无纠正实体
    ]
    
    print("开始测试匹配逻辑：")
    print("=" * 60)
    
    for i, input_entity_str in enumerate(test_cases, 1):
        print(f"\n测试 {i}: 输入='{input_entity_str}'")
        print("-" * 60)
        
        # 调用总处理函数，传入已初始化的模型
        matched_result = find_closest_pattern(
            input_entity_str, 
            patterns_data,
            embedding_model=embedding_model,
            fuzzy_threshold=fuzzy_threshold,
            embedding_threshold=embedding_threshold
        )
        
        print(f"最终匹配结果: {matched_result or '无匹配'}")
        print("-" * 60)


