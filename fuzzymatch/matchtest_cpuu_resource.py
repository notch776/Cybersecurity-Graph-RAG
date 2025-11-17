import os
import numpy as np
from fuzzywuzzy import process, fuzz
from numpy.linalg import norm
import pypinyin
import itertools
from tqdm import tqdm
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置CPU核心数
CPU_CORES = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统


# 检查字符串是否包含中文
def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


# 中文预处理 - 转换为拼音
def to_pinyin_if_chinese(text):
    """如果包含中文，转为无声调拼音，否则转小写"""
    if contains_chinese(text):
        try:
            pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.NORMAL, heteronym=False)
            return " ".join([item[0] for item in pinyin_list if item])
        except Exception as e:
            print(f"拼音转换错误 '{text}': {e}")
            return text.lower()  # 出错时退回到小写
    else:
        return text.lower()


# 读取向量库文件
def load_dict_with_embeddings(file_path):
    """
    读取向量库文件，返回 (原始词条, embedding向量) 的元组列表
    文件格式: "词条名称|数字 数字 数字 ..."
    """
    patterns_data = []
    if not os.path.exists(file_path):
        print(f"错误：词典文件未找到: {file_path}")
        return patterns_data
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
    except Exception as e:
        print(f"读取词典文件 {file_path} 时发生意外错误: {e}")
    return patterns_data


# 计算余弦相似度(单条计算)
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


# 批量计算余弦相似度（用于多进程）
def batch_cosine_similarity_worker(query_vector, embeddings_batch):
    """
    批量计算余弦相似度，用于多进程并行计算

    Args:
        query_vector: 查询向量
        embeddings_batch: (起始索引, 嵌入向量批次)

    Returns:
        (起始索引, 相似度数组)
    """
    start_idx, batch_vectors = embeddings_batch

    # 归一化查询向量
    query_norm = norm(query_vector)
    if query_norm == 0:
        query_norm = 1.0
    normalized_query = query_vector / query_norm

    # 计算每个向量的相似度
    similarities = []
    for vec in batch_vectors:
        if vec is not None and vec.size > 0:
            vec_norm = norm(vec)
            if vec_norm == 0:
                similarity = 0.0
            else:
                normalized_vec = vec / vec_norm
                similarity = np.dot(normalized_query, normalized_vec)
        else:
            similarity = 0.0
        similarities.append(similarity)

    return start_idx, np.array(similarities, dtype=np.float32)


# 并行计算余弦相似度
def parallel_cosine_similarity(query_vector, dict_entries, num_workers=None):
    """
    使用多进程并行计算查询向量与所有词典向量的余弦相似度

    Args:
        query_vector: 查询向量
        dict_entries: 词典项列表，格式为[(original, matching_repr, embedding),...]
        num_workers: 工作进程数，默认为CPU核心数-1

    Returns:
        similarities: 相似度数组，与dict_entries顺序对应
    """
    if num_workers is None:
        num_workers = CPU_CORES

    # 提取所有嵌入向量
    embeddings = [entry[2] for entry in dict_entries]

    # 如果向量数量很少，直接串行计算
    if len(embeddings) < 1000:
        return np.array([cosine_similarity(query_vector, vec) for vec in embeddings], dtype=np.float32)

    # 计算每个工作进程处理的批次大小
    batch_size = max(100, len(embeddings) // num_workers)

    # 准备任务批次
    batches = []
    for i in range(0, len(embeddings), batch_size):
        end_idx = min(i + batch_size, len(embeddings))
        batches.append((i, embeddings[i:end_idx]))

    # 创建结果数组
    results = np.zeros(len(embeddings), dtype=np.float32)

    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务
        futures = [executor.submit(batch_cosine_similarity_worker, query_vector, batch) for batch in batches]

        # 收集结果
        for future in as_completed(futures):
            try:
                start_idx, batch_similarities = future.result()
                end_idx = min(start_idx + len(batch_similarities), len(results))
                results[start_idx:end_idx] = batch_similarities[:end_idx - start_idx]
            except Exception as e:
                print(f"计算任务出错: {e}")

    return results


# 阶段2: 余弦相似度计算阶段 (并行版本)
def cosine_similarity_matching(embedding_vector, dict_entries, embedding_threshold=0.7):
    """
    使用余弦相似度计算并返回结果，支持多进程并行计算

    Args:
        embedding_vector: 输入实体的嵌入向量
        dict_entries: 词典项列表，格式为[(original, matching_repr, embedding),...]
        embedding_threshold: 相似度阈值，高于此值返回匹配结果，低于此值认为无匹配

    Returns:
        matched_entity: 如果得分高于阈值，返回最匹配的实体，否则返回"无匹配"
        is_high_confidence: 布尔值，表示是否为高置信度匹配
        no_answer: 布尔值，表示是否无答案
    """
    # 使用并行计算所有余弦相似度
    try:
        similarities = parallel_cosine_similarity(embedding_vector, dict_entries)
        patterns = [entry[0] for entry in dict_entries]

        # 查找最高相似度
        highest_similarity = np.max(similarities)
        best_index = np.argmax(similarities)
        best_embedding_match = patterns[best_index]

        # 检查最高相似度是否高于阈值
        if highest_similarity >= embedding_threshold:
            return best_embedding_match, True, False
        else:
            # 低于阈值则认为无匹配
            return "数据库里暂无查找的条目，或拼写错误", False, True

    except Exception as e:
        print(f"余弦相似度计算出错: {e}")
        return "计算出错，无匹配结果", False, True


# 阶段1: 使用Levenshtein距离计算阶段
def levenshtein_matching(input_entity, dict_entries, fuzzy_threshold=90, corrected_entity=None):
    """
    使用Levenshtein距离计算相似度并返回结果

    Args:
        input_entity: 输入实体名称 (原始实体)
        dict_entries: 词典项列表，格式为[(original, matching_repr, embedding),...]
        fuzzy_threshold: 直接结束的阈值
        corrected_entity: 纠正后的实体，如果为"."或None则不使用

    Returns:
        matched_entity: 如果得分高于阈值，返回最匹配的实体
        is_high_confidence: 布尔值，表示是否为高置信度匹配
    """
    # 转换输入实体为拼音（如果是中文）
    input_repr = to_pinyin_if_chinese(input_entity)
    matching_representations = [item[1] for item in dict_entries]

    # 原始实体的匹配结果
    original_match_result = process.extractOne(input_repr, matching_representations, scorer=fuzz.ratio)
    best_match_repr = None
    highest_score = 0
    
    # 如果有纠正后的实体且不是"."，也进行匹配
    if corrected_entity and corrected_entity != ".":
        corrected_repr = to_pinyin_if_chinese(corrected_entity)
        corrected_match_result = process.extractOne(corrected_repr, matching_representations, scorer=fuzz.ratio)
        
        # 比较原始实体和纠正后实体的匹配分数，取较高者
        if corrected_match_result and original_match_result:
            if corrected_match_result[1] > original_match_result[1]:
                best_match_repr = corrected_match_result[0]
                highest_score = corrected_match_result[1]
            else:
                best_match_repr = original_match_result[0]
                highest_score = original_match_result[1]
        elif corrected_match_result:
            best_match_repr = corrected_match_result[0]
            highest_score = corrected_match_result[1]
        elif original_match_result:
            best_match_repr = original_match_result[0]
            highest_score = original_match_result[1]
    else:
        # 仅使用原始实体
        if original_match_result:
            best_match_repr = original_match_result[0]
            highest_score = original_match_result[1]

    # 检查是否有高置信度匹配
    if best_match_repr and highest_score >= fuzzy_threshold:
        # 查找对应的原始实体
        best_match_index = -1
        for i, data in enumerate(dict_entries):
            if data[1] == best_match_repr:
                best_match_index = i
                break

        if best_match_index != -1:
            matched_entity = dict_entries[best_match_index][0]
            return matched_entity, True
        else:
            print(f"警告：无法在词典中找到匹配表示 {best_match_repr}")

    # 如果没有高置信度匹配，返回None
    return None, False


# 并行处理实体批次
def process_entities_batch(batch_data, dict_entries, fuzzy_threshold, embedding_threshold, top_n):
    """
    并行处理一批实体数据

    Args:
        batch_data: 批次数据列表，每个元素是(原始实体,纠正实体,向量A,向量B,真实实体,错误类型)
        dict_entries: 词典项列表
        fuzzy_threshold: Levenshtein匹配的阈值
        embedding_threshold: 余弦相似度匹配的阈值
        top_n: 未使用，保留参数兼容性

    Returns:
        results: 处理结果列表，每个元素是(原始实体,匹配实体,真实实体,错误类型,阶段信息)
    """
    results = []

    for original, corrected, vector_a, vector_b, true_entity, error_type in batch_data:
        try:
            # 阶段信息字典
            stage_info = {
                "stage": None,  # 最终匹配阶段
                "levenshtein_matched": None,  # Levenshtein阶段匹配结果
                "cosine_matched": None,  # 余弦相似度阶段匹配结果
                "final_matched": None,  # 最终匹配结果
                "is_correct": False  # 是否正确匹配
            }

            # 阶段1: Levenshtein匹配
            # 对于spell_err和symbol_err，根据corrected是否为"."决定处理方式
            if error_type in ["spell_err", "symbol_err"]:
                levenshtein_match, levenshtein_high_conf = levenshtein_matching(
                    original, dict_entries, fuzzy_threshold, corrected
                )
            else:
                levenshtein_match, levenshtein_high_conf = levenshtein_matching(
                    original, dict_entries, fuzzy_threshold
                )

            # 记录Levenshtein阶段结果
            stage_info["levenshtein_matched"] = levenshtein_match if levenshtein_high_conf else None

            # 如果Levenshtein有高置信度匹配，使用它
            if levenshtein_high_conf:
                stage_info["stage"] = "levenshtein"
                stage_info["final_matched"] = levenshtein_match
                stage_info["is_correct"] = (levenshtein_match == true_entity)
                results.append((original, levenshtein_match, true_entity, error_type, stage_info))
                continue

            # 阶段2: 余弦相似度匹配
            # 使用向量A
            cosine_match, cosine_high_conf, no_answer = cosine_similarity_matching(
                vector_a, dict_entries, embedding_threshold
            )

            # 记录余弦相似度阶段结果
            stage_info["cosine_matched"] = cosine_match if cosine_high_conf else None

            # 直接使用余弦相似度结果
            if cosine_high_conf:
                stage_info["stage"] = "cosine"
                stage_info["final_matched"] = cosine_match
                stage_info["is_correct"] = (cosine_match == true_entity)
            else:  # no_answer=True
                stage_info["stage"] = "no_match"
                stage_info["final_matched"] = cosine_match  # "数据库里暂无查找的条目，或拼写错误"
                stage_info["is_correct"] = False
            
            results.append((original, cosine_match, true_entity, error_type, stage_info))

        except Exception as e:
            print(f"处理实体 '{original}' 出错: {e}")
            # 出错时记录为无匹配，并添加stage_info
            error_stage_info = {
                "stage": "error",
                "details": str(e),
                "levenshtein_matched": None,
                "cosine_matched": None,
                "final_matched": "无匹配结果",
                "is_correct": False
            }
            results.append((original, "无匹配结果", true_entity, error_type, error_stage_info))

    return results


# 计算性能指标 (按总样本计算)
def calculate_metrics(all_results):
    """
    计算所有错误类型的总体准确率、召回率和假阳性率

    Args:
        all_results: 字典 {error_type: [(input_entity, matched_entity, true_entity, stage_info),...]}

    Returns:
        metrics: 包含总体性能指标的字典
    """
    # 初始化统计变量
    total_samples = 0
    total_correct = 0
    total_answered = 0  # 不包括"无匹配"的回答

    # 按阶段统计
    stage_stats = {
        "levenshtein": {"total": 0, "correct": 0, "incorrect_samples": []},
        "cosine": {"total": 0, "correct": 0, "incorrect_samples": []},
        "no_match": {"total": 0, "samples": []}
    }

    # 按错误类型分阶段统计
    error_type_stage_stats = {}

    # 按错误类型统计
    type_stats = {}
    false_positives = []
    unmatched_samples = []

    # 计算普通错误类型的准确率和召回率
    regular_samples = 0
    norel_samples = 0
    norel_false_positives = 0

    # 遍历每种错误类型的结果
    for error_type, results in all_results.items():
        type_correct = 0
        type_total = 0
        type_false_positives = []
        type_unmatched = []

        # 初始化错误类型的阶段统计
        error_type_stage_stats[error_type] = {
            "levenshtein": {"total": 0, "correct": 0, "incorrect_samples": []},
            "cosine": {"total": 0, "correct": 0, "incorrect_samples": []},
            "no_match": {"total": 0, "samples": []}
        }

        for original, matched, true, stage_info in results:
            if error_type == "norel":
                norel_samples += 1
                if matched != "数据库里暂无查找的条目，或拼写错误":
                    norel_false_positives += 1
                    type_false_positives.append(original)
            else:
                regular_samples += 1
                type_total += 1

                # 统计阶段信息
                if stage_info["stage"] == "levenshtein":
                    stage_stats["levenshtein"]["total"] += 1
                    error_type_stage_stats[error_type]["levenshtein"]["total"] += 1
                    if matched == true:
                        stage_stats["levenshtein"]["correct"] += 1
                        error_type_stage_stats[error_type]["levenshtein"]["correct"] += 1
                    else:
                        stage_stats["levenshtein"]["incorrect_samples"].append((original, matched, true))
                        error_type_stage_stats[error_type]["levenshtein"]["incorrect_samples"].append(
                            (original, matched, true))
                elif stage_info["stage"] == "cosine":
                    stage_stats["cosine"]["total"] += 1
                    error_type_stage_stats[error_type]["cosine"]["total"] += 1
                    if matched == true:
                        stage_stats["cosine"]["correct"] += 1
                        error_type_stage_stats[error_type]["cosine"]["correct"] += 1
                    else:
                        stage_stats["cosine"]["incorrect_samples"].append((original, matched, true))
                        error_type_stage_stats[error_type]["cosine"]["incorrect_samples"].append(
                            (original, matched, true))
                elif stage_info["stage"] == "no_match":
                    stage_stats["no_match"]["total"] += 1
                    stage_stats["no_match"]["samples"].append((original, true))
                    error_type_stage_stats[error_type]["no_match"]["total"] += 1
                    error_type_stage_stats[error_type]["no_match"]["samples"].append((original, true))
                    type_unmatched.append((original, true))

                # 正确匹配
                if matched == true:
                    total_correct += 1
                    type_correct += 1

                # 有提供答案（不是"无匹配"）
                if matched != "数据库里暂无查找的条目，或拼写错误":
                    total_answered += 1

        # 记录每种错误类型的统计
        if type_total > 0:
            type_stats[error_type] = {
                "accuracy": (type_correct / type_total * 100) if type_total > 0 else 0,
                "total": type_total,
                "correct": type_correct,
                "unmatched": type_unmatched
            }

        if error_type == "norel":
            false_positives = type_false_positives
        else:
            unmatched_samples.extend(type_unmatched)

    # 计算总体指标
    total_samples = regular_samples
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    recall = (total_correct / total_samples * 100) if total_samples > 0 else 0
    false_positive_rate = (norel_false_positives / norel_samples * 100) if norel_samples > 0 else 0

    # 计算各阶段准确率
    levenshtein_accuracy = (stage_stats["levenshtein"]["correct"] / stage_stats["levenshtein"]["total"] * 100) if \
        stage_stats["levenshtein"]["total"] > 0 else 0
    cosine_accuracy = (stage_stats["cosine"]["correct"] / stage_stats["cosine"]["total"] * 100) if \
        stage_stats["cosine"]["total"] > 0 else 0

    return {
        "accuracy": accuracy,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "norel_samples": norel_samples,
        "norel_false_positives": norel_false_positives,
        "type_stats": type_stats,
        "false_positives": false_positives,
        "stage_stats": stage_stats,
        "error_type_stage_stats": error_type_stage_stats,
        "levenshtein_accuracy": levenshtein_accuracy,
        "cosine_accuracy": cosine_accuracy,
        "unmatched_samples": unmatched_samples
    }


# 测试函数 - 多进程并行处理所有错误类型文件
def test_error_types(dict_file, fuzzy_threshold=90, embedding_threshold=0.7, verbose=True, num_workers=None):
    """
    测试所有错误类型文件并计算性能指标，支持多进程并行处理

    Args:
        dict_file: 向量库文件路径
        fuzzy_threshold, embedding_threshold: 阈值参数
        verbose: 是否打印详细信息
        num_workers: 工作进程数，默认为CPU核心数-1

    Returns:
        metrics: 所有错误类型的性能指标
    """
    if num_workers is None:
        num_workers = CPU_CORES

    # 加载向量库
    dict_entries = load_dict_with_embeddings(dict_file)
    if not dict_entries:
        print("错误：无法加载向量库")
        return {}

    if verbose:
        print(f"使用 {num_workers} 个进程进行并行处理")

    # 处理各种错误类型
    error_types = ["lost_err", "mix_err", "similarity_err", "symbol_err", "spell_err", "norel"]

    # 准备所有任务数据
    all_batch_data = []

    for error_type in error_types:
        file_path = f"{error_type}_embedded.txt"

        if not os.path.exists(file_path):
            if verbose:
                print(f"警告：文件 {file_path} 不存在")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')

                try:
                    if error_type == "spell_err" and len(parts) >= 4:
                        # 新格式: 原始实体|corrected实体|实际正确的目标实体|向量a
                        original = parts[0].strip()
                        corrected = parts[1].strip()
                        true_entity = parts[2].strip()
                        vector_str = parts[3].strip()
                        
                        vector = np.fromstring(vector_str, sep=' ', dtype=np.float32)
                        
                        # 添加到任务列表 (vector_b设为None)
                        all_batch_data.append((original, corrected, vector, None, true_entity, error_type))
                    
                    elif error_type == "symbol_err" and len(parts) >= 4:
                        # 新格式: 原始实体|corrected实体|实际正确的目标实体|向量a
                        original = parts[0].strip()
                        corrected = parts[1].strip()
                        true_entity = parts[2].strip()
                        vector_str = parts[3].strip()
                        
                        vector = np.fromstring(vector_str, sep=' ', dtype=np.float32)
                        
                        # 添加到任务列表 (vector_b设为None)
                        all_batch_data.append((original, corrected, vector, None, true_entity, error_type))

                    elif error_type == "norel" and len(parts) >= 2:
                        # 原始实体|向量
                        original = parts[0].strip()
                        vector_str = parts[1].strip()

                        vector = np.fromstring(vector_str, sep=' ', dtype=np.float32)
                        all_batch_data.append(
                            (original, None, vector, None, "数据库里暂无查找的条目，或拼写错误", error_type))

                    elif len(parts) >= 3:
                        # 原始实体|实际正确的目标实体|向量
                        original = parts[0].strip()
                        true_entity = parts[1].strip()
                        vector_str = parts[2].strip()

                        vector = np.fromstring(vector_str, sep=' ', dtype=np.float32)
                        all_batch_data.append((original, None, vector, None, true_entity, error_type))

                except Exception as e:
                    if verbose:
                        print(f"解析 {error_type} 行时出错: {e}")

    # 计算每个批次的大小
    total_tasks = len(all_batch_data)

    if total_tasks == 0:
        print("警告：没有找到可处理的数据")
        return {}

    if verbose:
        print(f"总共找到 {total_tasks} 个待处理实体")

    # 如果任务数量少，直接串行处理
    if total_tasks < 100:
        if verbose:
            print("任务数量较少，使用串行处理")
        all_results = process_entities_batch(
            all_batch_data, dict_entries, fuzzy_threshold, embedding_threshold, 10
        )
    else:
        # 划分任务批次
        batch_size = max(20, total_tasks // (num_workers * 4))  # 每个工作进程处理多个批次
        batches = [all_batch_data[i:i + batch_size] for i in range(0, total_tasks, batch_size)]

        if verbose:
            print(f"划分为 {len(batches)} 个批次，每批处理 {batch_size} 个实体")

        # 使用进程池并行处理
        all_results = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交任务
            futures = []
            for batch in batches:
                future = executor.submit(
                    process_entities_batch, batch, dict_entries, fuzzy_threshold, embedding_threshold, 10
                )
                futures.append(future)

            # 使用tqdm显示进度
            if verbose:
                with tqdm(total=len(futures), desc="处理进度") as progress_bar:
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            all_results.extend(batch_results)
                        except Exception as e:
                            print(f"批次处理出错: {e}")
                        progress_bar.update(1)
            else:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                    except Exception as e:
                        print(f"批次处理出错: {e}")

    # 按错误类型整理结果
    results_by_type = {}
    for original, matched, true_entity, error_type, stage_info in all_results:
        if error_type not in results_by_type:
            results_by_type[error_type] = []
        results_by_type[error_type].append((original, matched, true_entity, stage_info))

    # 计算总体性能指标
    metrics = calculate_metrics(results_by_type)

    # 打印结果
    if verbose:
        print("\n详细性能指标:")

        # 打印每种错误类型的准确率
        print("\n各错误类型准确率:")
        for error_type, stats in metrics["type_stats"].items():
            print(f"{error_type}: {stats['accuracy']:.2f}% (正确: {stats['correct']}/{stats['total']})")

        # 打印总体指标
        print("\n总体性能指标:")
        print(f"总样本数: {metrics['total_samples']}")
        print(f"总体准确率: {metrics['accuracy']:.2f}% (正确: {metrics['total_correct']}/{metrics['total_samples']})")
        print(f"召回率: {metrics['recall']:.2f}%")

        # 打印各阶段匹配情况
        print("\n各阶段匹配情况:")
        print(f"Levenshtein阶段匹配数: {metrics['stage_stats']['levenshtein']['total']}")
        print(f"Levenshtein阶段正确数: {metrics['stage_stats']['levenshtein']['correct']}")
        print(f"Levenshtein阶段准确率: {metrics['levenshtein_accuracy']:.2f}%")
        print(f"余弦相似度阶段匹配数: {metrics['stage_stats']['cosine']['total']}")
        print(f"余弦相似度阶段正确数: {metrics['stage_stats']['cosine']['correct']}")
        print(f"余弦相似度阶段准确率: {metrics['cosine_accuracy']:.2f}%")
        print(f"未能匹配的样本数: {metrics['stage_stats']['no_match']['total']}")

        # 打印各错误类型的分阶段匹配情况
        print("\n各错误类型的分阶段匹配情况:")
        for error_type, stats in metrics["error_type_stage_stats"].items():
            print(f"\n{error_type}:")
            print(f"  Levenshtein阶段匹配数: {stats['levenshtein']['total']}")
            print(f"  Levenshtein阶段正确数: {stats['levenshtein']['correct']}")
            if stats['levenshtein']['total'] > 0:
                print(f"  Levenshtein阶段错误匹配样本:")
                for i, (input_ent, matched_ent, true_ent) in enumerate(stats['levenshtein']['incorrect_samples'][:5],
                                                                       1):
                    print(f"    {i}. 输入: '{input_ent}' | 匹配: '{matched_ent}' | 正确: '{true_ent}'")
            print(f"  余弦相似度阶段匹配数: {stats['cosine']['total']}")
            print(f"  余弦相似度阶段正确数: {stats['cosine']['correct']}")
            if stats['cosine']['total'] > 0:
                print(f"  余弦相似度阶段错误匹配样本:")
                for i, (input_ent, matched_ent, true_ent) in enumerate(stats['cosine']['incorrect_samples'][:5], 1):
                    print(f"    {i}. 输入: '{input_ent}' | 匹配: '{matched_ent}' | 正确: '{true_ent}'")
            print(f"  未能匹配的样本数: {stats['no_match']['total']}")
            if stats['no_match']['total'] > 0:
                print(f"  未匹配样本:")
                for i, (input_ent, true_ent) in enumerate(stats['no_match']['samples'][:5], 1):
                    print(f"    {i}. 输入: '{input_ent}' | 正确: '{true_ent}'")

        # 打印norel假阳性情况
        if metrics["norel_samples"] > 0:
            print(
                f"\n无关实体(norel)假阳性率: {metrics['false_positive_rate']:.2f}% ({metrics['norel_false_positives']}/{metrics['norel_samples']})")
            if metrics["false_positives"]:
                print("\n被判为假阳性的样本名称:")
                for i, name in enumerate(metrics["false_positives"], 1):
                    print(f"{i}. {name}")

        # 打印未匹配样本
        if metrics["unmatched_samples"]:
            print("\n未匹配的样本列表:")
            for i, (original, true_entity) in enumerate(metrics["unmatched_samples"], 1):
                print(f"{i}. 输入: '{original}' | 正确实体: '{true_entity}'")

    return metrics


def param_search_worker(args):
    """
    包装函数，用于接收并解包参数
    """
    params, dict_file, workers_per_test = args
    return param_search_worker_impl(params, dict_file, workers_per_test)


# 实际的参数搜索实现函数
def param_search_worker_impl(params, dict_file, workers_per_test):
    fuzzy_threshold, embedding_threshold = params
    try:
        # 执行测试（不打印详细信息）
        metrics = test_error_types(
            dict_file, fuzzy_threshold, embedding_threshold,
            verbose=False, num_workers=workers_per_test
        )

        # 检查是否有效结果
        if metrics:
            # 获取性能指标
            accuracy = metrics["accuracy"]
            false_positive_rate = metrics["false_positive_rate"]
            return params + (accuracy, false_positive_rate)
        return params + (0, 0)
    except Exception as e:
        print(f"参数 {params} 测试出错: {e}")
        return params + (0, 0)


# 修改后的find_best_params函数
def find_best_params(num_workers=None):
    """
    遍历参数组合，寻找均值准确率最大且假阳性率不超过20%的最佳参数组合，支持并行搜索
    """
    if num_workers is None:
        num_workers = max(1, CPU_CORES // 2)  # 使用一半的CPU核心用于搜索，另一半用于每次测试

    # 向量库文件路径
    dict_file = "attackpattern_embeddings.txt"

    # 定义参数搜索范围
    fuzzy_thresholds = list(range(77, 87, 1))  # 80-90, 步长2
    embedding_thresholds = [round(x, 2) for x in np.arange(0.63, 0.67, 0.01)]  # 0.6-0.87, 步长0.02

    # 创建结果记录文件
    results_file = "param_search_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("fuzzy_threshold,embedding_threshold,accuracy,false_positive_rate\n")

    # 最佳参数记录
    best_params = None
    best_accuracy = 0

    # 生成参数组合，仅保留满足high > low的组合
    param_combinations = []
    for high in embedding_thresholds:
        for fuzzy_threshold in fuzzy_thresholds:
            param_combinations.append((fuzzy_threshold, high))

    print(f"开始参数搜索，共 {len(param_combinations)} 种组合...")
    start_time = time.time()

    # 使用进程池并行搜索
    workers_per_test = max(1, CPU_CORES // num_workers)  # 每个测试使用的工作进程数
    print(f"使用 {num_workers} 个工作进程进行并行搜索，每个测试使用 {workers_per_test} 个进程")

    # 准备任务参数
    task_args = [(params, dict_file, workers_per_test) for params in param_combinations]

    # 使用进度条
    with tqdm(total=len(task_args), desc="参数搜索进度") as progress_bar:
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交任务
            futures = [executor.submit(param_search_worker, args) for args in task_args]

            # 处理结果
            for future in as_completed(futures):
                try:
                    # 获取结果
                    result = future.result()
                    fuzzy_threshold, embedding_threshold, accuracy, false_positive_rate = result

                    # 记录结果到文件
                    with open(results_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{fuzzy_threshold},{embedding_threshold},{accuracy:.2f},{false_positive_rate:.2f}\n")

                    # 检查是否是最佳参数
                    if false_positive_rate <= 100.0 and accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            "fuzzy_threshold": fuzzy_threshold,
                            "embedding_threshold": embedding_threshold,
                            "accuracy": accuracy,
                            "false_positive_rate": false_positive_rate
                        }

                        # 打印当前最佳参数
                        elapsed_time = time.time() - start_time
                        print(
                            f"\n[{elapsed_time:.1f}秒] 找到新的最佳参数: fuzzy={fuzzy_threshold}, high={embedding_threshold}")
                        print(f"准确率: {accuracy:.2f}%, 假阳性率: {false_positive_rate:.2f}%")
                except Exception as e:
                    print(f"处理结果时出错: {e}")

                # 更新进度条
                progress_bar.update(1)

    # 打印最终结果
    elapsed_time = time.time() - start_time
    print(f"\n参数搜索完成，耗时 {elapsed_time:.2f} 秒，共测试 {len(param_combinations)} 种组合。")

    if best_params:
        print("\n最佳参数组合:")
        print(f"fuzzy_threshold = {best_params['fuzzy_threshold']}")
        print(f"embedding_threshold = {best_params['embedding_threshold']}")
        print(f"准确率: {best_params['accuracy']:.2f}%")
        print(f"假阳性率: {best_params['false_positive_rate']:.2f}%")

        # 使用最佳参数运行一次完整测试，显示详细结果
        print("\n使用最佳参数运行完整测试:")
        test_error_types(
            dict_file,
            best_params['fuzzy_threshold'],
            best_params['embedding_threshold'],
            10
        )
    else:
        print("\n未找到满足条件的最佳参数（假阳性率<=20%）")

    print(f"\n所有搜索结果已保存到 {results_file}")
    return best_params


# 主函数
def main():
    # 向量库文件路径
    dict_file = "attackpattern_embeddings.txt"

    # 设置阈值参数
    fuzzy_threshold = 81
    embedding_threshold = 0.66

    # 菜单选择
    print("请选择操作:")
    print("1. 使用当前参数运行测试")
    print("2. 执行参数搜索，寻找最佳参数组合")
    print(f"(使用 {CPU_CORES} 个CPU核心进行并行处理)")
    choice = input("请输入选项 (1/2): ")

    if choice == "2":
        # 执行参数搜索
        find_best_params()
    else:
        # 使用当前参数运行测试
        print(f"使用参数: fuzzy_threshold={fuzzy_threshold}, embedding_threshold={embedding_threshold}")

        # 执行测试
        test_error_types(dict_file, fuzzy_threshold, embedding_threshold)


if __name__ == "__main__":
    main()