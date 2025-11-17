import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm


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


def calculate_similarity(model, text_a, text_b):
    """计算两个文本的相似度"""
    # 获取文本的向量表示
    vec_a = model.encode(text_a, convert_to_numpy=True)
    vec_b = model.encode(text_b, convert_to_numpy=True)

    # 计算并返回余弦相似度
    return cosine_similarity(vec_a, vec_b)


# 主程序
if __name__ == "__main__":
    # 初始化模型
    model, device = init_model()

    # 定义两个待比较的文本
    text_a = "The majority party in Australia (where Canberra is located) is the Australian Labor Party (ALP), led by Prime Minister Anthony Albanese, as of the 2022 federal election."
    text_b = "politics "

    # 计算相似度
    similarity = calculate_similarity(model, text_a, text_b)

    # 输出相似度
    print(f"文本相似度: {similarity:.4f}")
