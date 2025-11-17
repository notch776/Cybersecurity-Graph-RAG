import random

# 读取原始文本文件
input_file = 'attackpattern.txt'  # 替换为您的输入文件名
output_file = 'picked.txt'

# 读取所有词组
with open(input_file, 'r', encoding='utf-8') as f:
    phrases = f.readlines()

# 去掉每个词组末尾的换行符
phrases = [phrase.strip() for phrase in phrases]

# 随机选取100条词组
picked_phrases = random.sample(phrases, 100)

# 将选中的词组写入到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    for phrase in picked_phrases:
        f.write(phrase + '\n')

print(f"已随机选取100条词组并保存到 {output_file}。")
