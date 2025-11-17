# 文件名
input_file = 'similarity_err_embedded.txt'
output_file = 'similarity_err.txt'

# 打开输入文件并读取内容
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 处理每一行，去除第三个字段
modified_lines = []
for line in lines:
    # 去除行末的换行符，并按 '|' 分割
    parts = line.strip().split('|')

    # 检查分割后的部分是否足够
    if len(parts) >= 3:
        # 去掉第三个部分
        modified_line = '|'.join([parts[1]])
        modified_lines.append(modified_line)
    else:
        print(f"行格式错误: {line.strip()}")  # 打印格式错误的行

# 将修改后的内容写入新的文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for modified_line in modified_lines:
        outfile.write(modified_line + '\n')

print(f"已成功生成文件 {output_file}，去除了每行的第三个字段。")
