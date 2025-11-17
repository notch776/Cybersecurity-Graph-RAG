import os
import re
from volcenginesdkarkruntime import Ark
import json

# 从环境变量中读取方舟API Key
client = Ark(api_key=os.environ.get("ARK_API_KEY"))

def process_entities_with_llm():
    """
    读取spell.txt文件，将有拼写错误的实体发送给大模型进行处理，
    并根据返回结果更新格式。

    输出格式为：
    "原始实体|corrected实体|正确实体"
    
    当大模型无法提供纠正时，corrected实体部分输出为"."
    这与matchtest_cpu_resource.py的处理逻辑兼容：
    - 如果corrected实体为"."，只用原始实体进行Levenshtein计算
    - 如果corrected实体不为"."，则用原始实体和corrected实体都进行计算，取分数最高的
    """
    input_file = "spell.txt"
    output_file = "corrected_spell.txt"
    results = []
    
    # 读取spell.txt文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    
    total_lines = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 解析行内容：错误实体|正确实体
        parts = line.split('|')
        if len(parts) != 2:
            print(f"警告：第{i+1}行格式不正确: {line}")
            results.append(line)
            continue
            
        misspelled_entity, correct_entity = parts
        
        # 构建问题
        question = f"{misspelled_entity}的攻击方式是什么？"
        
        # 构建发送给大模型的问句
        prompt = f"""请从以下问题中识别出提及的实体及其最可能的类型。
问题: "{question}"
可能的实体类型: "attackpattern", "skill", "consequences", "indicator", "prerequisite"。
请以JSON格式返回结果，键是实体类型，值是实体名称。例如：{{"attackpattern": ["SQL Injection","XSS Using Alternate Syntax"], "skill": "Commercial tools are available"}}。
如果找不到明确的实体或类型，请返回空的JSON对象 {{}}。如果你认为识别出的实体名称有错别字，拼写错误或符号上的问题，请在原实体名称后加上你认为正确的实体名称。如：{{"attackpattern":["sql 注人|sql 注入","Signture Spof|Signature Spoof","Byp@ss!ng@Phy$ica|#L0cks|Bypassing Physical Locks","Fuzzing'|Fuzzing"]}}。"""
        
        
        try:
            # 调用大模型API
            completion = client.chat.completions.create(
                model="deepseek-v3-250324",
                messages=[
                    {"role": "system", "content": "你是一个帮助识别网络安全问题文本中的实体的助手，实体类型限制在给定列表中。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 获取大模型的回复
            llm_response = completion.choices[0].message.content
            
            # 尝试解析JSON
            try:
                # 从回复中提取JSON部分
                json_str = re.search(r'```json\n(.*?)\n```', llm_response, re.DOTALL)
                if json_str:
                    json_str = json_str.group(1)
                else:
                    json_str = llm_response
                
                # 解析JSON
                entities = json.loads(json_str)
                
                # 检查是否有包含管道符"|"的实体名称，表示纠正
                llm_correction = None
                for entity_type, entity_values in entities.items():
                    if isinstance(entity_values, list):
                        for entity in entity_values:
                            if isinstance(entity, str) and "|" in entity:
                                parts = entity.split("|")
                                if len(parts) == 2 and parts[0].strip() == misspelled_entity:
                                    llm_correction = parts[1].strip()
                                    break
                    elif isinstance(entity_values, str) and "|" in entity_values:
                        parts = entity_values.split("|")
                        if len(parts) == 2 and parts[0].strip() == misspelled_entity:
                            llm_correction = parts[1].strip()
                            break
                
                if llm_correction:
                    # 更新为"有拼写错误的实体|b|正确实体"格式
                    updated_line = f"{misspelled_entity}|{llm_correction}|{correct_entity}"
                else:
                    # 没有找到纠正，使用"有拼写错误的实体|.|正确实体"格式
                    updated_line = f"{misspelled_entity}|.|{correct_entity}"
                
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"JSON解析错误: {e}")
                # 尝试使用正则表达式查找a|b格式
                pipe_pattern = re.search(r'([^|]+)\|([^|]+)', llm_response)
                
                if pipe_pattern:
                    # 提取大模型的纠正结果
                    llm_correction = pipe_pattern.group(2).strip()
                    # 更新为"有拼写错误的实体|b|正确实体"格式
                    updated_line = f"{misspelled_entity}|{llm_correction}|{correct_entity}"
                else:
                    # 没有找到a|b格式的纠正，使用"有拼写错误的实体|.|正确实体"格式
                    updated_line = f"{misspelled_entity}|.|{correct_entity}"
            
            results.append(updated_line)
            print(f"处理进度：{i+1}/{total_lines} - {misspelled_entity}")
            
        except Exception as e:
            print(f"处理实体 '{misspelled_entity}' 时出错: {e}")
            # 出错时保持原格式
            results.append(line)
    
    # 将结果写入新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result}\n")
        print(f"处理完成，结果已保存至 {output_file}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}")

if __name__ == "__main__":
    process_entities_with_llm()
