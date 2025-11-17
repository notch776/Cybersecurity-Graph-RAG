#!/usr/bin/env python
# coding: utf-8

# ## answer_search.py

# In[1]:

import os
import ahocorasick
from fuzzywuzzy import process,fuzz
import re
from py2neo import Graph
class AnswerSearcher:
    def __init__(self):
        #self.g = Graph("bolt://localhost:7687", user="neo4j", password="1234")
        #self.g = Graph("neo4j+ssc://dea5d244.databases.neo4j.io:7687", auth=("neo4j", "kK7WKtx7UMX2pECguU5FvJg5cMHgjNg4q-jte-b_Au4"))
        self.num_limit = 20

    '''执行cypher查询，并返回相应结果'''
    def search_main(self, sqls):
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']
            answers = []
            for query in queries:
                ress = self.g.run(query).data()
                answers += ress
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    '''根据对应的qustion_type，调用相应的回复模板'''
    def answer_prettify(self, question_type, answers):
        final_answer = []
        if not answers:
            return ''
        
        if question_type == 'attack_skill':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}攻击模式用到的攻击技巧:{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        
        elif question_type == 'attack_pattern_same':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '与之同类型的攻击模式：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'attack_pre':
            desc = [i['n.type'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的攻击条件有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'attack_indicator':
            desc = [i['n.type'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的攻击迹象有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'attack_child':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '在{0}攻击模式中包含了下面的子攻击模式：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'attack_follow':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}攻击的下一步是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        
        elif question_type == 'attack_cause':
            desc = [i['n.type'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}攻击的结果是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
            
        elif question_type == 'attack_proceed':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}攻击可能会导致以下其他攻击：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
            
        return final_answer

# ## question_parser.py

# In[2]:
class QuestionPaser:

    '''构建实体节点'''
    def build_entitydict(self, args):
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)

        return entity_dict

    '''解析主函数'''
    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']
        sqls = []
        for question_type in question_types:
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []
            if question_type == 'attack_skill':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))
                
            elif question_type == 'attack_pattern_same':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_pre':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_indicator':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_child':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_follow':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_cause':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))

            elif question_type == 'attack_proceed':
                sql = self.sql_transfer(question_type, entity_dict.get('CyberAttackPattern'))
   
            if sql:
                sql_['sql'] = sql

                sqls.append(sql_)

        return sqls

    '''针对不同的问题，分开进行处理'''
    def sql_transfer(self, question_type, entities):

        # 查询语句
        sql = []
        # 已知忌口查疾病
        if question_type == 'attack_skill':
            sql = ["MATCH (m:CyberAttackPattern)-[r:REQUIRES_SKILL]->(n:Skill) where m.name = '{0}' return m.name, n.name".format(i) for i in entities]

        elif question_type == 'attack_pattern_same':
            sql = ["MATCH (m:CyberAttackPattern)-[r:PEER_OF]-(n:CyberAttackPattern) WHERE m.name = '{0}' RETURN m.name, n.name".format(i) for i in entities]


        # 查询疾病的持续时间
        elif question_type == 'attack_pre':
            sql = ["MATCH (m:CyberAttackPattern)-[r:HAS_PREREQUISITE]->(n:Prerequisite) where m.name = '{0}' return m.name, n.type".format(i) for i in entities]

        # 查询疾病的治愈概率
        elif question_type == 'attack_indicator':
            sql = ["MATCH (m:CyberAttackPattern)-[r:HAS_INDICATOR]->(n:Indicator) where m.name = '{0}' return m.name, n.type".format(i) for i in entities]

        # 查询疾病的治疗方式
        elif question_type == 'attack_child':
            sql = ["MATCH (m:CyberAttackPattern)-[r:HAS_CHILD]->(n:CyberAttackPattern) where m.name = '{0}' return m.name, n.name".format(i) for i in entities]

        # 查询疾病的易发人群
        elif question_type == 'attack_follow':
            sql = ["MATCH (m:CyberAttackPattern)-[r:FOLLOWS]->(n:CyberAttackPattern) where m.name = '{0}' return m.name, n.name".format(i) for i in entities]

        # 查询疾病的相关介绍
        elif question_type == 'attack_cause':
            sql = ["MATCH (m:CyberAttackPattern)-[r:CAUSES]->(n:Consequences) where m.name = '{0}' return m.name, n.type".format(i) for i in entities]

        # 查询疾病有哪些症状
        elif question_type == 'attack_proceed':
            sql = ["MATCH (m:CyberAttackPattern)-[r:CAN_PROCEED]->(n:CyberAttackPattern) where m.name = '{0}' return m.name, n.name".format(i) for i in entities]

        return sql

# ## question_classifier.py

# In[3]:
class QuestionClassifier:
    def __init__(self):
        # cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        cur_dir = os.getcwd()
        #　特征词路径
        self.attackpattern_path = os.path.join( r'C:/Users/A/Desktop/dc/attackpattern.txt')
        self.skill_path = os.path.join( r'C:/Users/A/Desktop/dc/skill.txt')
        self.consequences_path = os.path.join( r'C:/Users/A/Desktop/dc/consequences.txt')
        self.indicator_path = os.path.join( r'C:/Users/A/Desktop/dc/indicator.txt')
        self.prerequisite_path = os.path.join( r'C:/Users/A/Desktop/dc/prerequisite.txt')

        # 加载特征词
        self.attackpattern_wds= [i.strip() for i in open(self.attackpattern_path,encoding="utf-8") if i.strip()]#encoding="utf-8"
        self.skill_wds= [i.strip() for i in open(self.skill_path,encoding="utf-8") if i.strip()]
        self.consequences_wds= [i.strip() for i in open(self.consequences_path,encoding="utf-8") if i.strip()]
        self.indicator_wds= [i.strip() for i in open(self.indicator_path,encoding="utf-8") if i.strip()]
        self.prerequisite_wds= [i.strip() for i in open(self.prerequisite_path,encoding="utf-8") if i.strip()]
        self.region_words = set(self.attackpattern_wds + self.skill_wds + self.consequences_wds + self.indicator_wds + self.prerequisite_wds)
#         self.deny_words = [i.strip() for i in open(self.deny_path,encoding="utf-8") if i.strip()]
#         # 构造领域actree
        self.region_tree = self.build_actree(list(self.region_words))
        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()
        # 问句疑问词
        self.skill_qwds = ['技巧', '技能', 'skill']
        self.consequences_qwds = ['结果', '后果','consequence', 'consequences']
        self.attackpatternsame_qwds = [ '同','same']
        self.prerequisite_qwds = ['前提', '条件', 'prerequisite','prerequisites']
        self.indicator_qwds = ['迹象', '显示', '表明','痕迹','表现','征兆','show','indicator']
        self.follow_qwds = ['下一步', '接着','之后','后面', 'follow','follows']
        self.child_qwds = ['子', '中','本质','核心']
        self.proceed_qwds = ['其他攻击','其他的攻击','另外的攻击']
        print('model init finished ......')

        return

    '''分类主函数'''
    def classify(self, question):
        data = {}
        medical_dict = self.check_medical(question)
        if not medical_dict:
            return {}
        data['args'] = medical_dict
        #收集问句当中所涉及到的实体类型
        types = []
        for type_ in medical_dict.values():
            types += type_
        question_type = 'others'

        question_types = []

        # 症状
        if self.check_words(self.skill_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_skill'
            question_types.append(question_type)
            
        elif self.check_words(self.attackpatternsame_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_pattern_same'
            question_types.append(question_type)
            
        elif self.check_words(self.prerequisite_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_pre'
            question_types.append(question_type)
            
        elif self.check_words(self.indicator_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_indicator'
            question_types.append(question_type)
            
        elif self.check_words(self.child_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_child'
            question_types.append(question_type)
            
        elif self.check_words(self.follow_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_follow'
            question_types.append(question_type)
            
        elif self.check_words(self.consequences_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_cause'
            question_types.append(question_type)
            
        elif self.check_words(self.proceed_qwds, question) and ('CyberAttackPattern' in types):
            question_type = 'attack_proceed'
            question_types.append(question_type)

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''构造词对应的类型'''
    def build_wdtype_dict(self):
        wd_dict = dict()
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.skill_wds:
                wd_dict[wd].append('Skill')
            if wd in self.attackpattern_wds:
                wd_dict[wd].append('CyberAttackPattern')
            if wd in self.prerequisite_wds:
                wd_dict[wd].append('Prerequisite')
            if wd in self.consequences_wds:
                wd_dict[wd].append('Consequences')
            if wd in self.indicator_wds:
                wd_dict[wd].append('Indicator')
        return wd_dict

    '''构造actree，加速过滤'''
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def check(self, lmatches):
        with open(self.attackpattern_path, 'r') as file:
            file.seek(0)
            names = file.read().splitlines()
        best_matches = []  # 使用列表来收集匹配项，这样可以保持添加的顺序
        best_matches_set = set()  # 使用集合辅助去重，但不直接用于最终结果
        for match in lmatches:
            matches = self.find_closest_match(match, names)
            # 检查最佳匹配项的分数是否不等于100
            if matches and matches[0][1] < 100:
                # 如果匹配项还没有在集合中，说明是一个新的不重复项，那么将其添加到集合和列表中
                if matches[0][0] not in best_matches_set:
                    best_matches_set.add(matches[0][0])  # 用于去重检查
                    best_matches.append(matches[0][0])  # 保持顺序的列表
        return best_matches

    '''问句过滤'''
    def check_medical(self, question):
        region_wds = []
        for i in self.region_tree.iter(question):
            wd = i[1][1]
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i:self.wdtype_dict.get(i) for i in final_wds}

        return final_dict
    
    def find_closest_match(self, query, choices, limit=1):
    # 使用fuzzywuzzy找到最接近的匹配项
        results =process.extract(query, choices, limit=limit,scorer=fuzz.ratio,processor=None)
        return results
    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False
# ## 实例化辅助函数

# In[5]:
classifier = QuestionClassifier()

parser = QuestionPaser()

searcher = AnswerSearcher()

# In[7]:

def handle(question):
    pattern = re.compile(r'[a-zA-Z0-9]+(?:[\s-]+[a-zA-Z0-9]+)*')
    lmatches = pattern.findall(question)
    list=classifier.check(lmatches)
    if len(list):
        matches_str = ", ".join(list)
        # 将转换后的字符串插入到句子中并打印
        string=f"似乎你拼写的不太正确呢！猜你想搜：{matches_str}"
        return string
    else:
           res_classify = classifier.classify(question)
           res_cypher = parser.parser_main(res_classify)
           final_answers = searcher.search_main(res_cypher)
           return final_answers


