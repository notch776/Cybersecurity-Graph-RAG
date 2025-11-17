from flask import Flask, request, jsonify, render_template
from myscript import handle
from llmm_could_run import handle2
from llmm_without_sonquestion import handle3
from llmm_sub_question import handle5
from llmm import handle6
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# 假设这是您要初始化的模型
embedding_model = None

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

@app.route('/', methods=['GET'])
def show():
    # 这个视图函数现在只处理GET请求，显示页面
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
    # 获取前端发送的数据
    data = request.json['data']
    # 将 embedding_model 作为参数传递给 handle2
    result = handle6(data, embedding_model)  # 传递模型
    # 返回处理结果
    return jsonify(result=result)

if __name__ == '__main__':
    embedding_model, device = init_model()  # 在应用启动时初始化模型
    if embedding_model is not None:
        app.run(debug=True)
    else:
        print("无法启动 Flask 应用，因为模型加载失败。")
