from flask import Flask, request, jsonify, Response
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time
import json
from functools import wraps
from threading import Lock
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import asyncio

# 加载 .env 文件
load_dotenv()

# 从 .env 文件获取 API keys
huggingface_api_keys = [key.strip() for key in os.getenv("HUGGINGFACE_API_KEY", "").split(",") if key.strip()]
api_key = os.getenv("API_KEY")
# 获取每个 key 的并发数
concurrent_per_key = int(os.getenv("CONCURRENT_PER_KEY", "5"))

class APIKeyManager:
    def __init__(self, api_keys, concurrent_per_key):
        if not api_keys:
            raise ValueError("至少需要提供一个 API key")
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = Lock()
        # 为每个 API key 创建一个请求队列
        self.client_pools = {}
        # 设置线程池
        self.executor = ThreadPoolExecutor(max_workers=len(api_keys) * concurrent_per_key)
        
    def get_client(self, model):
        with self.lock:
            try:
                if not self.api_keys:
                    raise ValueError("API keys 列表为空")
                
                self.current_index = self.current_index % len(self.api_keys)
                current_key = self.api_keys[self.current_index]
                
                # 为每个请求创建新的 client 实例
                client = InferenceClient(
                    model=model,
                    api_key=current_key
                )
                
                # 更新索引，实现轮询
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                return client
            except Exception as e:
                print(f"获取 API client 时发生错误: {str(e)}")
                # 如果出错，尝试使用第一个可用的 key
                first_key = self.api_keys[0] if self.api_keys else None
                if first_key:
                    return InferenceClient(
                        model=model,
                        api_key=first_key
                    )
                return None

    def execute_request(self, model, messages, temperature, max_tokens, top_p, stream):
        client = self.get_client(model)
        if not client:
            raise ValueError("无法获取可用的 API client")
            
        return client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream
        )

# 创建 API key 管理器实例
key_manager = APIKeyManager(huggingface_api_keys, concurrent_per_key)

app = Flask(__name__)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        provided_key = request.headers.get('Authorization')
        if provided_key:
            if provided_key.startswith("Bearer "):
                provided_key = provided_key.split("Bearer ")[1]
            if provided_key == api_key:
                return f(*args, **kwargs)
        return jsonify({"error": "Invalid or missing API key"}), 401
    return decorated

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    model = data.get('model', 'gpt2')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 8196)
    top_p = min(max(data.get('top_p', 0.9), 0.0001), 0.9999)
    stream = data.get('stream', False)

    try:
        # 直接在线程池中执行同步请求
        future = key_manager.executor.submit(
            key_manager.execute_request,
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stream
        )
        response = future.result()  # 等待结果

        if stream:
            def generate():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(generate(), mimetype='text/event-stream')
        else:
            content = response.choices[0].message.content
            return jsonify({
                'id': f'chatcmpl-{int(time.time())}',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': model,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': content
                        },
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': -1,
                    'completion_tokens': -1,
                    'total_tokens': -1
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)