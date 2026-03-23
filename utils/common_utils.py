import hashlib
import json
import re
from datetime import datetime

def generate_conversation_id(user_id, costumer_id):
    """生成会话ID"""
    sorted_ids = sorted([str(user_id), str(costumer_id)])
    combined = f"{sorted_ids[0]}|{sorted_ids[1]}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def extract_json_objects(text):
    """从文本中提取所有JSON对象"""
    json_objects = []
    
    # 匹配完整的JSON对象
    json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\[.*\]|\{.*?\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            json_objects.append(data)
        except json.JSONDecodeError:
            continue
    
    return json_objects

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")