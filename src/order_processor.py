# src/order_processor.py

import os
import sys
import json
import logging
import configparser
import hashlib
from datetime import datetime, timedelta
from time import sleep
import requests
import urllib3
import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ====================== 配置部分 ======================
def get_base_path():
    return os.getenv('SERVICE_BASE_PATH', r'C:\CODE\compliance_1202')

# 线程锁，用于保证日志输出的线程安全
log_lock = threading.Lock()

# 忽略HTTPS警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====================== 加解密函数 ======================
def rsa_encrypt(data, public_key_path):
    with open(public_key_path, 'r') as f:
        key = RSA.import_key(f.read())
    cipher = Cipher_pkcs1_v1_5.new(key)
    default_length = 245
    data_bytes = data.encode('utf-8')
    encrypted = b''.join([cipher.encrypt(data_bytes[i:i+default_length]) for i in range(0, len(data_bytes), default_length)])
    return base64.b64encode(encrypted).decode('utf-8')

def aes_decrypt(data, key):
    cipher = AES.new(key.ljust(32, '\0').encode('utf-8'), AES.MODE_ECB)
    decrypted = cipher.decrypt(base64.b64decode(data))
    try:
        return decrypted.decode('utf-8').strip()
    except Exception:
        return decrypted.strip()

def aes_en(plain_text: str, key: str) -> str:
    try:
        cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
        encrypted_bytes = cipher.encrypt(pad(plain_text.encode('utf-8'), AES.block_size))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        with log_lock:
            logging.error(f"加密失败: {e}")
        return None

# ====================== 配置读取与校验 ======================
def load_and_validate_config(BASE_PATH):
    CONFIG_FILE = os.path.join(BASE_PATH, 'config.ini')
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    
    required_api_keys = [
        'url_get_order_list', 'url_get_chat_log', 'url_submit_encrypted',
        'appsecret', 'agent', 'appid', 'preprocess', 'aes_DEFAULT_KEY'
    ]
    required_key_paths = ['public_key_path', 'private_key_path']
    
    missing_api_keys = [k for k in required_api_keys if not config.has_option('API', k)]
    missing_key_paths = [k for k in required_key_paths if not config.has_option('Keys', k)]
    
    if missing_api_keys or missing_key_paths:
        msg = ""
        if missing_api_keys:
            msg += f"API 配置缺少参数: {', '.join(missing_api_keys)}\n"
        if missing_key_paths:
            msg += f"密钥路径配置缺少参数: {', '.join(missing_key_paths)}"
        with log_lock:
            logging.error(msg.strip())
        sys.exit(1)
    
    return config

def check_key_files(public_key_path, private_key_path):
    if not os.path.exists(public_key_path):
        with log_lock:
            logging.error(f"公钥文件未找到: {public_key_path}")
        sys.exit(1)
    if not os.path.exists(private_key_path):
        with log_lock:
            logging.error(f"私钥文件未找到: {private_key_path}")
        sys.exit(1)

# ====================== 订单列表获取功能 ======================
def fetch_and_decrypt_order_list(start_time_string, end_time_string, order_range, public_key_path, private_key_path, url, appsecret, agent, appid):
    timestamp = str(int(time.time()))
    nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[4:16]
    start_dt = datetime.strptime(start_time_string, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S")
    data = {
        'start': str(int(start_dt.timestamp())),
        'end': str(int(end_dt.timestamp())),
        'range': order_range
    }
    sign_str = '&'.join([f"{k}={v}" for k, v in sorted(data.items())]) + f"&appsecret={appsecret}"
    sign = hashlib.md5(sign_str.encode()).hexdigest().upper()
    headers = {
        'Agent': agent,
        'Content-Type': 'application/json;charset=utf-8',
        'Appid': appid,
        'Timestamp': timestamp,
        'Nonce': nonce,
        'Sign': sign
    }
    encrypted_data = rsa_encrypt(json.dumps(data), public_key_path)
    response = requests.post(url, headers=headers, data=encrypted_data, verify=False)
    result = response.json()
    
    if result.get('errcode') == 0:
        data_block = None
        if not result.get('data'):
            return []
        if isinstance(result['data'], list) and result['data']:
            data_block = result['data'][0]
        elif isinstance(result['data'], dict):
            data_block = result['data']
        else:
            return []
        
        with open(private_key_path, 'r') as f:
            key = RSA.import_key(f.read())
        cipher = Cipher_pkcs1_v1_5.new(key)
        aes_key = cipher.decrypt(base64.b64decode(data_block['encryptKey']), None).decode('utf-8')
        decrypted_result = aes_decrypt(data_block['encryptData'], aes_key)
        order_number_list = validate_and_parse_json(decrypted_result)[0]
        return order_number_list
    return []

def validate_and_parse_json(decrypted_result):
    try:
        return [json.loads(decrypted_result)]
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            pos = e.pos
            valid_json, extra_data = decrypted_result[:pos], decrypted_result[pos:]
            result = [json.loads(valid_json)]
            extra_data = extra_data.lstrip(', ')
            if extra_data:
                result.extend(validate_and_parse_json(extra_data))
            return result
        return []

# ====================== 聊天记录处理功能 ======================
def extract_user_text_content(data):
    extracted_contents = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'messages' in item and isinstance(item['messages'], list):
                    for message in item['messages']:
                        if (isinstance(message, dict) and 
                            message.get('entity_type') == 'user' and 
                            message.get('msg_type') == 'text' and 
                            'content' in message):
                            extracted_contents.append(message['content'])
                elif (item.get('entity_type') == 'user' and 
                      item.get('msg_type') == 'text' and 
                      'content' in item):
                    extracted_contents.append(item['content'])
    elif isinstance(data, dict):
        if 'messages' in data and isinstance(data['messages'], list):
            for message in data['messages']:
                if (isinstance(message, dict) and 
                    message.get('entity_type') == 'user' and 
                    message.get('msg_type') == 'text' and 
                    'content' in message):
                    extracted_contents.append(message['content'])
        elif (data.get('entity_type') == 'user' and 
              data.get('msg_type') == 'text' and 
              'content' in data):
            extracted_contents.append(data['content'])
    return ' '.join(extracted_contents)

def validate_and_parse_chat_data(decrypted_result):
    try:
        if isinstance(decrypted_result, str):
            data = json.loads(decrypted_result)
        elif isinstance(decrypted_result, (dict, list)):
            data = decrypted_result
        else:
            raise ValueError(f"不支持的数据类型: {type(decrypted_result)}")
        
        user_content = extract_user_text_content(data)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                data[0]['user_concatenated_content'] = user_content
            return data
        elif isinstance(data, dict):
            data['user_concatenated_content'] = user_content
            return [data]
        else:
            raise ValueError(f"无法处理的数据结构: {type(data)}")
    except Exception as e:
        with log_lock:
            logging.error(f"数据验证失败: {str(e)} | 原始数据: {decrypted_result}")
        return []

def generate_conversation_id(a_id, b_id):
    sorted_ids = sorted([a_id, b_id])
    combined = f"{sorted_ids[0]}|{sorted_ids[1]}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def replace_with_conversation_id(data):
    for item in data:
        user_id = item.get('user_id')
        costumer_id = item.get('costumer_id')
        if user_id and costumer_id:
            conv_id = generate_conversation_id(user_id, costumer_id)
            new_item = {'conversation_id': conv_id}
            new_item.update(item)
            item.clear()
            item.update(new_item)
    return data

def fetch_and_decrypt_chat_log(order_id, public_key_path, private_key_path, url, appsecret, agent, appid):
    timestamp = str(int(time.time()))
    nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[4:16]
    data = {'order_id': order_id}
    sign_str = '&'.join([f"{k}={v}" for k, v in sorted(data.items())]) + f"&appsecret={appsecret}"
    sign = hashlib.md5(sign_str.encode()).hexdigest().upper()
    headers = {
        'Agent': agent,
        'Content-Type': 'application/json;charset=utf-8',
        'Appid': appid,
        'Timestamp': timestamp,
        'Nonce': nonce,
        'Sign': sign
    }
    encrypted_data = rsa_encrypt(json.dumps(data), public_key_path)
    response = requests.post(url, headers=headers, data=encrypted_data, verify=False)
    try:
        result = response.json()
    except json.JSONDecodeError:
        return f"响应格式错误: {response.text}"
    
    if result.get('errcode') == 0:
        if 'data' in result and 'encryptKey' in result['data'] and 'encryptData' in result['data']:
            with open(private_key_path, 'r') as f:
                key = RSA.import_key(f.read())
            cipher = Cipher_pkcs1_v1_5.new(key)
            aes_key = cipher.decrypt(base64.b64decode(result['data']['encryptKey']), None).decode('utf-8')
            decrypted_result = aes_decrypt(result['data']['encryptData'], aes_key)
            return decrypted_result
        else:
            return f"订单 {order_id} 返回数据结构不完整"
    else:
        return f"订单 {order_id} 接口调用失败: errcode={result.get('errcode')}, errmsg={result.get('errmsg', 'Unknown error')}"

# ====================== AI分析功能 ======================
def analyze_ai_response(order_id, response_json):
    pass  # 可保留或简化，此处仅需生成日志，AI分析非必须

def process_single_order(order_id, config, BASE_PATH, public_key_path, private_key_path, progress_info=""):
    url_get_chat_log = config.get('API', 'url_get_chat_log')
    url_submit = config.get('API', 'url_submit_encrypted')
    appsecret = config.get('API', 'appsecret')
    agent = config.get('API', 'agent')
    appid = config.get('API', 'appid')
    preprocess = int(config.get('API', 'preprocess'))
    aes_DEFAULT_KEY = config.get('API', 'aes_DEFAULT_KEY')
    
    try:
        decrypted_result = fetch_and_decrypt_chat_log(
            order_id, public_key_path, private_key_path,
            url_get_chat_log, appsecret, agent, appid
        )
        parsed_data = validate_and_parse_chat_data(decrypted_result)
        if not parsed_data:
            return order_id, False, "聊天记录为空"
        
        json_content = replace_with_conversation_id(parsed_data)
        if isinstance(json_content[0], dict):
            json_content[0]['preprocess'] = preprocess
        else:
            return order_id, False, "JSON 数据格式不正确"
        
        plain_text = json.dumps(json_content[0], ensure_ascii=False)
        encrypted_data = aes_en(plain_text, aes_DEFAULT_KEY)
        if not encrypted_data:
            return order_id, False, "数据加密失败"
        
        headers = {
            'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
            'Content-Type': 'application/json'
        }
        payload = {"data": encrypted_data}
        response = requests.post(url_submit, headers=headers, data=json.dumps(payload), timeout=30)
        return order_id, response.status_code == 200, "成功" if response.status_code == 200 else f"状态码: {response.status_code}"
    except Exception as e:
        return order_id, False, str(e)

# ====================== 主函数（可调用） ======================
def process_orders_and_generate_log(fetch_date: int):
    """
    执行完整流程：获取订单 + 处理聊天记录 + AI分析，并生成日志文件。
    日志文件名：YYYYMMDD.log
    """
    BASE_PATH = get_base_path()
    specified_date = datetime(2025, 12, fetch_date, 0, 0, 0)
    start_time_string = specified_date.strftime("%Y-%m-%d %H:%M:%S")
    end_time_string = (specified_date + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE = os.path.join(BASE_PATH, f'{specified_date.strftime("%Y%m%d")}.log')
    
    # 重新配置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    PUBLIC_KEY_FILENAME = 'public_key.pem'
    PRIVATE_KEY_FILENAME = 'private_key.pem'
    public_key_path = os.path.join(BASE_PATH, PUBLIC_KEY_FILENAME)
    private_key_path = os.path.join(BASE_PATH, PRIVATE_KEY_FILENAME)

    config = load_and_validate_config(BASE_PATH)
    check_key_files(public_key_path, private_key_path)

    url_get_order_list = config.get('API', 'url_get_order_list')
    appsecret = config.get('API', 'appsecret')
    agent = config.get('API', 'agent')
    appid = config.get('API', 'appid')

    try:
        order_ids = fetch_and_decrypt_order_list(
            start_time_string, end_time_string, 'all',
            public_key_path, private_key_path,
            url_get_order_list, appsecret, agent, appid
        )

        if not order_ids:
            logging.info("没有获取到订单，程序结束")
            return LOG_FILE

        max_workers = min(10, len(order_ids))
        success_orders = []
        failed_orders = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_order = {
                executor.submit(process_single_order, oid, config, BASE_PATH, public_key_path, private_key_path): oid
                for oid in order_ids
            }
            completed = 0
            for future in as_completed(future_to_order):
                order_id, success, msg = future.result()
                completed += 1
                if success:
                    success_orders.append(order_id)
                else:
                    failed_orders.append((order_id, msg))

        logging.info("=" * 60)
        logging.info("完整流程执行完成总结:")
        logging.info(f"总订单数: {len(order_ids)}")
        logging.info(f"成功: {len(success_orders)}, 失败: {len(failed_orders)}")
        logging.info(f"成功率: {len(success_orders)/len(order_ids)*100:.2f}%")
        logging.info("=" * 60)

        return LOG_FILE

    except Exception as e:
        logging.error(f"主流程执行失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return LOG_FILE