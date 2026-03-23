#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并版完整代码：获取订单列表 + 处理聊天记录 + AI分析（并行版本）
修复版：保留数据收集与日志完整记录，升级 JSON 递归解析逻辑处理 Extra data 异常
速度优化版：增加并行度、连接复用和缓存优化
"""

import sys
import os
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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
from pathlib import Path
import concurrent.futures
from functools import lru_cache
import queue
import asyncio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# 与 DataExtractor 一致：只记录并保存 create_time/createtime > 此日期的订单，减少 .log 体积与写入时间
CREATE_TIME_CUTOFF = datetime(2026, 1, 1)


class OrderFetcher:
    """合规分析器类"""
    
    def __init__(self, base_path=None):
        """初始化合规分析器"""
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent
        
        if base_path:
            self.BASE_PATH = Path(base_path)
        elif os.getenv('SERVICE_BASE_PATH'):
            self.BASE_PATH = Path(os.getenv('SERVICE_BASE_PATH'))
        else:
            self.BASE_PATH = project_root
        
        self.CONFIG_FILE = os.path.join(self.BASE_PATH, 'config.ini')
        # 日志写入 logs 目录，避免在项目根目录生成 20260226.log 等文件
        logs_dir = os.path.join(self.BASE_PATH, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.LOG_FILE = os.path.join(logs_dir, f'{datetime.now().strftime("%Y%m%d")}_order_fetcher.log')

        self.DATA_OUTPUT_DIR = os.path.join(self.BASE_PATH, 'order_data')
        os.makedirs(self.DATA_OUTPUT_DIR, exist_ok=True)
        
        self.PUBLIC_KEY_FILENAME = 'public_key.pem'
        self.PRIVATE_KEY_FILENAME = 'private_key.pem'
        self.public_key_path = os.path.join(self.BASE_PATH, self.PUBLIC_KEY_FILENAME)
        self.private_key_path = os.path.join(self.BASE_PATH, self.PRIVATE_KEY_FILENAME)
        
        self.log_lock = threading.Lock()
        self.config = None
        self.logger = None
        
        # 创建全局的requests会话用于连接复用
        self.session = self._create_requests_session()
        self.config_session = self._create_requests_session()
        
        self._setup_logging()
        
        # 语音处理器
        self.voice_processor = None
        try:
            from src.voice_processor import VoiceProcessor
            # 固定使用你提供的 DashScope key
            os.environ["DASHSCOPE_API_KEY"] = "sk-eb015732b43844a7980f0daf9eba556d"
            self.voice_processor = VoiceProcessor(os.getenv("DASHSCOPE_API_KEY"))
            if self.voice_processor.enabled:
                self.logger.info("语音处理器初始化成功")
            else:
                self.logger.warning("未检测到 DASHSCOPE_API_KEY，语音转写将跳过")
        except Exception as e:
            self.logger.warning(f"语音处理器初始化失败或跳过: {e}")
            
        # 成功处理的订单数据存储（保留第一个代码的功能）
        self.successful_orders_data = []
        # 缓存RSA密钥和AES密码器
        self._rsa_key_cache = {}
        self._aes_cipher_cache = {}

    def _create_requests_session(self):
        """创建可复用的requests会话，优化连接池"""
        session = requests.Session()
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,  # 增加连接池大小
            pool_maxsize=100,      # 增加最大连接数
            pool_block=False
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _setup_logging(self):
        """设置日志，优化日志写入性能"""
        self.logger = logging.getLogger(f"OrderFetcher_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # 使用QueueHandler实现异步日志写入（大幅提升性能）
        try:
            from logging.handlers import QueueHandler, QueueListener
            self.log_queue = queue.Queue(-1)
            queue_handler = QueueHandler(self.log_queue)
            self.logger.addHandler(queue_handler)
            
            # 文件处理器
            file_handler = logging.FileHandler(self.LOG_FILE, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # 启动队列监听器
            self.queue_listener = QueueListener(self.log_queue, file_handler, console_handler)
            self.queue_listener.start()
        except ImportError:
            # 回退到传统方式
            file_handler = logging.FileHandler(self.LOG_FILE, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ====================== 加解密函数 ======================
    @lru_cache(maxsize=128)
    def _get_rsa_key(self, key_path, is_public=True):
        """缓存RSA密钥读取"""
        cache_key = f"{key_path}_{is_public}"
        if cache_key not in self._rsa_key_cache:
            with open(key_path, 'r') as f:
                key = RSA.import_key(f.read())
            self._rsa_key_cache[cache_key] = key
        return self._rsa_key_cache[cache_key]

    @lru_cache(maxsize=128)
    def _get_aes_cipher(self, key):
        """缓存AES密码器"""
        if key not in self._aes_cipher_cache:
            self._aes_cipher_cache[key] = AES.new(
                key.ljust(32, '\0').encode('utf-8'), 
                AES.MODE_ECB
            )
        return self._aes_cipher_cache[key]

    def rsa_encrypt(self, data, public_key_path):
        key = self._get_rsa_key(public_key_path, is_public=True)
        cipher = Cipher_pkcs1_v1_5.new(key)
        default_length = 245
        data_bytes = data.encode('utf-8')
        
        # 使用列表推导式提高性能
        chunks = [data_bytes[i:i+default_length] for i in range(0, len(data_bytes), default_length)]
        encrypted = b''.join(cipher.encrypt(chunk) for chunk in chunks)
        return base64.b64encode(encrypted).decode('utf-8')

    def aes_decrypt(self, data, key):
        cipher = self._get_aes_cipher(key)
        decrypted = cipher.decrypt(base64.b64decode(data))
        try:
            return decrypted.decode('utf-8').strip()
        except Exception:
            return decrypted.strip()

    def aes_en(self, plain_text: str, key: str) -> str:
        try:
            cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
            encrypted_bytes = cipher.encrypt(pad(plain_text.encode('utf-8'), AES.block_size))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            self.logger.error(f"加密失败: {e}")
            return None

    # ====================== 订单获取与解析 ======================
    @lru_cache(maxsize=256)
    def validate_and_parse_json(self, decrypted_result):
        """递归解析可能有'Extra data'异常的JSON串（第二个代码的核心解析逻辑）"""
        try:
            return [json.loads(decrypted_result)]
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                pos = e.pos
                valid_json = decrypted_result[:pos]
                extra_data = decrypted_result[pos:].lstrip(', ')
                result = [json.loads(valid_json)]
                if extra_data:
                    result.extend(self.validate_and_parse_json(extra_data))
                return result
            return []

    def fetch_and_decrypt_order_list(self, start_time_string, end_time_string, order_range, public_key_path, private_key_path, url, appsecret, agent, appid):
        timestamp = str(int(time.time()))
        nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[4:16]
        data = {'start': str(int(datetime.strptime(start_time_string, "%Y-%m-%d %H:%M:%S").timestamp())),
                'end': str(int(datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S").timestamp())),
                'range': order_range}
        sign_str = '&'.join([f"{k}={v}" for k, v in sorted(data.items())]) + f"&appsecret={appsecret}"
        sign = hashlib.md5(sign_str.encode()).hexdigest().upper()
        headers = {'Agent': agent, 'Content-Type': 'application/json;charset=utf-8', 'Appid': appid, 'Timestamp': timestamp, 'Nonce': nonce, 'Sign': sign}
        encrypted_data = self.rsa_encrypt(json.dumps(data), public_key_path)
        # 使用会话而不是requests.post
        response = self.config_session.post(url, headers=headers, data=encrypted_data, verify=False, timeout=30)
        result = response.json()
        if result.get('errcode') == 0:
            data = result.get('data')
            if isinstance(data, list):
                if not data:
                    self.logger.info("订单列表接口返回空列表（该时段无订单）")
                    return []
                data_block = data[0]
            else:
                data_block = data
            if not data_block:
                return []
            key = self._get_rsa_key(private_key_path, is_public=False)
            cipher = Cipher_pkcs1_v1_5.new(key)
            aes_key = cipher.decrypt(base64.b64decode(data_block['encryptKey']), None).decode('utf-8')
            decrypted_result = self.aes_decrypt(data_block['encryptData'], aes_key)
            parsed = self.validate_and_parse_json(decrypted_result)
            if not parsed:
                return []
            return parsed[0]
        return []

    # ====================== 聊天记录处理功能 ======================
    def extract_all_user_content(self, data, voice_processor=None):
        """提取用户所有内容（脚本1功能）"""
        extracted_contents = []
        voice_urls = []
        
        # 优化：使用局部函数减少属性查找
        def collect_voice_urls(obj):
            if isinstance(obj, dict):
                if obj.get('entity_type') == 'user' and obj.get('msg_type') == 'voice' and 'content' in obj:
                    voice_urls.append(obj['content'])
                for value in obj.values(): 
                    collect_voice_urls(value)
            elif isinstance(obj, list):
                for item in obj: 
                    collect_voice_urls(item)
        
        collect_voice_urls(data)
        
        # 批量处理语音转文本（如果可用）
        voice_transcriptions = {}
        if voice_processor and voice_urls:
            voice_transcriptions = voice_processor.batch_transcribe(voice_urls)

        # 优化处理逻辑
        def process_node(node):
            if isinstance(node, list):
                for item in node: 
                    process_node(item)
            elif isinstance(node, dict):
                if node.get('entity_type') == 'user':
                    msg_type = node.get('msg_type')
                    content = node.get('content')
                    if msg_type == 'text' and content: 
                        extracted_contents.append(content)
                    elif msg_type == 'voice' and content in voice_transcriptions:
                        extracted_contents.append(f"[语音转文本]: {voice_transcriptions[content]}")
                # 只处理dict和list，跳过其他类型
                for value in node.values():
                    if isinstance(value, (list, dict)): 
                        process_node(value)
        
        process_node(data)
        return ' '.join(extracted_contents)

    def validate_and_parse_chat_data(self, decrypted_result, voice_processor=None):
        """验证并解析聊天记录（合入脚本2的递归解析）"""
        try:
            # 关键：使用脚本2的递归解析处理 Extra data
            parsed_list = self.validate_and_parse_json(decrypted_result)
            if not parsed_list: 
                return []
            
            # 合并多块数据
            if len(parsed_list) > 1:
                data = []
                for block in parsed_list:
                    if isinstance(block, list): 
                        data.extend(block)
                    else: 
                        data.append(block)
            else:
                data = parsed_list[0]

            user_content = self.extract_all_user_content(data, voice_processor)
            
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    data[0]['user_concatenated_content'] = user_content
                return data
            elif isinstance(data, dict):
                data['user_concatenated_content'] = user_content
                return [data]
            return []
        except Exception as e:
            self.logger.error(f"解析失败: {e}")
            return []

    def _parse_create_time(self, value):
        """解析 create_time/createtime 为 datetime，与 DataExtractor 逻辑一致。"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        s = (value if isinstance(value, str) else str(value)).strip()
        if not s:
            return None
        for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M:%S.%f", 26), ("%Y-%m-%d", 10)):
            try:
                return datetime.strptime(s[:size], fmt)
            except ValueError:
                continue
        return None

    def _get_order_create_time_value(self, order_data):
        """只使用 create_time/createtime（不使用 hg_time）。"""
        for k in ("create_time", "createtime", "createTime"):
            if k in order_data:
                return order_data.get(k)
        return None

    def _is_order_on_or_after_cutoff(self, order_data):
        """仅当订单 create_time/createtime > 2026-01-01 时返回 True；缺失或解析失败则不写入 .log / _data.json。"""
        ct = self._get_order_create_time_value(order_data)
        dt = self._parse_create_time(ct)
        if dt is None:
            return False
        return dt > CREATE_TIME_CUTOFF

    def fetch_and_decrypt_chat_log(self, order_id, public_key_path, private_key_path, url, appsecret, agent, appid):
        """获取并解密聊天记录，并确保记录到日志（保留脚本1的日志输出逻辑）"""
        timestamp = str(int(time.time()))
        nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[4:16]
        data = {'order_id': order_id}
        sign_str = '&'.join([f"{k}={v}" for k, v in sorted(data.items())]) + f"&appsecret={appsecret}"
        sign = hashlib.md5(sign_str.encode()).hexdigest().upper()
        headers = {'Agent': agent, 'Content-Type': 'application/json;charset=utf-8', 'Appid': appid, 'Timestamp': timestamp, 'Nonce': nonce, 'Sign': sign}
        
        encrypted_data = self.rsa_encrypt(json.dumps(data), public_key_path)
        # 使用会话连接复用
        response = self.session.post(url, headers=headers, data=encrypted_data, verify=False, timeout=30)
        result = response.json()
        
        if result.get('errcode') == 0:
            db = result['data']
            key = self._get_rsa_key(private_key_path, is_public=False)
            cipher = Cipher_pkcs1_v1_5.new(key)
            aes_key = cipher.decrypt(base64.b64decode(db['encryptKey']), None).decode('utf-8')
            decrypted_result = self.aes_decrypt(db['encryptData'], aes_key)
            
            # --- 只对 create_time/createtime > 2026-01-01 的订单写 .log 并加入 _data.json，减少 I/O 与 .log 体积 ---
            try:
                parsed_blocks = self.validate_and_parse_json(decrypted_result)
                full_data = parsed_blocks[0] if parsed_blocks else {}
                record = full_data[0] if isinstance(full_data, list) else full_data
                if isinstance(record, dict):
                    record['order_id'] = order_id
                    if self._is_order_on_or_after_cutoff(record):
                        self.logger.info(f"订单 {order_id} 解析后的完整数据: {json.dumps(full_data, ensure_ascii=False, indent=2)}")
                        self.successful_orders_data.append(record)
            except Exception as e:
                self.logger.error(f"订单 {order_id} 记录到日志失败: {e}")
                
            return decrypted_result
        return f"接口错误: {result.get('errmsg')}"

    # ====================== 保存数据功能 ======================
    def save_order_data_to_file(self, output_file_path=None):
        if not self.successful_orders_data: 
            return None
        if output_file_path is None:
            output_file_path = os.path.join(os.path.dirname(self.LOG_FILE), f"{os.path.basename(self.LOG_FILE).split('.')[0]}_data.json")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.successful_orders_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ 成功保存数据到: {output_file_path}")
            return output_file_path
        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            return None

    # ====================== AI分析与执行 ======================
    def process_single_order(self, order_id, config, progress_info="", voice_processor=None):
        try:
            decrypted_result = self.fetch_and_decrypt_chat_log(
                order_id, self.public_key_path, self.private_key_path, 
                config.get('API', 'url_get_chat_log'), 
                config.get('API', 'appsecret'), 
                config.get('API', 'agent'), 
                config.get('API', 'appid')
            )
            
            parsed_data = self.validate_and_parse_chat_data(decrypted_result, voice_processor)
            if not parsed_data: 
                return order_id, False, "数据为空"

            # 替换会话ID - 优化计算
            for item in parsed_data:
                u, c = item.get('user_id'), item.get('costumer_id')
                if u and c:
                    # 优化：使用更快的排序和哈希
                    sorted_ids = sorted([str(u), str(c)])
                    item['conversation_id'] = hashlib.md5(f"{sorted_ids[0]}|{sorted_ids[1]}".encode()).hexdigest()
            
            target = parsed_data[0]
            target['preprocess'] = int(config.get('API', 'preprocess'))
            
            plain = json.dumps(target, ensure_ascii=False)
            enc = self.aes_en(plain, config.get('API', 'aes_DEFAULT_KEY'))
            
            resp = self.session.post(
                config.get('API', 'url_submit_encrypted'), 
                data=json.dumps({"data": enc}), 
                timeout=30
            )
            if resp.status_code == 200:
                return order_id, True, "成功"
            return order_id, False, f"状态码: {resp.status_code}"
        except Exception as e:
            return order_id, False, str(e)

    def execute_full_analysis(self, start_datetime=None, end_datetime=None, analysis_hours=9):
        self.logger.info("=" * 60)
        self.logger.info("开始执行流程（增强解析版）")
        self.successful_orders_data = [] # 重置
        
        if start_datetime is None: 
            start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if end_datetime is None: 
            end_datetime = start_datetime + timedelta(hours=analysis_hours)
        
        config = configparser.ConfigParser()
        config.read(self.CONFIG_FILE)
        
        order_ids = self.fetch_and_decrypt_order_list(
            start_datetime.strftime("%Y-%m-%d %H:%M:%S"), 
            end_datetime.strftime("%Y-%m-%d %H:%M:%S"), 
            'all',
            self.public_key_path, 
            self.private_key_path, 
            config.get('API', 'url_get_order_list'), 
            config.get('API', 'appsecret'), 
            config.get('API', 'agent'), 
            config.get('API', 'appid')
        )
        
        self.logger.info(f"获取到订单: {len(order_ids)} 个")
        if not order_ids: 
            return [], [], 0, None

        success_orders, failed_orders = [], []
        
        # 动态调整线程数：根据订单数量调整，但不超过CPU核心数的2倍
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(len(order_ids), cpu_count * 2, 50)  # 最大50个线程
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 批量提交任务
            futures = []
            batch_size = 10  # 小批量提交，避免内存峰值
            
            for i in range(0, len(order_ids), batch_size):
                batch = order_ids[i:i + batch_size]
                for oid in batch:
                    future = executor.submit(
                        self.process_single_order, 
                        oid, 
                        config, 
                        "", 
                        self.voice_processor
                    )
                    futures.append(future)
            
            # 批量获取结果
            for future in concurrent.futures.as_completed(futures):
                oid, success, msg = future.result()
                if success: 
                    success_orders.append(oid)
                else: 
                    failed_orders.append((oid, msg))

        # 保存为 JSON 文件  
        data_file_path = self.save_order_data_to_file()
        
        # 清理会话和缓存
        self.session.close()
        self.config_session.close()
        
        return success_orders, failed_orders, len(order_ids), data_file_path

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'queue_listener'):
            self.queue_listener.stop()
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'config_session'):
            self.config_session.close()


if __name__ == "__main__":
    analyzer = OrderFetcher()
    start_time = datetime(2025, 12, 30, 7, 0, 0)
    
    # 记录开始时间
    start_t = time.time()
    analyzer.execute_full_analysis(start_datetime=start_time, analysis_hours=4)
    
    # 记录总耗时
    end_t = time.time()
    print(f"总执行时间: {end_t - start_t:.2f} 秒")