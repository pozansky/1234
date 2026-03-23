import json
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import hashlib
import time
import threading
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
import urllib3
from pathlib import Path

# 忽略HTTPS警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class UploadHandler:
    # CRM 请求超时（秒），避免单次请求挂死导致整批变慢
    CRM_REQUEST_TIMEOUT = 60
    CRM_UPLOAD_RETRIES = 3
    CRM_RETRY_SLEEP = 2

    # 后台上传线程池大小，避免无限开线程导致内存累积
    UPLOAD_POOL_WORKERS = 10

    def __init__(self, output_dir="upload_results", config_file="config.ini"):
        self.output_dir = output_dir
        self.uploaded_count = 0
        self.failed_count = 0
        self._lock = threading.Lock()
        self._upload_executor = ThreadPoolExecutor(max_workers=self.UPLOAD_POOL_WORKERS, thread_name_prefix="crm_upload")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载配置
        self.config = self._load_config(config_file)
        if self.config:
            self._init_crm_config()
    
    def _load_config(self, config_file):
        """加载配置文件"""
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file)
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return None
    
    def _init_crm_config(self):
        """初始化CRM配置"""
        try:
            self.url_crm = self.config.get('API', 'url_crm')
            self.appsecret = self.config.get('API', 'appsecret')
            self.agent = self.config.get('API', 'agent')
            self.appid = self.config.get('API', 'appid')
            
            # 密钥路径
            current_file_path = Path(__file__).resolve()
            # 获取项目根目录（当前文件的祖父目录：src的父目录）
            BASE_PATH = current_file_path.parent.parent

            self.public_key_path = os.path.join(BASE_PATH, 'public_key.pem')
            self.private_key_path = os.path.join(BASE_PATH, 'private_key.pem')

            # 可选性能配置（从 config.ini 的 [PERF] 读取，未配置则使用默认类属性）
            if self.config.has_section('PERF'):
                try:
                    self.CRM_REQUEST_TIMEOUT = self.config.getint('PERF', 'crm_timeout', fallback=self.CRM_REQUEST_TIMEOUT)
                    self.CRM_UPLOAD_RETRIES = self.config.getint('PERF', 'crm_retries', fallback=self.CRM_UPLOAD_RETRIES)
                    self.UPLOAD_POOL_WORKERS = self.config.getint('PERF', 'upload_workers', fallback=self.UPLOAD_POOL_WORKERS)
                    # 重新创建线程池以应用新的并发度
                    try:
                        self._upload_executor.shutdown(wait=False)
                    except Exception:
                        pass
                    self._upload_executor = ThreadPoolExecutor(
                        max_workers=self.UPLOAD_POOL_WORKERS,
                        thread_name_prefix="crm_upload"
                    )
                    logger.info(
                        f"PERF配置加载成功: upload_workers={self.UPLOAD_POOL_WORKERS}, "
                        f"crm_timeout={self.CRM_REQUEST_TIMEOUT}, crm_retries={self.CRM_UPLOAD_RETRIES}"
                    )
                except Exception as e:
                    logger.warning(f"加载 PERF 配置失败，使用默认值: {e}")
            
            logger.info("CRM配置加载成功")
        except Exception as e:
            logger.error(f"CRM配置初始化失败: {e}")
    
    def rsa_encrypt(self, data, public_key_path):
        """RSA加密"""
        try:
            with open(public_key_path, 'r') as f:
                key = RSA.import_key(f.read())
            cipher = Cipher_pkcs1_v1_5.new(key)
            default_length = 245
            data_bytes = data.encode('utf-8')
            encrypted = b''.join([cipher.encrypt(data_bytes[i:i+default_length]) 
                                for i in range(0, len(data_bytes), default_length)])
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"RSA加密失败: {e}")
            return None
    
    def upload_to_crm(self, data, timeout=None, max_retries=None):
        """上传数据到CRM系统。使用较短超时+重试，避免单次请求挂死导致整批变慢。"""
        if not hasattr(self, 'url_crm'):
            logger.error("CRM配置未初始化，无法上传")
            return False, "CRM配置未初始化"

        timeout = timeout if timeout is not None else self.CRM_REQUEST_TIMEOUT
        max_retries = max_retries if max_retries is not None else self.CRM_UPLOAD_RETRIES
        last_error = None

        total_start = time.time()

        for attempt in range(max_retries):
            try:
                # 准备上传数据
                data_test = data['data']

                # 如果msgs是列表，需要转换为JSON字符串
                if 'msgs' in data_test and isinstance(data_test['msgs'], list):
                    data_test['msgs'] = json.dumps(data_test['msgs'], ensure_ascii=False)

                timestamp = str(int(time.time()))
                nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[4:16]

                sign_str = '&'.join([f"{k}={v}" for k, v in sorted(data_test.items())]) + f"&appsecret={self.appsecret}"
                sign = hashlib.md5(sign_str.encode()).hexdigest().upper()

                headers = {
                    'Agent': self.agent,
                    'Content-Type': 'application/json;charset=utf-8',
                    'Appid': self.appid,
                    'Timestamp': timestamp,
                    'Nonce': nonce,
                    'Sign': sign
                }

                enc_start = time.time()
                encrypted_data = self.rsa_encrypt(json.dumps(data_test), self.public_key_path)
                enc_cost = time.time() - enc_start
                if not encrypted_data:
                    return False, "数据加密失败"

                if attempt > 0:
                    logger.info(f"上传数据到CRM (第{attempt + 1}次重试): {self.url_crm}，加密耗时 {enc_cost:.3f}s")
                else:
                    logger.info(f"上传数据到CRM: {self.url_crm}，加密耗时 {enc_cost:.3f}s")

                req_start = time.time()
                response = requests.post(
                    self.url_crm, headers=headers, data=encrypted_data,
                    verify=False, timeout=timeout
                )
                req_cost = time.time() - req_start

                resp_start = time.time()
                result = response.json()
                resp_cost = time.time() - resp_start

                if result.get('errcode') == 0:
                    total_cost = time.time() - total_start
                    logger.info(
                        f"CRM上传成功: {result.get('msg', 'success')} "
                        f"(enc={enc_cost:.3f}s, net={req_cost:.3f}s, parse={resp_cost:.3f}s, total={total_cost:.3f}s)"
                    )
                    return True, "上传成功"
                else:
                    last_error = (
                        f"CRM上传失败: errcode={result.get('errcode')}, errmsg={result.get('errmsg', 'Unknown error')}, "
                        f"enc={enc_cost:.3f}s, net={req_cost:.3f}s, parse={resp_cost:.3f}s"
                    )
                    logger.warning(last_error)
                    if attempt < max_retries - 1:
                        time.sleep(self.CRM_RETRY_SLEEP)
                    continue

            except requests.exceptions.Timeout as e:
                last_error = f"CRM上传超时({timeout}s): {e}"
                logger.warning(last_error)
                if attempt < max_retries - 1:
                    time.sleep(self.CRM_RETRY_SLEEP)
            except Exception as e:
                last_error = f"CRM上传异常: {str(e)}"
                logger.warning(last_error)
                if attempt < max_retries - 1:
                    time.sleep(self.CRM_RETRY_SLEEP)

        total_cost = time.time() - total_start
        logger.error(f"{last_error} (total={total_cost:.3f}s)")
        return False, last_error or "上传失败"
    
    def _do_upload_async(self, filepath, order_id):
        """后台上传到 CRM：从文件读数据再上传，避免主流程长期持有 upload_data 导致内存累积。"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                upload_data = json.load(f)
        except Exception as e:
            with self._lock:
                self.failed_count += 1
            logger.warning(f"CRM后台上传读取文件失败: {filepath}, {e}")
            return
        start = time.time()
        crm_success, crm_message = self.upload_to_crm(upload_data)
        elapsed = time.time() - start
        with self._lock:
            if crm_success:
                self.uploaded_count += 1
                logger.info(f"CRM后台上传成功: {filepath}，总耗时 {elapsed:.3f}s")
            else:
                self.failed_count += 1
                logger.warning(f"CRM后台上传失败: {filepath}, 原因: {crm_message}，总耗时 {elapsed:.3f}s")

    def save_upload_data(self, upload_data, order_id=None, upload_async=False):
        """
        保存上传数据到文件并上传到CRM。
        upload_async=True 时：先落盘，CRM 上传在后台线程执行，主流程不阻塞（适合批量时避免“上传到CRM特别慢”）。
        """
        try:
            if order_id is None:
                order_id = upload_data['data']['order_id']

            # --- 提取订单日期并结合当前处理时间 ---
            str_order_id = str(order_id)
            if len(str_order_id) >= 8 and str_order_id[:8].isdigit():
                order_date = str_order_id[:8]
            else:
                order_date = datetime.now().strftime("%Y%m%d")
            process_time = datetime.now().strftime("%H%M%S")
            filename = f"upload_{order_date}_{order_id}_{process_time}.json"
            filepath = os.path.join(self.output_dir, filename)

            # 先落盘（始终同步写文件）
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(upload_data, f, ensure_ascii=False, indent=2)

            if upload_async:
                # 后台上传：只传 filepath/order_id，线程内从文件读数据，避免主流程和线程池队列长期持有大对象
                self._upload_executor.submit(self._do_upload_async, filepath, order_id)
                logger.info(f"数据已保存，CRM上传已在后台进行: {filepath}")
                return {
                    'success': True,
                    'filepath': filepath,
                    'crm_upload': 'async',
                    'crm_message': '后台上传中',
                    'order_id': order_id
                }

            # 同步上传
            crm_success, crm_message = self.upload_to_crm(upload_data)
            if crm_success:
                with self._lock:
                    self.uploaded_count += 1
                logger.info(f"上传数据已保存并成功上传到CRM: {filepath}")
                return {
                    'success': True,
                    'filepath': filepath,
                    'crm_upload': True,
                    'crm_message': crm_message,
                    'order_id': order_id
                }
            else:
                with self._lock:
                    self.failed_count += 1
                logger.warning(f"数据保存成功但CRM上传失败: {filepath}, 原因: {crm_message}")
                return {
                    'success': False,
                    'filepath': filepath,
                    'crm_upload': False,
                    'crm_message': crm_message,
                    'order_id': order_id
                }

        except Exception as e:
            with self._lock:
                self.failed_count += 1
            logger.error(f"保存上传数据失败 {order_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    def batch_save_upload_data(self, upload_data_list):
        """批量保存上传数据"""
        results = []
        for upload_data in upload_data_list:
            result = self.save_upload_data(upload_data)
            results.append(result)
        return results
    
    def get_statistics(self):
        """获取上传统计"""
        total = self.uploaded_count + self.failed_count
        success_rate = self.uploaded_count / total * 100 if total > 0 else 0
        
        return {
            'uploaded_count': self.uploaded_count,
            'failed_count': self.failed_count,
            'success_rate': success_rate
        }