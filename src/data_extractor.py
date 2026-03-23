import json
import re
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 流式读取时每块大小，避免整文件进内存导致 MemoryError（大小时 log 可能几百 MB）
_STREAM_CHUNK_SIZE = 512 * 1024  # 512KB
_MARKER = "解析后的完整数据:"

# 只提取 create_time/createtime 严格大于此日期的聊天记录，不满足则不提取且不参与语音转写
CREATE_TIME_CUTOFF = datetime(2026, 1, 1)


class DataExtractor:
    def __init__(self):
        self.processed_orders = 0
        self.total_messages = 0

    def _parse_create_time(self, value):
        """解析 create_time/createtime 为 datetime，支持 '2026-01-06 00:04:11' 或 '2026-01-01' 等格式。"""
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
        """仅当订单 create_time/createtime > 2026-01-01 时返回 True；缺失或解析失败视为不满足，不提取。"""
        ct = self._get_order_create_time_value(order_data)
        dt = self._parse_create_time(ct)
        if dt is None:
            return False
        return dt > CREATE_TIME_CUTOFF

    def _extract_orders_streaming(self, input_file):
        """
        流式读取：不把整文件读入内存，按块读并解析「解析后的完整数据:」后的 JSON。
        支持单个订单 JSON 跨多块，避免大小时 log 导致 MemoryError。
        """
        orders = []
        with open(input_file, 'r', encoding='utf-8') as f:
            buffer = ""
            pending_json = False  # 当前 buffer 是否从「未读完的一条 JSON」开头
            while True:
                if not pending_json:
                    chunk = f.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    buffer += chunk
                # 若正在拼一条跨块的 JSON，先尝试在当前 buffer 里找完整结尾
                if pending_json:
                    depth = 0
                    end = -1
                    for i, c in enumerate(buffer):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end != -1:
                        json_str = buffer[:end]
                        buffer = buffer[end:]
                        pending_json = False
                        try:
                            data = json.loads(json_str)
                            if self._is_valid_order_data(data) and self._is_order_on_or_after_cutoff(data):
                                filtered_data = self._filter_order_messages(data)
                                orders.append(filtered_data)
                                self.processed_orders += 1
                                self.total_messages += len(filtered_data.get('msgs', []))
                                logger.info(f"提取到订单: {data.get('order_id', '未知')}, 消息数: {len(filtered_data.get('msgs', []))}")
                            elif self._is_valid_order_data(data):
                                logger.debug(f"跳过订单( create_time/createtime 早于 2026-01-01 或缺失): {data.get('order_id', '未知')}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON解析失败: {e}")
                    else:
                        # 还没读完，再读一块
                        chunk = f.read(_STREAM_CHUNK_SIZE)
                        if not chunk:
                            break
                        buffer += chunk
                    continue
                # 查找「解析后的完整数据:」后的 JSON
                if _MARKER not in buffer:
                    # 避免 buffer 无限增长：只保留可能跨块的尾部
                    if len(buffer) > _STREAM_CHUNK_SIZE:
                        buffer = buffer[-_STREAM_CHUNK_SIZE:]
                    continue
                idx = buffer.index(_MARKER)
                buffer = buffer[idx + len(_MARKER):].lstrip()
                start = buffer.find('{')
                if start == -1:
                    buffer = _MARKER + buffer
                    continue
                buffer = buffer[start:]
                depth = 0
                end = -1
                for i, c in enumerate(buffer):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end == -1:
                    pending_json = True
                    continue
                json_str = buffer[:end]
                buffer = buffer[end:]
                try:
                    data = json.loads(json_str)
                    if self._is_valid_order_data(data) and self._is_order_on_or_after_cutoff(data):
                        filtered_data = self._filter_order_messages(data)
                        orders.append(filtered_data)
                        self.processed_orders += 1
                        self.total_messages += len(filtered_data.get('msgs', []))
                        logger.info(f"提取到订单: {data.get('order_id', '未知')}, 消息数: {len(filtered_data.get('msgs', []))}")
                    elif self._is_valid_order_data(data):
                        logger.debug(f"跳过订单( create_time/createtime 早于 2026-01-01 或缺失): {data.get('order_id', '未知')}")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败: {e}")
        return orders

    def extract_orders_from_data_json(self, input_file):
        """从 _data.json 直接加载订单（用于 --hour 重跑，确保与语音缓存 URL 一致）。"""
        logger.info(f"开始从 _data.json 加载订单: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            orders = []
            for item in (raw if isinstance(raw, list) else [raw]):
                if self._is_valid_order_data(item) and self._is_order_on_or_after_cutoff(item):
                    filtered = self._filter_order_messages(item)
                    orders.append(filtered)
                    self.processed_orders += 1
                    self.total_messages += len(filtered.get('msgs', []))
            logger.info(f"成功加载 {len(orders)} 个订单（仅 create_time/createtime>=2026-01-01），共 {self.total_messages} 条消息")
            return orders
        except Exception as e:
            logger.error(f"从 _data.json 加载失败: {e}")
            return []

    def extract_orders_from_log(self, input_file):
        """
        从日志文件中提取订单数据（支持大文件流式读取，避免 MemoryError）
        专门提取解密后的完整订单JSON数据
        """
        logger.info(f"开始从文件提取订单数据: {input_file}")
        try:
            # 优先用流式解析（方法1 的语义），不整文件读入
            orders = self._extract_orders_streaming(input_file)
            logger.info(f"找到 {len(orders)} 个'解析后的完整数据'")

            # 若流式没找到，再整文件试方法2、方法3（仅当文件不大时，避免大文件再次 MemoryError）
            if not orders:
                max_size = 100 * 1024 * 1024  # 100MB
                if os.path.getsize(input_file) > max_size:
                    logger.warning("文件过大，跳过方法2/3 整文件解析，仅使用流式结果")
                else:
                    try:
                        with open(input_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except MemoryError:
                        logger.error("文件过大导致内存不足，仅使用流式解析结果")
                    else:
                        # 方法2
                        logger.info("尝试查找完整订单结构的JSON")
                        order_pattern = r'\{\s*"user_id"[^}]*"costumer_id"[^}]*"order_id"[^}]*"msgs"[^}]*\}'
                        for match in re.findall(order_pattern, content, re.DOTALL):
                            try:
                                fixed_match = self._fix_json_format(match)
                                data = json.loads(fixed_match)
                                if self._is_valid_order_data(data) and self._is_order_on_or_after_cutoff(data):
                                    filtered_data = self._filter_order_messages(data)
                                    orders.append(filtered_data)
                                    self.processed_orders += 1
                                    self.total_messages += len(filtered_data.get('msgs', []))
                            except json.JSONDecodeError:
                                continue
                        if not orders:
                            logger.info("尝试查找包含msgs数组的JSON对象")
                            json_pattern = r'\{\s*[^}]*"msgs"\s*:\s*\[[^]]*\][^}]*\}'
                            for match in re.findall(json_pattern, content, re.DOTALL):
                                try:
                                    data = json.loads(match)
                                    if self._is_valid_order_data(data) and self._is_order_on_or_after_cutoff(data):
                                        filtered_data = self._filter_order_messages(data)
                                        orders.append(filtered_data)
                                        self.processed_orders += 1
                                        self.total_messages += len(filtered_data.get('msgs', []))
                                except json.JSONDecodeError:
                                    continue

            logger.info(f"成功提取 {len(orders)} 个订单，共 {self.total_messages} 条消息")
            return orders
        except Exception as e:
            logger.error(f"提取订单数据失败: {e}")
            return []
    
    def _filter_order_messages(self, order_data):
        """
        过滤订单消息：
        1. entity_type 只取 'user' (客服消息)
        2. msg_type 只取 'text', 'voice', 'meeting_voice_call'
        """
        if not isinstance(order_data, dict):
            return order_data
        
        if 'msgs' not in order_data:
            return order_data
        
        # 深拷贝订单数据
        filtered_order = order_data.copy()
        
        # 过滤消息
        filtered_msgs = []
        for msg in order_data.get('msgs', []):
            if not isinstance(msg, dict):
                continue
            
            # 检查 entity_type
            entity_type = msg.get('entity_type')
            if entity_type != 'user':  # 只保留客服消息
                continue
            
            # 检查 msg_type
            msg_type = msg.get('msg_type')
            if msg_type not in ['text', 'voice', 'meeting_voice_call']:  # 只保留指定类型
                continue
            
            # 保留这条消息
            filtered_msgs.append(msg)
        
        # 更新过滤后的消息列表
        filtered_order['msgs'] = filtered_msgs
        
        # 【可选】添加统计信息到订单中，便于调试
        filtered_order['filtered_stats'] = {
            'original_msg_count': len(order_data.get('msgs', [])),
            'filtered_msg_count': len(filtered_msgs),
            'filtered_types': ['text', 'voice', 'meeting_voice_call'],
            'filtered_entity': 'user'
        }
        
        return filtered_order
    
    def _is_valid_order_data(self, obj):
        """判断是否为有效的订单数据"""
        if not isinstance(obj, dict):
            return False
        
        # 必须包含的基本字段
        required_fields = ['user_id', 'costumer_id', 'order_id', 'msgs']
        if not all(field in obj for field in required_fields):
            return False
        
        # msgs 必须是列表
        if not isinstance(obj['msgs'], list):
            return False
        
        # 【修改】现在允许空消息列表，因为后续会过滤
        # if len(obj['msgs']) == 0:
        #     return False
        
        # 如果有消息，检查第一条消息
        if len(obj['msgs']) > 0:
            first_msg = obj['msgs'][0]
            if not isinstance(first_msg, dict):
                return False
            
            # 消息应该包含这些字段中的一些
            chat_fields = ['msg_id', 'content', 'entity_type', 'msg_type']
            if not any(field in first_msg for field in chat_fields):
                return False
        
        return True
    
    def _fix_json_format(self, json_str):
        """修复JSON格式问题"""
        # 移除可能的尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str
    
    def get_statistics(self):
        """获取处理统计"""
        return {
            'processed_orders': self.processed_orders,
            'total_messages': self.total_messages
        }