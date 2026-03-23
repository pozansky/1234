# src/compliance_analyzer.py
from src.rag_engine import ComplianceRAGEngine
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import logging
import time
from utils.common_utils import get_timestamp

logger = logging.getLogger(__name__)

class ComplianceAnalyzer:
    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.engine = ComplianceRAGEngine()
        self.analyzed_count = 0
        self.violation_count = 0
    
    def analyze_order_messages(self, order_data):
        """
        分析单个订单的所有消息
        返回格式: [analysis_result1, analysis_result2, ...]
        """
        logger.info(f"开始分析订单 {order_data['order_id']} 的消息")
        
        analysis_results = []
        user_messages = []
        
        # 提取需要分析的用户消息
        for msg in order_data.get('msgs', []):
            # 检查是否为用户消息且有内容
            if (msg.get('entity_type') == 'user' and 
                msg.get('content')):
                
                # 原始逻辑：只处理文本消息
                # if msg.get('msg_type') == 'text':
                #     user_messages.append(msg)
                
                # 新逻辑：处理多种消息类型
                msg_type = msg.get('msg_type', '')
                content = msg.get('content', '')
                
                # 1. 文本消息：直接分析
                if msg_type == 'text':
                    user_messages.append(msg)
                
                # 2. 语音消息（已转文本）：标记为已处理的语音
                elif msg_type == 'voice' and content.startswith('[语音转文本]:'):
                    # 复制消息，但修改 msg_type 以便后续处理
                    processed_msg = msg.copy()
                    processed_msg['original_msg_type'] = 'voice'  # 保留原始类型
                    user_messages.append(processed_msg)

                
                # 2. 语音消息（已转文本）：标记为已处理的语音
                elif msg_type == 'meeting_voice_call':
                    processed_msg = msg.copy()
                    processed_msg['original_msg_type'] = 'meeting_voice_call'  # 保留原始类型
                    user_messages.append(processed_msg)
                    
                # 3. 其他类型的语音消息（可选处理）
                elif msg_type == 'voice':
                    # 可以创建一个占位消息进行分析
                    processed_msg = msg.copy()
                    processed_msg['content'] = '[语音消息，内容无法识别]'
                    processed_msg['original_msg_type'] = 'voice'
                    user_messages.append(processed_msg)
                
                # 4. 图片消息：可选处理
                elif msg_type == 'image':
                    processed_msg = msg.copy()
                    processed_msg['content'] = '[图片消息]'
                    processed_msg['original_msg_type'] = 'image'
                    user_messages.append(processed_msg)
        
        logger.info(f"订单 {order_data['order_id']} 有 {len(user_messages)} 条用户消息需要分析")
        
        # 如果消息太多，先输出一些调试信息
        if len(user_messages) > 50:
            logger.warning(f"警告: 订单 {order_data['order_id']} 消息较多 ({len(user_messages)}条)，可能需要较长时间")
        
        # 并行分析所有用户消息（传入订单的 product_type，用于按产品类型过滤特定事件）
        product_type = order_data.get('product_type')
        if user_messages:
            analysis_results = self._analyze_messages_parallel(user_messages, product_type=product_type)
        
        # 更新统计
        self.analyzed_count += len(user_messages)
        self.violation_count += sum(1 for result in analysis_results 
                                if result.get('analysis_result', {}).get('violation'))
        
        return analysis_results
    
    def _analyze_messages_parallel(self, messages, product_type=None):
        """并行分析多条消息。product_type 来自订单，用于「虚假宣传案例精选及人工推票」仅1.0、「对投研调研活动夸大宣传」仅3.0 的过滤。"""
        results = []
        
        def process_single_message(msg):
            """处理单条消息"""
            try:
                # 简单清理消息内容
                content = msg['content'].strip()
                if not content or len(content) < 2:
                    return {
                        'original_message': msg,
                        'analysis_result': {
                            'violation': False,
                            'triggered_event': '无',
                            'reason': '消息内容过短或为空'
                        },
                        'timestamp': get_timestamp()
                    }
                
                # 调用RAG引擎分析（传入 product_type 以按产品类型过滤特定事件），并记录耗时
                # start_time = time.perf_counter()
                analysis_result = self.engine.predict(content, product_type=product_type)
                # duration = time.perf_counter() - start_time
                # logger.info(f"消息 {msg.get('msg_id')} predict 耗时: {duration:.2f} 秒")
                return {
                    'original_message': msg,
                    'analysis_result': analysis_result,
                    'timestamp': get_timestamp()
                }
            except Exception as e:
                logger.error(f"分析消息失败 {msg.get('msg_id')}: {e}")
                return {
                    'original_message': msg,
                    'analysis_result': {
                        'violation': False,
                        'triggered_event': '分析失败',
                        'reason': str(e)
                    },
                    'timestamp': get_timestamp()
                }
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            futures = {}
            for msg in messages:
                future = executor.submit(process_single_message, msg)
                futures[future] = msg
            
            # 显示进度
            try:
                for future in tqdm(as_completed(futures), 
                                 total=len(messages), 
                                 desc=f"分析订单消息"):
                    try:
                        # 设置超时时间为30秒
                        result = future.result(timeout=30)
                        results.append(result)
                    except TimeoutError:
                        msg = futures[future]
                        logger.error(f"消息分析超时 {msg.get('msg_id')}")
                        results.append({
                            'original_message': msg,
                            'analysis_result': {
                                'violation': False,
                                'triggered_event': '分析超时',
                                'reason': '分析超过30秒超时'
                            },
                            'timestamp': get_timestamp()
                        })
                    except Exception as e:
                        msg = futures[future]
                        logger.error(f"消息分析异常 {msg.get('msg_id')}: {e}")
                        results.append({
                            'original_message': msg,
                            'analysis_result': {
                                'violation': False,
                                'triggered_event': '分析异常',
                                'reason': str(e)
                            },
                            'timestamp': get_timestamp()
                        })
            except KeyboardInterrupt:
                logger.warning("用户中断了分析过程")
                raise
            except Exception as e:
                logger.error(f"分析过程中出现未知错误: {e}")
        
        return results
    
    def get_statistics(self):
        """获取分析统计"""
        return {
            'analyzed_count': self.analyzed_count,
            'violation_count': self.violation_count,
            'compliance_rate': (self.analyzed_count - self.violation_count) / self.analyzed_count * 100 
                              if self.analyzed_count > 0 else 0
        }