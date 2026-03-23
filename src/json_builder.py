import json
import logging
# 注意：请确保这些引用在您的项目中路径正确
from config.event_config import EVENT_SCORES, get_event_level, parse_triggered_events
from utils.common_utils import generate_conversation_id

logger = logging.getLogger(__name__)

class JSONBuilder:
    def __init__(self):
        self.built_count = 0
        self.violation_orders = 0
    
    def build_upload_json(self, order_data, analysis_results):
        """
        构建符合上传格式的JSON（全量不省略版）
        """
        logger.info(f"为订单 {order_data['order_id']} 构建上传JSON")
        
        # 收集有违规的消息
        violation_msgs = self._collect_violation_messages(analysis_results)
        
        # 生成conversation_id
        conversation_id = generate_conversation_id(
            order_data['user_id'], 
            order_data['costumer_id']
        )
        
        # 构建完整的上传数据
        upload_data = {
            "data": {
                "conversation_id": conversation_id,
                "user_id": order_data['user_id'],
                "costumer_id": order_data['costumer_id'], 
                "order_id": order_data['order_id'],
                "is_msgs": 0 if violation_msgs else 1,
                "msgs": violation_msgs if violation_msgs else ['AI审查未发现问题']
            }
        }
        
        # 更新统计
        self.built_count += 1
        if violation_msgs:
            self.violation_orders += 1
        
        logger.info(f"订单 {order_data['order_id']} 构建完成: 违规消息 {len(violation_msgs)} 条")
        return upload_data

    def _collect_violation_messages(self, analysis_results):
        """收集有违规的消息 - 支持分拆理由和合并字段"""
        violation_msgs = []
        
        for result in analysis_results:
            analysis = result['analysis_result']
            original_msg = result['original_message']
            
            # 只要命中了有效事件且 risk_score > 0，就上传到 CRM。
            # 不再强依赖 violation=True，避免 review/低分命中事件被误判成“AI审查未发现问题”。
            triggered_events = parse_triggered_events(analysis.get('triggered_event', ''))
            if triggered_events and analysis.get('risk_score', 0) > 0:
                
                # 2. 获取 AI 拆分后的理由字典（由 predict 方法产生）
                event_reasons_map = analysis.get('event_reasons', {})
                
                # 3. 创建事件对象列表，传入字典进行精准分配
                events = self._create_event_objects(
                    triggered_events, 
                    analysis['reason'], 
                    event_reasons_map
                )
                
                # 4. 构建单条消息的违规结果对象
                msg_data = {
                    'msg_id': original_msg['msg_id'],
                    'table_name': original_msg['table_name'],
                    'external_userid': original_msg['external_userid'],
                    'userid': original_msg['userid'],
                    'content': original_msg['content'],
                    'events': events
                }
                
                # 合并消息特有的字段处理
                if 'msg_ids' in original_msg:
                    msg_data['msg_ids'] = original_msg['msg_ids']
                
                if 'msgtimes' in original_msg:
                    msg_data['msgtimes'] = original_msg['msgtimes']
                
                violation_msgs.append(msg_data)
        
        return violation_msgs
        
    def _create_event_objects(self, triggered_events, default_reason, event_reasons_map=None):
        """
        核心方法：为每个事件分配独立的理由
        """
        events = []
        
        for event_name in triggered_events:
            event_name = event_name.strip()
            if not event_name or event_name == "无":
                continue

            # A. 获取得分和等级
            score = EVENT_SCORES.get(event_name, 5)  
            level = get_event_level(score)
            
            # B. 【关键逻辑】理由分配
            # 优先从拆分好的字典里找，找不到则用整句理由兜底
            specific_reason = default_reason
            if event_reasons_map and event_name in event_reasons_map:
                specific_reason = event_reasons_map[event_name]
            
            # C. 构建单个 Event JSON 节点
            events.append({
                'event': event_name,
                'event_score': score,
                'event_level': level,
                'reason': specific_reason,
                'indexs': [0, 100],  
                'eventType': '确定违规项'
            })
        
        return events
    
    def get_statistics(self):
        """获取构建统计"""
        return {
            'built_count': self.built_count,
            'violation_orders': self.violation_orders,
            'compliance_orders': self.built_count - self.violation_orders
        }
