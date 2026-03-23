# -*- coding: utf-8 -*-
import os
import dashscope
import requests
from typing import List, Dict, Optional
import logging
from http import HTTPStatus
import time
import concurrent.futures
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    极速版语音处理器 (全并行优化版)
    - Flash阶段：高并发处理短语音或快速识别
    - 长语音阶段：并行处理 Flash 失败的任务，采用用户提供的成功脚本逻辑
    """
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 20): 
        resolved_key = (api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        self.api_key = resolved_key or None
        self.enabled = bool(self.api_key)

        # 显式设置 dashscope.api_key：避免仅设置环境变量但 SDK 未读取到导致 “No api key provided”
        if self.api_key:
            dashscope.api_key = self.api_key
            os.environ["DASHSCOPE_API_KEY"] = self.api_key
        else:
            logger.warning("未提供 DASHSCOPE_API_KEY，语音转写将跳过（不会影响文本消息分析）")

        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 线程池配置
        self.flash_workers = max_workers
        self.long_workers = 12 # 长语音并发数
        
        self._lock = threading.Lock()
        
        # 统计计数
        self.flash_success = 0
        self.long_audio_success = 0
        self.failed_count = 0

    # ================= 关键新增：补全 main.py 需要调用的单条处理方法 =================
    def transcribe_audio(self, audio_url: str, enable_itn: bool = False) -> Optional[str]:
        """
        供 main.py 中的 OrderProcessor 调用，处理单条语音
        """
        if not self.enabled or not audio_url:
            return None

        # 1. 优先尝试极速 Flash 模型
        res = self._try_flash_asr(audio_url, enable_itn)
        
        # 2. 如果失败，尝试长语音模型
        if not res:
            res = self._try_long_audio(audio_url)
        
        if res:
            # 按照你要求的格式输出，这样分析器才能检测到违规原文
            return f"[语音转文本(voice)]: {res} 违规原文"
        
        return None

    def _try_flash_asr(self, audio_url: str, enable_itn: bool) -> Optional[str]:
        """【阶段1】Flash 并发逻辑"""
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen3-asr-flash",
                messages=[{"role": "user", "content": [{"audio": audio_url}]}],
                asr_options={"enable_itn": enable_itn}
            )
            if response.status_code == 200:
                res_content = response.output.choices[0].message.content
                text = res_content[0].get('text', '').strip() if isinstance(res_content, list) else str(res_content).strip()
                if text:
                    return text
            return None
        except Exception:
            return None

    def _try_long_audio(self, audio_url: str) -> Optional[str]:
        """【阶段2】长语音逻辑 - 移植自测试成功的脚本"""
        url_id = audio_url[-20:]
        try:
            if not self.api_key:
                return None
            # 1. 提交任务
            task_response = dashscope.audio.asr.Transcription.async_call(
                model='fun-asr',
                file_urls=[audio_url],
                language_hints=["zh"],
                channel_id=[1]
            )

            if task_response.status_code != HTTPStatus.OK:
                logger.error(f"[{url_id}] 提交失败: {task_response.message}")
                return None

            task_id = task_response.output.task_id
            
            # 2. 等待任务完成
            result_response = dashscope.audio.asr.Transcription.wait(task=task_id)

            if result_response.status_code != HTTPStatus.OK:
                logger.error(f"[{url_id}] 任务执行出错: {result_response.message}")
                return None

            # 3. 检查状态并获取 URL
            output = result_response.output
            if output.task_status == 'SUCCEEDED':
                results = output.get('results', [])
                if not results:
                    return None

                url = results[0].get('transcription_url')
                if not url:
                    return None

                # 4. 下载 JSON 并提取文本
                res = requests.get(url, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    return self._extract_text(data)
            return None
        except Exception as e:
            logger.error(f"[{url_id}] 长语音识别异常: {str(e)}")
            return None

    def _extract_text(self, data):
        """文本提取逻辑"""
        texts = []
        if 'transcripts' in data:
            for item in data['transcripts']:
                if 'text' in item:
                    texts.append(item['text'].strip())
        elif 'sentences' in data:
            for item in data['sentences']:
                if 'text' in item:
                    texts.append(item['text'].strip())
        elif 'text' in data and isinstance(data['text'], str):
            texts.append(data['text'].strip())

        return " ".join([t for t in texts if t])

    def batch_transcribe(self, audio_urls: List[str], enable_itn: bool = False) -> Dict[str, Optional[str]]:
        """批量处理接口 (保留了你的并发优化)"""
        if not self.enabled or not audio_urls:
            return {}

        self.flash_success = 0
        self.long_audio_success = 0
        self.failed_count = 0
        
        unique_urls = list(set(audio_urls))
        final_results = {url: None for url in unique_urls}

        # --- 第一阶段: Flash 并发 ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.flash_workers) as executor:
            future_to_url = {executor.submit(self._try_flash_asr, url, enable_itn): url for url in unique_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    res = future.result()
                    if res:
                        final_results[url] = res
                        with self._lock: self.flash_success += 1
                except Exception: pass

        # --- 第二阶段: 长语音并行 ---
        failed_urls = [url for url, res in final_results.items() if res is None]
        if failed_urls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.long_workers) as long_executor:
                future_to_long_url = {long_executor.submit(self._try_long_audio, url): url for url in failed_urls}
                for future in concurrent.futures.as_completed(future_to_long_url):
                    url = future_to_long_url[future]
                    try:
                        res = future.result()
                        if res:
                            final_results[url] = res
                            with self._lock: self.long_audio_success += 1
                        else:
                            with self._lock: self.failed_count += 1
                    except Exception:
                        with self._lock: self.failed_count += 1

        # 返回格式化的结果
        return {
            # url: f"[语音转文本(voice)]: {final_results.get(url)} 违规原文" if final_results.get(url) else None 
            url: final_results.get(url) if final_results.get(url) else None
            for url in audio_urls
        }