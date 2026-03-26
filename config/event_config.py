# 事件分数和级别配置
EVENT_SCORES = {
    "直接承诺收益": 10,
    "向客户索要手机号": 10,
    "异常开户": 10,
    "干扰风险测评独立性": 10,
    "不文明用语": 10,
    "以退费为营销卖点": 10,
    "怂恿客户使用他人身份证办理服务": 10,
    "违规指导": 10,
    "将具体股票策略接入权限作为即时办理卖点": 10,
    "突出客户盈利反馈": 10,
    "突出描述个股涨幅绩效": 10,
    "收受客户礼品": 10,
    "夸大宣传策略重仓操作": 5,
    "对投研调研活动夸大宣传": 5,
    "冒用沈杨老师名义": 5,
    "使用敏感词汇": 5,
    "错误表述服务合同生效起始周期": 5,
    "虚假宣传案例精选及人工推票": 5,
    "变相承诺投资收益": 10,
    "虚假宣传": 10,
    "怂恿或知晓客户借贷投资": 10,
}


def get_event_level(score):
    """根据分数获取事件级别"""
    return "红线" if score == 10 else "黄线"


def parse_triggered_events(event_string):
    """解析触发事件字符串"""
    if event_string == "无" or not event_string or event_string.strip() == "":
        return []

    # 按逗号分割，去除空格
    events = [event.strip() for event in event_string.split(",") if event.strip()]
    return events
