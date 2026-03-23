import os
import re
import json
from typing import Dict, Any, List, Tuple
import warnings

# 鍏抽棴鎵€鏈?LangChain 鐩稿叧璀﹀憡
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 鍏抽棴 LangChain 鎺у埗鍙拌拷韪?
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_VERBOSE", "false")
# 鍏抽棴 LangChain 鎺у埗鍙拌拷韪紝閬垮厤鎵撳嵃妫€绱㈢粨鏋滅瓑涓棿姝ラ

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 璁剧疆 DashScope API Key

class ComplianceRAGEngine:
    # 鎸変骇鍝佺被鍨嬬敓鏁堢殑浜嬩欢锛氫簨浠跺悕绉?-> 浠呭湪瀵瑰簲 product_type 鏃朵繚鐣欙紙"1.0"/"2.0"/"3.0"锛?
    PRODUCT_TYPE_GATED_EVENTS: Dict[str, str] = {
        "铏氬亣瀹ｄ紶妗堜緥绮鹃€夊強浜哄伐鎺ㄧエ": "1.0",
        "鍐掔敤娌堟潹鑰佸笀鍚嶄箟": "2.0",
        "瀵规姇鐮旇皟鐮旀椿鍔ㄥじ澶у浼?: "3.0",
        "澶稿ぇ瀹ｄ紶绛栫暐閲嶄粨鎿嶄綔": "3.0",
    }
    # 瑙勫垯 ID -> 瑙勫垯鍚嶇О锛堜笌 prompt 鐧藉悕鍗曘€乸roduct_type 杩囨护鍏辩敤锛?
    RULE_NAMES: Dict[int, str] = {
        1: "鐩存帴鎵胯鏀剁泭",
        2: "绐佸嚭瀹㈡埛鐩堝埄鍙嶉",
        3: "绐佸嚭鎻忚堪涓偂娑ㄥ箙缁╂晥",
        4: "瀵规姇鐮旇皟鐮旀椿鍔ㄥじ澶у浼?,
        5: "鍚戝鎴风储瑕佹墜鏈哄彿",
        6: "浣跨敤鏁忔劅璇嶆眹",
        7: "寮傚父寮€鎴?,
        8: "骞叉壈椋庨櫓娴嬭瘎鐙珛鎬?,
        9: "閿欒琛ㄨ堪鏈嶅姟鍚堝悓鐢熸晥璧峰鍛ㄦ湡",
        10: "涓嶆枃鏄庣敤璇?,
        11: "浠ラ€€娆句负钀ラ攢鍗栫偣",
        12: "鎬傛伩瀹㈡埛浣跨敤浠栦汉韬唤鍔炵悊鏈嶅姟",
        13: "杩濊鎸囧",
        14: "灏嗗叿浣撹偂绁ㄧ瓥鐣ユ帴鍏ユ潈闄愪綔涓哄嵆鏃跺姙鐞嗗崠鐐?,
        15: "铏氬亣瀹ｄ紶妗堜緥绮鹃€夊強浜哄伐鎺ㄧエ",
        16: "鍐掔敤娌堟潹鑰佸笀鍚嶄箟",
        17: "鏀跺彈瀹㈡埛绀煎搧",
        18: "澶稿ぇ瀹ｄ紶绛栫暐閲嶄粨鎿嶄綔",
    }

    def __init__(
        self,
        retrieve_k: int = None,
        retrieve_score_threshold: float = None,
        max_rules: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        # 妫€绱笌鍒嗗潡鍙傛暟锛堝彲鐢辫皟鐢ㄦ柟浼犲叆鎴栦粠鐜鍙橀噺璇诲彇锛屼究浜庤皟鍙傦級
        def _int_env(name: str, default: int) -> int:
            v = os.getenv(name)
            return int(v) if v is not None and v.strip() != "" else default

        def _float_env(name: str, default: float) -> float:
            v = os.getenv(name)
            return float(v) if v is not None and v.strip() != "" else default

        self._retrieve_k = retrieve_k if retrieve_k is not None else _int_env("RAG_RETRIEVE_K", 20)
        self._retrieve_score_threshold = (
            retrieve_score_threshold
            if retrieve_score_threshold is not None
            else _float_env("RAG_RETRIEVE_SCORE_THRESHOLD", 0.35)
        )
        self._max_rules = max_rules if max_rules is not None else _int_env("RAG_MAX_RULES", 6)
        self._chunk_size = chunk_size if chunk_size is not None else _int_env("RAG_CHUNK_SIZE", 600)
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else _int_env("RAG_CHUNK_OVERLAP", 200)

        # 1. 鍒濆鍖栧祵鍏ユā鍨?(浣跨敤鏈湴妯″瀷纭繚璇箟鍖归厤绮惧害)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. 鏋勫缓鍚戦噺搴擄紙浣跨敤鍒嗗潡绛栫暐鎻愰珮鍙洖鐜囷級
        self._initialize_vector_store()

        # 3. 鍒濆鍖?LLM (淇濇寔鍘熸湁鍙傛暟浠ョ‘淇濈‘瀹氭€?
        self.llm = ChatOpenAI(
            model="deepseek-v3.2",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=800,
            top_p=1.0,
            seed=42,
            max_retries=2,
            request_timeout=60,
        )

        # 4. 瀹氫箟 RAG 涓撶敤 Prompt锛堣瘎鍒嗙増锛氳緭鍑洪闄╁垎 + 缃俊搴?+ 浜嬩欢鍥犲瓙锛?
        _prompt_raw = """
浣犳槸涓€涓瘉鍒告姇椤惧満鏅殑銆愬悎瑙勯闄╄瘎鍒嗗紩鎿庛€戙€?

浣犵殑浠诲姟涓嶆槸绠€鍗曞仛鈥滃懡涓嵆杩濊鈥濈殑浜屽厓鍒ゆ柇锛岃€屾槸锛?
1锛夋牴鎹?18 鏉¤繚瑙勪簨浠惰鍒欙紝璇嗗埆鏂囨湰涓殑銆愰闄╁洜瀛愩€戜笌銆愪繚鎶ゅ洜瀛愩€戯紱
2锛夋寜鐓х粺涓€鐨勫姞鍑忓垎閫昏緫锛岀粰鍑?0鈥?00 鐨勯闄╂€诲垎 risk_score锛?
3锛夊啀鏍规嵁鍒嗘暟涓庤澧冿紝杈撳嚭鏈€缁堝喅绛?decision锛坴iolation/review/compliant锛変笌 confidence銆?

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愰噸瑕佹彁閱掞紙蹇呴』閬靛畧锛夈€?

1. 涓嶅厑璁稿洜涓衡€滄暣浣撴劅瑙夎繚瑙勨€濆氨鐩存帴缁欏嚭 violation锛屽繀椤诲熀浜庤鍒欎腑鏄庣‘鐨勬枃瀛楄瘉鎹€?
2. 涓€涓枃鏈彲浠ュ悓鏃跺寘鍚绉嶉闄╁洜瀛愬拰淇濇姢鍥犲瓙锛屽繀椤荤患鍚堝姞鍑忓垎锛岃€屼笉鏄€滃懡涓竴鍙ヨ瘽灏辩洿鎺ヨ繚瑙勨€濄€?
3. 淇濇姢鍥犲瓙锛堥闄╂彁绀恒€佸巻鍙茶鏄庛€佹ā鍨?鑰佸笀鏉ユ簮銆佸厤璐ｅ０鏄庣瓑锛夊懡涓椂锛屽繀椤诲噺鍒嗐€?

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愯鍒欒儗鏅紙渚涘弬鑰冿級銆?

涓嬮潰鏄?18 鏉′簨浠惰鍒欑殑璇︾粏鏂囨湰銆傛瘡鏉¤鍒欏潎涓恒€愬姞鍑忓垎鍒躲€戯細
- 銆愰闄╁姞鍒嗗洜瀛愩€戯細鍛戒腑鍒欐寜瑙勫垯缁欏嚭鐨勫垎鍊煎姞鍒嗭紙楂?+50~70銆佷腑 +20~40銆佷綆 +5~15锛夈€?
- 銆愪繚鎶ゅ噺鍒嗗洜瀛愩€戯細鍛戒腑鍒欐寜瑙勫垯缁欏嚭鐨勫垎鍊煎噺鍒嗭紙鏅€?-10~-20銆佸己 -30~-40锛夈€?
璇蜂弗鏍间緷鎹鍒欎腑鐨勫姞鍒?鍑忓垎椤硅瘑鍒苟绱姞锛屽緱鍒?risk_score锛?鈥?00锛夛紝涓嶈鈥滃懡涓嵆杩濊鈥濄€?

{context}

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愪簨浠跺悕绉扮櫧鍚嶅崟锛堝彧鑳戒粠涓€夋嫨 rule_name锛夈€?

__EVENT_WHITELIST__

褰撲綘鍦?risk_factors / protective_factors 涓紩鐢ㄨ鍒欐椂锛宺ule_name 蹇呴』鏉ヨ嚜涓婇潰鐨勭櫧鍚嶅崟涔嬩竴銆?

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愭墦鍒嗗師鍒欙紙璇蜂弗鏍兼墽琛岋級銆?

1. 楂橀闄╄涓猴紙閫氬父 +60 鍒嗗乏鍙筹紝鑼冨洿 +50 ~ +70锛夛細
   - 鐩存帴鎵胯瀹㈡埛鏈潵鎶曡祫鏀剁泭缁撴灉锛堝鈥滀繚璇佽禋閽扁€濃€滀竴瀹氱炕鍊嶁€濓級锛屽搴斾簨浠讹細鐩存帴鎵胯鏀剁泭銆?
   - 绗﹀悎瑙勫垯涓€滅洿鎺ヨ繚瑙勬儏褰⑩€濈殑瀹㈡埛鐩堝埄鏅掑崟銆佹垬缁╄惀閿€锛屼笖鍖呭惈鍏蜂綋閲戦/姣斾緥/鏃堕棿绛夈€?
   - 鍛樺伐涓汉鍙ｅ惢涓嬭揪涔板叆/鍗栧嚭/鎸佹湁/閲嶄粨绛夋寚浠わ紝鎴栦釜浜轰富瑙傛妧鏈瘖鑲★紙鏃犺€佸笀/杞欢/鍥㈤槦鑳屼功锛夈€?
   - 绱㈣瀹㈡埛鎵嬫満鍙风粰鑷繁浣跨敤銆佹寚瀵间娇鐢ㄤ粬浜鸿韩浠藉紑鎴枫€佷互姝ｅ紡鏈嶅姟閫€娆炬壙璇哄仛鎴愪氦鍗栫偣绛夈€?
   - 鍐掔敤鈥滄矆鏉ㄨ€佸笀鏈汉鈥濆悕涔夌粰绁?甯﹂槦/閫氱煡锛屾垨灏嗙瓥鐣ュ寘瑁呬负鈥滈噸浠撴墠閰嶅緱涓娾€濈瓑澶稿ぇ璇濇湳銆?

2. 涓闄╄涓猴紙閫氬父 +30 鍒嗗乏鍙筹紝鑼冨洿 +20 ~ +40锛夛細
   - 寮虹儓鏆楃ず鏀剁泭鎴栨垬缁╋紝浣嗘湭褰㈡垚瀹屽叏纭畾鎬ф壙璇恒€?
   - 澶稿ぇ鎶曠爺璋冪爺銆佷竴鎵嬭祫鏂欍€佹満鏋勬寔浠撶瓑锛屼絾鏈洿鎺ヨ惤鍒扮‘瀹氭€ф敹鐩婃壙璇恒€?
   - 浣跨敤鏁忔劅璇嶆眹骞跺甫鏈変竴瀹氭湭鏉ョ粨鏋滄剰鍛筹紝浣嗚姘斿瓨鍦ㄦā绯婂湴甯︺€?
   - 妯＄硦鐨勬定骞呭睍绀恒€佹垬缁╃綏鍒楋紝鎴栧寮€鎴蜂剑閲戠瓑瀛樺湪璇卞浣嗕笉鏋佺銆?

3. 浣庨闄╄涓猴紙閫氬父 +10 鍒嗗乏鍙筹紝鑼冨洿 +5 ~ +15锛夛細
   - 涓€鑸惀閿€璇皵銆佹儏缁己鍖栥€侀紦鍔辨€ц瘽鏈€佸績鎬佸缓璁剧瓑銆?
   - 娉涙硾鑰岃皥甯傚満銆佽鎯呫€佸浜у搧/鏈嶅姟鐨勬弧鎰忓害锛屼絾鏈洿鎺ユ瀯鎴愪笂杩伴珮/涓闄┿€?

4. 淇濇姢鍥犲瓙锛堝噺鍒嗭細-10 ~ -40锛夛細
   - **鏅€氫繚鎶ゅ洜瀛愶紙-10 ~ -20 鍒嗭級**锛氫緥濡傞闄╂彁绀恒€佹彁閱掓湁娉㈠姩銆佸己璋冮渶璋ㄦ厧銆佽瀹㈡埛鏍规嵁鑷韩鎯呭喌閫夋嫨绛夈€?
   - **寮轰繚鎶ゅ洜瀛愶紙-30 ~ -40 鍒嗭級**锛?
     - 鏄庣‘璇存槑鏄€滃巻鍙叉渚?鍘嗗彶鎴樼哗鈥濓紝涓嶆壙璇烘湭鏉ャ€?
     - 鏄庣‘璇存槑鏄€滄ā鍨?杞欢/鑰佸笀鍥㈤槦鈥濈殑绛栫暐缁撴灉锛岃€岄潪鍛樺伐涓汉鎷嶈剳琚嬨€?
     - 鏄庣‘鍐欏嚭鈥滈潪鎶曡祫寤鸿 / 鏈夐闄?/ 涓嶄繚璇佺泩鍒┾€濈瓑鍏嶈矗澹版槑銆?
     - 寮鸿皟瀹㈡埛闇€瀹屾垚椋庨櫓娴嬭瘎銆佸悎瑙勬祦绋嬶紝鎴栦笉鑳藉€熻捶鎶曡祫绛夊垰鎬х害鏉熴€?
   - 闂彿璇皵銆佷笉纭畾璇皵銆佸亣璁捐姘斻€佸弽闂紡鍚﹀畾銆佸鎴蜂富鍔ㄨ〃杈剧瓑锛屼篃鍙互瑙嗕负鍑忓急纭畾鎬х殑淇濇姢鍥犲瓙銆?

5. 鎬诲垎锛?
   - 灏嗘墍鏈夐闄╁洜瀛愮殑鍔犲垎涓庝繚鎶ゅ洜瀛愮殑鍑忓垎鐩稿姞锛屽緱鍒板師濮嬪垎鏁般€?
   - 鑻?< 0锛屽垯璁颁负 0锛涜嫢 > 100锛屽垯璁颁负 100銆?

銆愮壒鍒鏄庯細鍏充簬鈥滃共鎵伴闄╂祴璇勭嫭绔嬫€р€濄€?
- 鍙湁鍦ㄥ悓鏃舵弧瓒斥€滄槑纭寚瀹氬叿浣撻€夐」 / 鏁欏攩绡℃敼椋庨櫓绛夌骇鎴栨牳蹇冭韩浠戒俊鎭?/ 閫愰瑕佹眰姹囨姤骞剁籂姝ｇ瓟妗堚€濈瓑寮鸿瘉鎹椂锛? 
  鎵嶅簲瑙嗕负楂橀闄╁洜瀛愶紙+60 宸﹀彸锛夛紱鍚﹀垯涓€寰嬭涓轰腑浣庨闄╂垨淇濇姢鍥犲瓙銆? 
- 娴佺▼闄悓銆佽棰樸€佽В閲婇鐩惈涔夈€佸己璋冨瀹炲～鍐欍€佽鏄庡€熻捶/骞撮緞/椋庨櫓鎵垮彈鑳藉姏绛夊悎瑙勮姹傦紝  
  鍘熷垯涓婇兘搴斾綔涓轰繚鎶ゅ洜瀛愬噺鍒嗭紝鑰屼笉鏄姞鍒嗐€? 
- 褰撴枃鏈?*浠?*娑夊強椋庨櫓娴嬭瘎鍦烘櫙锛屼笖鍚屾椂鍑虹幇澶ч噺椋庨櫓鏁欒偛銆佹搷浣滄寚瀵笺€佸瀹炲～鍐欐彁绀虹瓑淇濇姢璇鏃讹紝  
  鎬讳綋 risk_score 搴旀槑鏄惧亸浣庯紝decision 鏇村€惧悜浜?"compliant" 鎴?"review"銆? 

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愬喅绛栭槇鍊硷紙蹇呴』鎵ц锛夈€?

鐢?risk_score 鍐冲畾鏈€缁?decision锛堜笌绯荤粺鍒ゅ畾涓€鑷达級锛?
1. 鑻?risk_score 鈮?30 鈫?decision = "violation"锛堣繚瑙勶級
2. 鑻?15 鈮?risk_score < 30 鈫?decision = "review"锛堝缓璁汉宸ュ鏍革級
3. 鑻?risk_score < 15 鈫?decision = "compliant"锛堝悎瑙勶級

缃俊搴?confidence 寤鸿锛?
- 0.8~1.0锛氳瘉鎹竻鏅帮紝澶氫釜楂橀闄╁洜瀛愬悓鍚戙€?
- 0.5~0.8锛氳瘉鎹瓨鍦紝浣嗘湁涓€瀹氭ā绯婃垨淇濇姢鍥犲瓙杈冨銆?
- 0.0~0.5锛氫粎鏈夊皯閲忎綆椋庨櫓鍥犲瓙鎴栬鍒欒澧冮珮搴︽ā绯娿€?

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愯緭鍑烘牸寮忥紙蹇呴』涓哄悎娉?JSON 涓斾笉寰楀寘鍚浣欐枃鏈級銆?

浣犲彧鑳借緭鍑轰竴涓?JSON 瀵硅薄锛屽瓧娈靛繀椤诲寘鍚紙澶у皬鍐欏畬鍏ㄤ竴鑷达級锛?

- "risk_score"锛?鈥?00 鐨勬暣鏁版垨娴偣鏁? 
- "decision"锛氬瓧绗︿覆锛屽彇鍊煎彧鑳芥槸 "violation" / "review" / "compliant"  
- "confidence"锛?鈥? 涔嬮棿鐨勫皬鏁? 
- "risk_factors"锛氭暟缁勶紝姣忎釜鍏冪礌鏄竴涓璞★紝鍖呭惈锛?
  - "rule_id"锛氬懡涓殑瑙勫垯缂栧彿锛?鈥?8锛?
  - "rule_name"锛氳鍒欏悕绉帮紙蹇呴』鏉ヨ嚜浜嬩欢鍚嶇О鐧藉悕鍗曪級
  - "level"锛?high" / "medium" / "low"
  - "weight"锛氭鏁帮紝鍔犲垎鍊硷紙濡?60 / 30 / 10锛?
  - "sentence"锛氬師鏂囦腑鑳戒綋鐜拌鍥犲瓙鐨勫叧閿彞
- "protective_factors"锛氭暟缁勶紝姣忎釜鍏冪礌鏄竴涓璞★紝鍖呭惈锛?
  - "rule_id"锛氬彲閫夛紝鑻ヨ兘瀵瑰簲鍒版煇鏉¤鍒欏垯濉啓
  - "rule_name"锛氬彲閫夛紝淇濇姢鍥犲瓙鍏宠仈鐨勮鍒欏悕鎴?"鍏ㄥ眬淇濇姢鍥犲瓙"
  - "weight"锛氳礋鏁帮紝鍑忓垎鍊硷紙濡?-20 / -30锛?
  - "sentence"锛氬師鏂囦腑浣撶幇淇濇姢鍥犲瓙鐨勫叧閿彞
- "summary_reason"锛氬瓧绗︿覆锛岀敤绠€娲佷腑鏂囩患鍚堣鏄庢湰娆¤瘎鍒嗕笌鍐崇瓥鐨勫師鍥?

銆愮‖鎬ц姹傘€戯細
- 鍙兘杈撳嚭涓婅堪 JSON锛屼笉鑳芥湁浠讳綍棰濆鏂囧瓧銆佹敞閲婃垨瑙ｉ噴銆?
- 鎵€鏈?rule_name 蹇呴』鏉ヨ嚜浜嬩欢鍚嶇О鐧藉悕鍗曘€?
- 鑻ユ病鏈変换浣曟槑鏄鹃闄╁洜瀛愶紝鍙护 risk_score 鎺ヨ繎 0锛宒ecision = "compliant"锛屽苟鍦?summary_reason 涓鏄庘€滄湭鍙戠幇鏄庢樉杩濊椋庨櫓锛屼粎瀛樺湪姝ｅ父涓氬姟/钀ラ攢/鏈嶅姟璇濇湳鈥濈瓑銆?

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愬緟妫€娴嬭亰澶╁唴瀹广€?

{input}

鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣
銆愭渶缁堟寚浠ゃ€?

璇蜂弗鏍兼寜鐓т笂杩扳€滄墦鍒嗗師鍒欌€濆拰鈥滆緭鍑烘牸寮忊€濓紝鍙緭鍑轰竴涓?JSON 瀵硅薄銆?
"""
        _event_whitelist = "\n".join(f"{i}. {self.RULE_NAMES[i]}銆? for i in range(1, 20)).rstrip("銆?)
        prompt = ChatPromptTemplate.from_template(_prompt_raw.replace("__EVENT_WHITELIST__", _event_whitelist))

        # 5. 妫€绱㈤厤缃細鎸夎鍒欏彫鍥炲畬鏁磋鍒欙紝涓?rerank锛堝弬鏁板凡鍦?__init__ 寮€澶翠粠 kwargs/env 璁剧疆锛?
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self._retrieve_k})
        self.retriever_with_score = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": self._retrieve_k, "score_threshold": self._retrieve_score_threshold}
        )

        self.chain = (
            {"context": RunnableLambda(self._retrieve_rules_full) | RunnableLambda(self._format_docs), "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """鏍煎紡鍖栨绱㈠埌鐨勬枃妗?""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _normalize_input_text(self, text: Any) -> str:
        """淇濊瘉妫€绱?棰勬祴鍏ュ彛鎷垮埌鐨勬槸瀛楃涓诧紝閬垮厤 dict/None 瀵艰嚧 TypeError銆?""
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            return str(text.get("input", text.get("context", "")) or "")
        return str(text)

    def _get_candidate_rule_ids(self, text: str) -> List[int]:
        """鐢ㄥ垎鍧楄涔夋绱?+ 鍏抽敭璇嶅尮閰嶅緱鍒板€欓€夎鍒?ID 鍒楄〃锛堝幓閲嶃€佷繚搴忥級銆?""
        text = self._normalize_input_text(text)
        seen: set = set()
        ordered: List[int] = []
        try:
            semantic_docs = self.retriever_with_score.invoke(text)
        except Exception:
            semantic_docs = self.retriever.invoke(text)
        for doc in semantic_docs:
            rid = doc.metadata.get("rule_id")
            if rid is not None and rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        for rule_id, _score, _kw in self._keyword_match_rules(text):
            if rule_id not in seen:
                seen.add(rule_id)
                ordered.append(rule_id)
        return ordered

    def _retrieve_rules_full(self, text: str) -> List[Document]:
        """鎸夎鍒欏彫鍥烇細鍏堝緱鍒板€欓€夎鍒?ID锛屽啀鍙栨瘡鏉¤鍒欑殑瀹屾暣鍘熸枃锛屾寜鍊欓€夐『搴忓彇 top N 鏉★紙涓嶅仛 rerank锛夈€?""
        text = self._normalize_input_text(text)
        candidate_ids = self._get_candidate_rule_ids(text)
        candidates: List[Tuple[int, str]] = []
        for rid in candidate_ids:
            full_text = self._full_rules_by_id.get(rid)
            if full_text:
                candidates.append((rid, full_text))
        if not candidates:
            return []
        candidates = candidates[: self._max_rules]
        return [
            Document(
                page_content=full_text,
                metadata={"rule_id": rid, "rule_name": self._get_rule_name_by_id(rid)}
            )
            for rid, full_text in candidates
        ]

    def _retrieve_hybrid(self, text: str) -> List[Document]:
        """娣峰悎妫€绱紙鍒嗗潡锛夛細浠呯敤浜庤皟璇曞姣旓紱涓绘祦绋嬪凡鏀逛负 _retrieve_rules_full銆?""
        try:
            semantic_docs = self.retriever_with_score.invoke(text)
        except Exception:
            semantic_docs = self.retriever.invoke(text)
        seen_chunks = {(doc.metadata.get("rule_id"), doc.metadata.get("chunk_type", "")) for doc in semantic_docs}
        result = list(semantic_docs)
        for rule_id, _score, _kw in self._keyword_match_rules(text)[:5]:
            for doc in self._rule_id_to_docs.get(rule_id, []):
                key = (doc.metadata.get("rule_id"), doc.metadata.get("chunk_type", ""))
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    result.append(doc)
        return result[:20] if len(result) > 20 else result

    def _initialize_vector_store(self):
        """灏嗗畬鏁寸殑18鏉¤鍒欏垎鍧楀瓨鍌ㄥ埌鍚戦噺搴擄紱鑻ュ瓨鍦ㄦ湭杩囨湡鐨勬湰鍦?FAISS 缂撳瓨鍒欑洿鎺ュ姞杞斤紝鍚﹀垯鏋勫缓骞惰惤鐩樸€?""
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(_src_dir, "rules.md")
        index_dir = os.path.join(_src_dir, "faiss_index")
        meta_file = os.path.join(index_dir, "meta.txt")

        def _build_full_rules_by_id_and_keyword():
            full_rules = self._get_full_rules_content()
            self._full_rules_by_id = {i + 1: full_rules[i] for i in range(len(full_rules))}
            self._build_rule_keyword_index()

        # 灏濊瘯浠庢湰鍦扮紦瀛樺姞杞斤紙闇€ rules.md 鏈彉鏇达級
        if os.path.isdir(index_dir) and os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    saved_mtime = f.read().strip()
                if os.path.isfile(rules_path) and saved_mtime == str(os.path.getmtime(rules_path)):
                    self.vectorstore = FAISS.load_local(
                        index_dir, self.embeddings, allow_dangerous_deserialization=True
                    )
                    _build_full_rules_by_id_and_keyword()
                    self._rule_id_to_docs = {}
                    for doc in getattr(self.vectorstore.docstore, "_dict", {}).values():
                        rid = doc.metadata.get("rule_id")
                        if rid is not None:
                            self._rule_id_to_docs.setdefault(rid, []).append(doc)
                    return
            except Exception:
                pass

        # 鏋勫缓鍚戦噺搴?
        full_rules = self._get_full_rules_content()
        self._full_rules_by_id = {i + 1: full_rules[i] for i in range(len(full_rules))}
        documents = []
        for i, rule_text in enumerate(full_rules):
            rule_id = i + 1
            rule_name = self._get_rule_name_by_id(rule_id)
            chunks = self._split_rule_into_chunks(rule_text, rule_id, rule_name)
            documents.extend(chunks)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._rule_id_to_docs = {}
        for doc in documents:
            rid = doc.metadata.get("rule_id")
            if rid is not None:
                self._rule_id_to_docs.setdefault(rid, []).append(doc)
        self._build_rule_keyword_index()

        # 钀界洏骞惰褰?rules.md mtime
        try:
            os.makedirs(index_dir, exist_ok=True)
            self.vectorstore.save_local(index_dir)
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(str(os.path.getmtime(rules_path)))
        except Exception:
            pass

    # 瑙勫垯姝ｆ枃婊戝姩绐楀彛锛氭寜銆屾銆嶅缓绱㈠紩锛宔mbedding 鑳藉懡涓换鎰忚繚瑙勭偣锛涢€佸叆 LLM 浠嶇敤 _full_rules_by_id 鏁存潯锛坃chunk_size/_chunk_overlap 鍦?__init__ 涓缃級

    def _split_rule_into_chunks(self, rule_text: str, rule_id: int, rule_name: str) -> List[Document]:
        """灏嗗崟鏉¤鍒欏垎鍓叉垚澶氫釜灏忓潡锛氭粦鍔ㄧ獥鍙ｈ鐩栧叏鏂?+ 璇箟娈碉紙鏍囬/杩濊鎯呭舰/鎺掗櫎鏉℃绛夛級銆?""
        chunks = []
        size, overlap = self._chunk_size, self._chunk_overlap
        step = max(1, size - overlap)
        # 婊戝姩绐楀彛瑕嗙洊鏁存潯瑙勫垯锛屼究浜庢绱㈠懡涓换鎰忎竴娈碉紙鍚悗鍗婃鎺掗櫎鏉℃绛夛級
        for i, start in enumerate(range(0, len(rule_text), step)):
            segment = rule_text[start : start + size]
            if not segment.strip():
                continue
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}: {rule_name}銆慭n{segment}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "window", "chunk_index": i}
            ))
        if not chunks:
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}: {rule_name}銆慭n{rule_text}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "window", "chunk_index": 0}
            ))

        # 璇箟娈碉細鏍囬銆佹牳蹇冮€昏緫銆佽繚瑙勬儏褰€佹帓闄ゆ潯娆俱€侀噸瑕佽鏄?
        # 1. 鎻愬彇鏍囬鍜屾牳蹇冮€昏緫
        title_match = re.search(r"### \d+\. (.*?)\n", rule_text)
        if title_match:
            title_part = title_match.group(1)
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}鏍囬銆憑title_part}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "title"}
            ))
        
        # 2. 鎻愬彇鏍稿績閫昏緫閮ㄥ垎锛堝鏋滄湁锛?
        core_logic_pattern = r"銆愭牳蹇冮€昏緫銆?*?(?=\n\n|$)"
        core_logic_match = re.search(core_logic_pattern, rule_text, re.DOTALL)
        if core_logic_match:
            core_logic = core_logic_match.group(0)
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}鏍稿績閫昏緫銆憑core_logic}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "core_logic"}
            ))
        
        # 3. 鎻愬彇鍏蜂綋杩濊鎯呭舰
        violation_pattern = r"鍏蜂綋杩濊鎯呭舰.*?(?=缁濆鎺掗櫎鏉℃|閲嶈璇存槑|鏈€缁堝垽鏂瓅$)"
        violation_match = re.search(violation_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if violation_match:
            violations = violation_match.group(0)
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}杩濊鎯呭舰銆憑violations}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "violation"}
            ))
        
        # 4. 鎻愬彇缁濆鎺掗櫎鏉℃
        exclusion_pattern = r"缁濆鎺掗櫎鏉℃.*?(?=\n\n|閲嶈璇存槑|鏈€缁堝垽鏂瓅$)"
        exclusion_match = re.search(exclusion_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if exclusion_match:
            exclusions = exclusion_match.group(0)
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}鎺掗櫎鏉℃銆憑exclusions}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "exclusion"}
            ))
        
        # 5. 鎻愬彇閲嶈璇存槑
        note_pattern = r"閲嶈璇存槑.*?(?=\n\n|鏈€缁堝垽鏂瓅$)"
        note_match = re.search(note_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if note_match:
            notes = note_match.group(0)
            chunks.append(Document(
                page_content=f"銆愯鍒檣rule_id}閲嶈璇存槑銆憑notes}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "note"}
            ))
        
        return chunks

    def _build_rule_keyword_index(self):
        """鏋勫缓瑙勫垯鍏抽敭璇嶇储寮?""
        self.rule_keywords = {}
        
        # 涓烘瘡鏉¤鍒欏畾涔夊叧閿瘝
        keyword_definitions = {
            1: ["淇濊瘉", "鎵胯", "涓€瀹?, "鑲畾", "缁濆", "璧氶挶", "鐩堝埄", "鏀剁泭", "鑾峰埄", "鍖呰禂", "绋宠禋", "淇濇湰"],
            2: ["鎶ュ枩", "璧氫簡", "鐩堝埄", "缈诲€?, "娑ㄥ仠", "鏈噾", "鏀剁泭", "鍥炶", "缈昏韩", "鎸佷粨", "鎴浘", "鏅掑崟"],
            3: ["澶ф定", "娑ㄥ仠", "杩炴澘", "鐙傞", "鏆存定", "娑ㄥ箙", "鐗涜偂", "濡栬偂", "鎴樼哗", "妗堜緥"],
            4: ["璋冪爺", "涓€鎵嬭祫鏂?, "鍐呭箷", "鐭ユ牴搴?, "浜嗗鎸囨帉", "鏈烘瀯", "鎸佷粨"],
            5: ["鐢佃瘽", "鎵嬫満鍙?],
            6: ["鎶撴定鍋?, "缈诲€?, "鍥炶", "鍥炴湰", "鏆存定", "鍚冭倝", "鎹￠挶", "绋宠禋", "鏈噾鏃犲咖"],
            7: ["寮€鎴?, "鍒稿晢", "浣ｉ噾", "鏈€浣?, "璇卞", "鍔犲井淇?],
            8: ["椋庨櫓娴嬭瘎", "閫塁", "閫夋渶楂?, "濉珮", "鏁欏攩", "淇敼", "鍒€?, "閿欒閫夐」"],
            9: ["鍚堝悓", "鐢熸晥", "璧峰", "涓嬪懆", "鏄庡ぉ", "鎺ㄨ繜", "鏃ユ湡"],
            10: ["鍌婚€?, "鑴戞畫", "绌烽", "鍨冨溇", "搴熺墿", "鐧界棿", "锠㈣揣", "楠備汉"],
            11: ["閫€娆?, "閫€璐?, "閫€閽?, "涓嶆弧鎰忛€€", "闅忔椂閫€", "鍏ㄩ閫€", "鏃犵悊鐢遍€€"],
            12: ["浠栦汉韬唤", "瀹朵汉韬唤", "鏈嬪弸韬唤", "鍊熻韩浠?, "鐢ㄥ埆浜?],
            13: ["涔板叆", "鍗栧嚭", "鎸佹湁", "鍑忎粨", "娓呬粨", "姝㈡崯", "鍋歍", "璋冧粨", "鍘嬪姏浣?],
            14: ["瀵规爣", "澶嶅埗", "涓€鏍?, "璧板娍", "娑ㄥ仠", "缈诲€?, "娑ㄥ箙", "鐩爣", "鍘嗗彶"],
            15: ["鏄庢棩", "灏剧洏", "寤轰粨", "璺熶笂", "浠ｇ爜", "涔板崠鍖洪棿", "閿佸畾", "鍚嶉"],
            16: ["娌堟潹鑰佸笀", "娌堣€佸笀", "浜查€?, "浜茶嚜", "浜茶嚜缁欑エ", "浜茶嚜甯﹂槦", "浜茶嚜閫氱煡", "浜茶嚜鎺ㄩ€?, "浜茶嚜闄即"],
            17: ["绀肩墿", "绾㈠寘", "璧犻€?, "鏀跺彈", "瀵勭粰鎮?, "涓€鐐瑰績鎰?, "瀵勪釜涓滆タ", "缁欐偍甯︾偣", "瀵勭偣鐗逛骇", "涓€鐐瑰皬鎰忔€?, "璇锋垜鍚冮キ", "鍒颁綘鐨勫煄甯傜帺", "浣犺瀹?],
            18: ["閲嶄粨", "灏忔墦灏忛椆", "寮€浠?, "鏈烘瀯閲嶄粨", "閲嶄粨鍙備笌", "涓婁粨浣?, "閲嶄粨鎿嶄綔", "鏈烘瀯閲嶄粨鑲?, "閲嶄粨鐜嬭偂", "璋冪爺澶嶆牳", "甯﹀鎴烽噸浠?]
        }
        
        for rule_id, keywords in keyword_definitions.items():
            self.rule_keywords[rule_id] = keywords

    def _get_rule_name_by_id(self, rule_id: int) -> str:
        """鏍规嵁瑙勫垯ID鑾峰彇瑙勫垯鍚嶇О"""
        return self.RULE_NAMES.get(rule_id, f"瑙勫垯{rule_id}")

    def _get_full_rules_content(self) -> List[str]:
        """
        浠庡閮?Markdown 鏂囦欢 `rules.md` 璇诲彇骞惰繑鍥?18 鏉″畬鏁磋鍒欏唴瀹广€?
        杩欐牱浣犲彧闇€瑕佺紪杈?Markdown 鏂囦欢鍗冲彲缁存姢瑙勫垯鏂囨湰銆?
        """
        rules_path = os.path.join(os.path.dirname(__file__), "rules.md")
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"瑙勫垯鏂囦欢涓嶅瓨鍦? {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 鎸夐《灞傝鍒欐爣棰?`### 1. ...` / `### 2. ...` 鍒囧垎
        lines = content.splitlines(keepends=True)
        rule_blocks: List[str] = []
        current: List[str] = []

        for line in lines:
            # 鍖归厤褰㈠ "### 1. 鏍囬" 鐨勮鍒欐爣棰橈紱宸插紑濮嬫敹闆嗗唴瀹规椂閬囧埌鏂扮殑鏍囬鍒欏紑濮嬫柊瑙勫垯
            if re.match(r"^###\s+\d+\.", line) and current:
                rule_blocks.append("".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            rule_blocks.append("".join(current).strip())

        # 杩囨护鎺夌┖鍧?
        rule_blocks = [blk for blk in rule_blocks if blk.strip()]

        if len(rule_blocks) != 18:
            raise ValueError(
                f"瑙ｆ瀽鍒扮殑瑙勫垯鏁伴噺涓?{len(rule_blocks)}锛岄鏈熶负 18锛岃妫€鏌?rules.md 涓《灞傛爣棰樻槸鍚︿负 '### 搴忓彿. 鏍囬' 鏍煎紡銆?
            )

        return rule_blocks

    def _keyword_match_rules(self, text: str) -> List[Tuple[int, int, List[str]]]:
        """鍩轰簬鍏抽敭璇嶅尮閰嶈鍒欙紙澶囩敤妫€绱㈡柟娉曪級銆傝繑鍥?(rule_id, score, matched_keywords)銆?""
        if not isinstance(text, str):
            text = self._normalize_input_text(text)
        text_lower = (text or "").lower()
        matches = []
        
        for rule_id, keywords in self.rule_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                matches.append((rule_id, score, matched_keywords))
        
        # 鎸夊垎鏁版帓搴?
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def predict(self, text: str, product_type=None) -> Dict[str, Any]:
        """棰勬祴杩濊鎯呭喌锛堝姞鍑忓垎 + 闃堝€硷級锛氭寜瑙勫垯璇嗗埆椋庨櫓鍥犲瓙/淇濇姢鍥犲瓙骞剁疮鍔犲緱鍒?risk_score锛屽啀鎸夐槇鍊煎垽杩濊銆?

        杩斿洖椋庨櫓鍒嗐€乨ecision銆乧onfidence銆佽Е鍙戜簨浠跺強鐞嗙敱绛夛紱鏄惁杩濊鐢?risk_score 涓?violation_threshold 鍐冲畾锛岄潪鈥滃懡涓嵆杩濊鈥濄€?

        product_type: 浜у搧绫诲瀷銆備粎褰卞搷鍥涗釜浜嬩欢鐨勭敓鏁堣寖鍥达細
        - "铏氬亣瀹ｄ紶妗堜緥绮鹃€夊強浜哄伐鎺ㄧエ" 浠呭湪 product_type 涓?1.0 鏃惰Е鍙戯紱
        - "鍐掔敤娌堟潹鑰佸笀鍚嶄箟" 浠呭湪 product_type 涓?2.0 鏃惰Е鍙戯紱
        - "瀵规姇鐮旇皟鐮旀椿鍔ㄥじ澶у浼?銆?澶稿ぇ瀹ｄ紶绛栫暐閲嶄粨鎿嶄綔" 浠呭湪 product_type 涓?3.0 鏃惰Е鍙戯紱
        - 鍏朵粬浜嬩欢涓庢湭浼?product_type 鏃跺潎涓哄叏閲忔娴嬨€?
        """
        # 闃堝€煎彲閫氳繃鐜鍙橀噺寰皟锛屾柟渚跨嚎涓婃牎鍑?
        def _float_env(name: str, default: float) -> float:
            v = os.getenv(name)
            try:
                return float(v) if v is not None and v.strip() != "" else default
            except Exception:
                return default

        # 鍔犲噺鍒嗗埗闃堝€硷細鈮?0 杩濊锛?5~30 浜哄伐澶嶆牳锛?15 鍚堣锛涘彲閫氳繃鐜鍙橀噺瑕嗙洊
        violation_threshold = _float_env("RISK_VIOLATION_THRESHOLD", 30.0)
        review_threshold = _float_env("RISK_REVIEW_THRESHOLD", 15.0)

        try:
            text = self._normalize_input_text(text)
            # 1. 璋冪敤 LLM 鑾峰彇鍘熷鍝嶅簲锛堥鏈熶负 JSON 瀛楃涓诧級
            raw_response = self.chain.invoke(text, config={"callbacks": []})
            if raw_response is None:
                raw_response = ""
            raw_response = str(raw_response).strip()

            # 2. 灏濊瘯瑙ｆ瀽 JSON锛堝閿欏鐞嗭細鎴彇鏈€澶栧眰鑺辨嫭鍙凤級
            parsed: Dict[str, Any] = {}
            parse_error = None
            try:
                parsed = json.loads(raw_response)
            except Exception as e1:
                parse_error = str(e1)
                try:
                    start = raw_response.find("{")
                    end = raw_response.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(raw_response[start : end + 1])
                        parse_error = None
                except Exception as e2:
                    parse_error = f"{parse_error} | {str(e2)}"

            if not isinstance(parsed, dict):
                raise ValueError(f"鏃犳硶瑙ｆ瀽妯″瀷 JSON 鍝嶅簲: {parse_error or '鏈煡閿欒'}")

            # 3. 璇诲彇鍩虹瀛楁锛堝甫榛樿鍊硷級
            risk_score_raw = parsed.get("risk_score", 0)
            try:
                risk_score = float(risk_score_raw)
            except Exception:
                risk_score = 0.0
            if risk_score < 0:
                risk_score = 0.0
            if risk_score > 100:
                risk_score = 100.0

            decision = str(parsed.get("decision", "") or "").strip().lower()
            confidence_raw = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0
            if confidence < 0:
                confidence = 0.0
            if confidence > 1:
                confidence = 1.0

            risk_factors = parsed.get("risk_factors") or []
            protective_factors = parsed.get("protective_factors") or []
            summary_reason = parsed.get("summary_reason") or ""

            # 4. 鏍规嵁闃堝€间笌 decision 寰楀埌甯冨皵杩濊缁撴灉
            violation_by_score = risk_score >= violation_threshold
            violation = decision == "violation" or violation_by_score

            # 5. 浠?risk_factors 涓娊鍙栬Е鍙戜簨浠跺強鐞嗙敱锛屽苟绱姣忎釜浜嬩欢鐨勯闄╁緱鍒?
            triggered_events: List[str] = []
            event_reasons: Dict[str, str] = {}
            event_scores: Dict[str, float] = {}

            if isinstance(risk_factors, list):
                for factor in risk_factors:
                    if not isinstance(factor, dict):
                        continue
                    rule_name = str(factor.get("rule_name") or "").strip()
                    if not rule_name:
                        continue
                    weight = factor.get("weight", 0)
                    try:
                        weight_val = float(weight)
                    except Exception:
                        weight_val = 0.0
                    # 鍙妸姝ｅ悜椋庨櫓鍥犲瓙褰撲綔鈥滆Е鍙戜簨浠垛€?
                    if weight_val <= 0:
                        continue
                    event_scores[rule_name] = event_scores.get(rule_name, 0.0) + weight_val
                    sentence = str(factor.get("sentence") or "").strip()
                    if rule_name not in triggered_events:
                        triggered_events.append(rule_name)
                    # 鑻ヨ浜嬩欢杩樻病鏈夌悊鐢憋紝鍒欒褰曚竴鍙ヤ唬琛ㄦ€ф枃鏈?
                    if rule_name not in event_reasons and sentence:
                        event_reasons[rule_name] = sentence

            triggered_event_str = ", ".join(triggered_events) if triggered_events else "鏃?

            # 6. 鎸?product_type 杩囨护鐗瑰畾浜嬩欢锛堜繚鎸佷笌鏃х増閫昏緫涓€鑷达級
            original_violation = violation
            original_triggered_event_str = triggered_event_str
            original_event_reasons = event_reasons.copy()
            pt = product_type
            if pt is not None:
                if pt in (1, "1", "1.0"):
                    pt = "1.0"
                elif pt in (2, "2", "2.0"):
                    pt = "2.0"
                elif pt in (3, "3", "3.0"):
                    pt = "3.0"
                else:
                    pt = None

            if pt is not None and triggered_event_str != "鏃?:
                normalized = re.sub(r"[锛屻€?锛沑s]+", ",", triggered_event_str)
                events = [e.strip() for e in normalized.split(",") if e.strip()]
                filtered_events: List[str] = []
                for e in events:
                    keep = True
                    for gated_name, required_pt in self.PRODUCT_TYPE_GATED_EVENTS.items():
                        if gated_name in e and pt != required_pt:
                            keep = False
                            break
                    if keep:
                        filtered_events.append(e)
                filtered_events = list(dict.fromkeys(filtered_events))

                filtered_event_reasons: Dict[str, str] = {}
                filtered_event_scores: Dict[str, float] = {}
                for evt_name, evt_reason in event_reasons.items():
                    keep = True
                    for gated_name, required_pt in self.PRODUCT_TYPE_GATED_EVENTS.items():
                        if gated_name in evt_name and pt != required_pt:
                            keep = False
                            break
                    if keep:
                        filtered_event_reasons[evt_name] = evt_reason
                        if evt_name in event_scores:
                            filtered_event_scores[evt_name] = event_scores[evt_name]

                triggered_event_str = ", ".join(filtered_events) if filtered_events else "鏃?
                event_reasons = filtered_event_reasons
                event_scores = filtered_event_scores

                # 鑻ユ寜浜у搧绫诲瀷杩囨护鍚庢病鏈夊墿浣欓珮椋庨櫓浜嬩欢锛屽垯鍙互瑙嗘儏鍐典笅璋冭繚瑙勭粨璁?
                if not filtered_events:
                    # 鑻?risk_score 涓昏鏉ユ簮浜庤杩囨护浜嬩欢锛岀悊璁轰笂搴旈檷浣庤繚瑙勭瓑绾с€?
                    # 涓轰簡瀹夊叏锛岃繖閲屼粎鍦?risk_score 涓嶉珮鏃讹紙灏忎簬 violation 闃堝€硷級鑷姩闄嶄负鍚堣銆?
                    if risk_score < violation_threshold:
                        violation = False

            # 濡傛灉娌℃湁浠讳綍瑙﹀彂浜嬩欢锛屼絾 risk_score 浠嶇劧寰堥珮锛宺eason 浣跨敤 summary_reason 鍏滃簳
            if not event_reasons and summary_reason:
                event_reasons = {"缁煎悎璇存槑": str(summary_reason)}

            # 7. 涓烘瘡涓簨浠剁悊鐢卞鍔犱簨浠跺緱鍒嗗墠缂€锛堣嫢鏈夛級锛屽苟鍦ㄦ暣浣撶悊鐢变腑闄勪笂鎬诲垎
            formatted_event_reasons: Dict[str, str] = {}
            for evt_name, evt_reason in event_reasons.items():
                prefix = ""
                if evt_name in event_scores:
                    prefix = f"[璇ヤ簨浠跺緱鍒嗙害 {event_scores[evt_name]:.1f}] "
                formatted_event_reasons[evt_name] = f"{prefix}{evt_reason}"

            final_decision = decision or (
                "violation"
                if violation_by_score
                else ("review" if risk_score >= review_threshold else "compliant")
            )
            if summary_reason:
                reason_text = f"鏁翠綋椋庨櫓鍒?{risk_score:.1f}锛屽喅绛?{final_decision}銆倇summary_reason}"
            else:
                reason_text = f"鏁翠綋椋庨櫓鍒?{risk_score:.1f}锛屽喅绛?{final_decision}銆傛ā鍨嬫湭缁欏嚭璇︾粏璇存槑"

            # 8. 灏?raw_response 瑙勮寖鍖栦负涓夎鏂囨湰杈撳嚭锛屾柟渚垮閮ㄧ郴缁熺洿鎺ヤ娇鐢?
            line_violation = f"鏄惁杩濊锛歿'鏄? if violation else '鍚?}"
            line_events = f"瑙﹀彂浜嬩欢锛歿triggered_event_str if triggered_event_str else '鏃?}"
            line_reason = f"鐞嗙敱锛歿reason_text}"
            raw_response = "\n".join([line_violation, line_events, line_reason])

            return {
                "raw_response": raw_response,
                "violation": violation,
                "triggered_event": triggered_event_str if violation else ("鏃? if triggered_event_str == "鏃? else triggered_event_str),
                "reason": reason_text,
                "event_reasons": formatted_event_reasons,
                # 鏂板璇勫垎鐩稿叧瀛楁锛屼緵涓婂眰浣跨敤
                "risk_score": risk_score,
                "decision": final_decision,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "_debug": {
                    "violation_threshold": violation_threshold,
                    "review_threshold": review_threshold,
                    "original_violation": original_violation,
                    "original_triggered_event": original_triggered_event_str,
                    "original_event_reasons": original_event_reasons,
                    "product_type": product_type,
                    "normalized_product_type": pt,
                } if product_type is not None else None,
            }

        except Exception as e:
            return {
                "raw_response": f"绯荤粺閿欒: {str(e)}",
                "violation": False,
                "triggered_event": "绯荤粺閿欒",
                "reason": str(e),
                "event_reasons": {},
                "risk_score": 0.0,
                "decision": "compliant",
                "confidence": 0.0,
                "risk_factors": [],
                "protective_factors": [],
            }


    def debug_retrieval(self, text: str) -> Dict[str, Any]:
        """璋冭瘯妫€绱㈣繃绋嬶細鍊欓€夎鍒?ID銆佹渶缁堥€佸叆鐨勫畬鏁磋鍒欏垪琛ㄣ€?""
        candidate_ids = self._get_candidate_rule_ids(text)
        full_rule_docs = self._retrieve_rules_full(text)
        keyword_matches = self._keyword_match_rules(text)
        return {
            "input": text,
            "candidate_rule_ids": candidate_ids,
            "candidate_rule_names": [self._get_rule_name_by_id(rid) for rid in candidate_ids],
            "keyword_matched_rules": [
                {
                    "rule_id": rule_id,
                    "rule_name": self._get_rule_name_by_id(rule_id),
                    "score": score,
                    "matched_keywords": matched_keywords
                }
                for rule_id, score, matched_keywords in keyword_matches[:5]
            ],
            "final_full_rules_ordered": [doc.metadata.get("rule_name", "鏈煡") for doc in full_rule_docs],
            "final_count": len(full_rule_docs),
            "max_rules": self._max_rules,
        }




