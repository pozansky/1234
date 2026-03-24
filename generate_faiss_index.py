#!/usr/bin/env python
"""
生成 FAISS 索引文件
用于 Streamlit Cloud 部署

运行方式：
python generate_faiss_index.py
"""

import os
import sys

# 确保项目路径在 sys.path 中
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print("=" * 60)
print("开始生成 FAISS 索引...")
print("=" * 60)

try:
    # 导入引擎（会自动构建向量库）
    print("\n1️⃣  导入 ComplianceRAGEngine...")
    from src.rag_engine import ComplianceRAGEngine
    
    print("2️⃣  初始化引擎（这可能需要几分钟下载模型）...")
    engine = ComplianceRAGEngine()
    
    print("3️⃣  检查 FAISS 索引是否已生成...")
    index_dir = os.path.join(root_dir, "src", "faiss_index")
    
    if os.path.exists(index_dir):
        files = os.listdir(index_dir)
        print(f"✅ FAISS 索引已生成在：{index_dir}")
        print(f"   文件列表：{files}")
        
        # 验证文件是否完整
        required_files = ["index.faiss", "index.pkl", "meta.txt"]
        missing_files = [f for f in required_files if f not in files]
        
        if missing_files:
            print(f"⚠️  缺少文件：{missing_files}")
        else:
            print("✅ 所有必需文件都已生成！")
    else:
        print(f"❌ FAISS 索引目录不存在：{index_dir}")
        print("   请检查代码是否有错误")
    
    print("\n" + "=" * 60)
    print("FAISS 索引生成完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 生成失败：{e}")
    print("\n可能的原因：")
    print("1. 缺少依赖包 - 运行：pip install -r requirements.txt")
    print("2. 模型下载失败 - 检查网络连接")
    print("3. rules.md 文件不存在或格式错误")
    print("\n详细错误信息：")
    import traceback
    traceback.print_exc()
    sys.exit(1)
