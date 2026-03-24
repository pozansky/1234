#!/usr/bin/env python
"""
调试 FAISS 索引生成
详细显示每一步的执行情况和错误信息
"""

import os
import sys
import traceback

# 确保项目路径在 sys.path 中
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
  
print("=" * 80)
print("FAISS 索引生成调试脚本")
print("=" * 80)

# 步骤 1：检查基本文件
print("\n[步骤 1] 检查基本文件...")
rules_path = os.path.join(root_dir, "src", "rules.md")
print(f"  rules.md 存在：{os.path.exists(rules_path)}")

if os.path.exists(rules_path):
    print(f"  rules.md 大小：{os.path.getsize(rules_path)} 字节")
    print(f"  rules.md 修改时间：{os.path.getmtime(rules_path)}")
else:
    print("  ❌ rules.md 不存在！无法生成 FAISS 索引")
    sys.exit(1)

# 步骤 2：检查 faiss_index 目录
print("\n[步骤 2] 检查 faiss_index 目录...")
index_dir = os.path.join(root_dir, "src", "faiss_index")
print(f"  目录路径：{index_dir}")
print(f"  目录存在：{os.path.exists(index_dir)}")

if os.path.exists(index_dir):
    try:
        files = os.listdir(index_dir)
        print(f"  目录内容：{files}")
        
        # 检查是否有实际内容
        for f in files:
            file_path = os.path.join(index_dir, f)
            if os.path.isfile(file_path):
                print(f"    - {f}: {os.path.getsize(file_path)} 字节")
    except Exception as e:
        print(f"  ❌ 无法列出目录内容：{e}")
else:
    print(f"  目录不存在，将创建：{index_dir}")
    os.makedirs(index_dir, exist_ok=True)
    print(f"  ✅ 目录已创建")

# 步骤 3：尝试初始化引擎
print("\n[步骤 3] 初始化 ComplianceRAGEngine...")
print("  这可能需要几分钟下载模型，请耐心等待...")

try:
    from src.rag_engine import ComplianceRAGEngine
    
    print("  ✅ 导入成功")
    print("  开始初始化引擎...")
    
    # 手动跟踪初始化过程
    import time
    start_time = time.time()
    
    engine = ComplianceRAGEngine()
    
    elapsed = time.time() - start_time
    print(f"  ✅ 引擎初始化完成（耗时：{elapsed:.2f}秒）")
    
except Exception as e:
    print(f"  ❌ 初始化失败：{e}")
    print("\n  详细错误堆栈：")
    traceback.print_exc()
    sys.exit(1)

# 步骤 4：检查 vectorstore 是否创建
print("\n[步骤 4] 检查 vectorstore...")
if hasattr(engine, 'vectorstore') and engine.vectorstore is not None:
    print("  ✅ vectorstore 已创建")
    
    # 检查文档数量
    try:
        doc_count = len(engine.vectorstore.docstore._dict)
        print(f"  文档数量：{doc_count}")
    except Exception as e:
        print(f"  ⚠️  无法获取文档数量：{e}")
    
    # 步骤 5：手动保存 vectorstore
    print("\n[步骤 5] 手动保存 vectorstore 到 faiss_index 目录...")
    try:
        print(f"  保存路径：{index_dir}")
        engine.vectorstore.save_local(index_dir)
        print("  ✅ save_local() 调用成功")
        
        # 验证保存的文件
        files = os.listdir(index_dir)
        print(f"  保存后的文件列表：{files}")
        
        for f in files:
            file_path = os.path.join(index_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"    - {f}: {size} 字节")
        
        # 检查必需文件
        required_files = ["index.faiss", "index.pkl", "meta.txt"]
        missing = [f for f in required_files if f not in files]
        
        if missing:
            print(f"  ⚠️  缺少必需文件：{missing}")
        else:
            print(f"  ✅ 所有必需文件都已生成")
            
    except Exception as e:
        print(f"  ❌ 保存失败：{e}")
        print("\n  详细错误堆栈：")
        traceback.print_exc()
else:
    print("  ❌ vectorstore 未创建")
    print("  可能的原因：")
    print("    1. rules.md 内容为空")
    print("    2. _initialize_vector_store() 方法执行失败")
    print("    3. FAISS 库有问题")

# 步骤 6：检查 rules.md 的 mtime 是否已记录
print("\n[步骤 6] 检查 meta.txt...")
meta_file = os.path.join(index_dir, "meta.txt")
if os.path.exists(meta_file):
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            mtime_str = f.read().strip()
        print(f"  meta.txt 内容：{mtime_str}")
        print(f"  rules.md 当前 mtime: {os.path.getmtime(rules_path)}")
        
        if float(mtime_str) == os.path.getmtime(rules_path):
            print("  ✅ mtime 匹配，下次启动会直接加载缓存")
        else:
            print("  ⚠️  mtime 不匹配，下次启动会重新构建")
    except Exception as e:
        print(f"  ❌ 读取 meta.txt 失败：{e}")
else:
    print("  ⚠️  meta.txt 不存在")

print("\n" + "=" * 80)
print("调试完成！")
print("=" * 80)
