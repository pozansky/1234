#!/usr/bin/env python
"""
测试 requirements.txt 中的包是否都能正常导入
在本地运行此脚本，确保所有依赖都能正常工作
"""

import sys

def test_import(package_name, import_name=None):
    """测试导入某个包"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ {package_name}: ERROR - {e}")
        return False

def main():
    print("=" * 60)
    print("测试依赖包导入...")
    print("=" * 60)
    
    # 核心包
    packages = [
        ("streamlit", "streamlit"),
        ("openai", "openai"),
        ("langchain", "langchain"),
        ("langchain-core", "langchain_core"),
        ("langchain-community", "langchain_community"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-huggingface", "langchain_huggingface"),
        ("pydantic", "pydantic"),
        ("python-dotenv", "dotenv"),
        ("PyYAML", "yaml"),
        ("httpx", "httpx"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("pypdf", "pypdf"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(test_import(pkg_name, import_name))
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"结果：{passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有依赖包都能正常导入！")
        return 0
    else:
        print(f"⚠️  有 {total - passed} 个包导入失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
