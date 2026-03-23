# Streamlit Cloud 部署指南

## ✅ 已修复的问题

### 1. **API Key 管理**
- ❌ 之前：硬编码在代码中
- ✅ 现在：使用 `.streamlit/secrets.toml` 管理

### 2. **requirements.txt 优化**
- 移除了重复的 `sentence-transformers`
- 添加了版本约束，确保兼容性
- 添加了缺失的 `httpx` 依赖

### 3. **代码兼容性**
- 修改 `src/rag_engine.py` 以支持从 Streamlit Secrets 读取 API Key
- 保持向后兼容，本地开发不受影响

## 📋 部署步骤

### 第一步：检查 Git 仓库

确保以下文件已提交到 Git：

```bash
# 检查 FAISS 索引文件是否在仓库中
git ls-files src/faiss_index/

# 检查 .streamlit/secrets.toml 是否存在
git ls-files .streamlit/secrets.toml

# 检查 requirements.txt
git ls-files requirements.txt
```

### 第二步：解决 FAISS 索引文件问题

你的 `.gitignore` 忽略了 `*.faiss` 和 `*.pkl` 文件，这会导致 Streamlit Cloud 无法获取向量索引。

**解决方案（选择其一）：**

#### 方案 A：在 `.gitignore` 中排除 FAISS 索引目录
```gitignore
# 注释掉或修改这两行
# *.faiss
# *.pkl

# 或者明确保留 src/faiss_index 目录
!src/faiss_index/*
```

#### 方案 B：在代码中动态构建索引（推荐）
如果索引文件太大，可以在应用启动时动态构建。

### 第三步：推送到 Git 仓库

```bash
git add .streamlit/secrets.toml
git add requirements.txt
git add src/rag_engine.py
git commit -m "Fix Streamlit Cloud deployment issues"
git push
```

### 第四步：在 Streamlit Cloud 配置

1. 登录 [Streamlit Cloud](https://streamlit.io/cloud)
2. 点击 "New app"
3. 选择你的 Git 仓库
4. 设置：
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.10` 或更高
   - **Advanced settings**（可选）: 如果需要额外环境变量

### 第五步：配置 Secrets（重要！）

在 Streamlit Cloud 的应用设置中，点击 "Secrets" 并添加：

```toml
[api_keys]
dashscope = "sk-eb015732b43844a7980f0daf9eba556d"
```

## 🔍 常见部署错误及解决方案

### 错误 1：ModuleNotFoundError: No module named 'httpx'
**原因**：依赖缺失  
**解决**：已在 `requirements.txt` 中添加 `httpx>=0.24.0`

### 错误 2：FAISS index not found
**原因**：FAISS 索引文件未上传到 Git  
**解决**：修改 `.gitignore` 或动态构建索引

### 错误 3：API Key 无效或缺失
**原因**：未配置 Secrets  
**解决**：在 Streamlit Cloud 配置 Secrets

### 错误 4：MemoryError / 内存不足
**原因**：`sentence-transformers` 模型太大  
**解决**：
- 在 Streamlit Cloud 设置中申请更多资源
- 或使用更小的嵌入模型

### 错误 5：构建超时（>30 分钟）
**原因**：下载模型和依赖时间过长  
**解决**：
- 在 `requirements.txt` 中指定精确版本号
- 考虑使用缓存

## 📊 部署后验证

部署成功后，访问你的应用并测试：

1. ✅ 页面正常加载
2. ✅ 输入测试文本能正常检测
3. ✅ 检测结果正确显示
4. ✅ 无报错信息

## 🆘 获取帮助

如果仍有问题，请提供：
- Streamlit Cloud 的部署日志
- 具体的错误信息
- 你的 Git 仓库链接（可选）

## 📝 本地测试

在本地模拟 Streamlit Cloud 环境测试：

```bash
# 清除本地缓存
rm -rf .streamlit/secrets.toml

# 重新创建 secrets 文件
echo '[api_keys]' > .streamlit/secrets.toml
echo 'dashscope = "sk-eb015732b43844a7980f0daf9eba556d"' >> .streamlit/secrets.toml

# 运行应用
streamlit run streamlit_app.py
```
