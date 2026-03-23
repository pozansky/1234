# 🔧 Streamlit Cloud "App in the oven" 卡住解决方案

## 问题诊断

"Your app is in the oven" 一直卡住没有报错，通常是因为：

1. **构建超时** - sentence-transformers 模型下载太慢
2. **内存不足** - 模型加载需要大量内存
3. **依赖冲突** - 版本不兼容导致卡住
4. **缺少系统依赖** - 需要编译的包无法安装

## ✅ 已完成的优化

### 1. 固定依赖版本
- 使用精确版本号（`==`）而非范围（`>=`）
- 避免 Streamlit Cloud 解析依赖时耗时过长

### 2. 添加系统依赖
创建了 `packages.txt`，包含：
- `libopenblas-dev` - 数值计算库
- `gfortran` - Fortran 编译器
- `libhdf5-dev` - HDF5 数据格式支持

### 3. 优化配置
- `.streamlit/config.toml` - 优化服务器配置
- `.streamlit/startup.sh` - 预下载模型（可选）

### 4. 更新 .gitignore
- 忽略 `.streamlit/secrets.toml`（安全）
- 允许 FAISS 索引文件

## 📋 部署步骤

### 方案 A：标准部署（推荐）

#### 1. 提交所有文件到 Git

```bash
git add .streamlit/config.toml
git add .streamlit/startup.sh
git add packages.txt
git add requirements.txt
git add src/rag_engine.py
git add .gitignore
git commit -m "Optimize for Streamlit Cloud deployment"
git push origin main
```

#### 2. 在 Streamlit Cloud 配置

1. 登录 [Streamlit Cloud](https://streamlit.io/cloud)
2. 找到你的应用
3. 点击 "Settings" → "Secrets"
4. 添加：
```toml
[api_keys]
dashscope = "sk-eb015732b43844a7980f0daf9eba556d"
```

#### 3. 重启应用
- 点击 "Restart app"
- 等待构建完成（可能需要 10-20 分钟）

### 方案 B：使用启动脚本（如果方案 A 失败）

修改 Streamlit Cloud 设置：
1. 在应用设置中找到 "Advanced"
2. 设置启动命令：
```bash
bash .streamlit/startup.sh
```

## 🔍 排查步骤

### 查看构建日志

1. 在 Streamlit Cloud 应用页面
2. 点击右上角 "..." → "View app logs"
3. 查看是否有以下错误：

#### 错误：MemoryError
```
解决方案：
- 联系 Streamlit 支持申请更多内存
- 或使用更小的嵌入模型
```

#### 错误：Timeout
```
解决方案：
- 检查网络连接
- 使用 Hugging Face 镜像
- 减少依赖数量
```

#### 错误：Module not found
```
解决方案：
- 检查 requirements.txt
- 确保 packages.txt 包含系统依赖
```

## 🎯 快速测试

在本地模拟 Streamlit Cloud 环境：

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装系统依赖（如果需要）
# Ubuntu/Debian: sudo apt-get install libopenblas-dev gfortran libhdf5-dev

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 创建 secrets 文件
mkdir -p .streamlit
echo '[api_keys]' > .streamlit/secrets.toml
echo 'dashscope = "sk-eb015732b43844a7980f0daf9eba556d"' >> .streamlit/secrets.toml

# 5. 运行应用
streamlit run streamlit_app.py
```

## 📊 预期构建时间

- **首次构建**：15-30 分钟（下载所有依赖和模型）
- **后续构建**：5-10 分钟（使用缓存）

## 🆘 如果仍然卡住

### 临时解决方案

1. **简化应用** - 暂时移除 sentence-transformers，测试基本功能
2. **使用替代模型** - 使用更轻量的嵌入模型
3. **联系支持** - [Streamlit 社区论坛](https://discuss.streamlit.io/)

### 备选部署方案

- **Hugging Face Spaces** - 免费提供 GPU
- **Render** - 免费 tier
- **Railway** - 免费额度
- **Vercel** - 适合轻量应用

## 📝 检查清单

部署前确认：

- [ ] `requirements.txt` 已更新（使用精确版本）
- [ ] `packages.txt` 已创建
- [ ] `.streamlit/secrets.toml` 已配置（本地测试）
- [ ] FAISS 索引文件已提交到 Git
- [ ] 所有文件已推送
- [ ] Streamlit Cloud Secrets 已配置

## 💡 优化建议

1. **使用缓存** - 在代码中缓存模型加载
2. **懒加载** - 只在需要时加载模型
3. **减少依赖** - 移除不必要的包
4. **监控资源** - 查看 Streamlit Cloud 资源使用情况

---

**下一步**：提交更改并重新部署，如果还有问题，请提供构建日志。
