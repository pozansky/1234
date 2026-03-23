# 🚀 快速修复 "App in the oven" 问题

## 立即执行

### 步骤 1：提交所有优化文件

```bash
# 复制粘贴这些命令到终端
git add -A
git commit -m "Fix: Optimize Streamlit Cloud deployment with exact versions and system deps"
git push origin main
```

### 步骤 2：在 Streamlit Cloud 配置 Secrets

1. 打开 https://streamlit.io/cloud
2. 找到你的应用
3. 点击 **Settings** → **Secrets**
4. 粘贴以下内容：

```toml
[api_keys]
dashscope = "sk-eb015732b43844a7980f0daf9eba556d"
```

5. 点击 **Save**

### 步骤 3：重启应用

1. 回到应用页面
2. 点击 **Restart app**
3. 等待构建完成（**首次可能需要 20-30 分钟**）

---

## 📊 构建进度判断

正常的构建流程：

1. ✅ **Cloning repository** (1-2 分钟)
2. ✅ **Installing system packages** (2-3 分钟)
3. ✅ **Installing Python dependencies** (5-10 分钟)
4. ✅ **Downloading models** (5-15 分钟) ← 最耗时
5. ✅ **Starting app** (1-2 分钟)

如果卡在某个步骤超过 30 分钟，说明有问题。

---

## 🔍 如何查看构建日志

### 方法 1：实时日志
1. 在应用页面点击右上角 **...**
2. 选择 **View app logs**
3. 可以看到实时构建进度

### 方法 2：部署历史
1. 点击 **Settings**
2. 滚动到 **Deployments**
3. 点击最近的部署查看日志

---

## ⚠️ 常见问题

### Q1: 卡在 "Installing Python dependencies" 超过 20 分钟
**原因**：依赖版本冲突或网络问题

**解决**：
```bash
# 在本地测试 requirements.txt
pip install -r requirements.txt
```

### Q2: 显示 "Build failed" 但没有详细信息
**解决**：
1. 检查 `packages.txt` 是否存在
2. 确认 `requirements.txt` 格式正确
3. 查看是否有循环依赖

### Q3: 构建成功但启动失败
**可能原因**：
- API Key 未配置
- FAISS 索引文件缺失
- 内存不足

**解决**：
1. 检查 Secrets 配置
2. 确认 `src/faiss_index/` 在 Git 中
3. 联系 Streamlit 支持

---

## 🎯 验证部署成功

部署成功后，你应该看到：

1. ✅ 应用标题 "🔍 合规检测系统"
2. ✅ 左侧有配置面板
3. ✅ 可以输入文本并检测
4. ✅ 检测结果正常显示

---

## 💡 加速构建的小技巧

### 1. 使用 Hugging Face 镜像
在 `requirements.txt` 顶部添加：
```
# 使用镜像加速下载
--extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 减少依赖
如果不需要某些功能，可以移除对应依赖：
- `pypdf` - PDF 处理
- `qwen` - 如果只用 DashScope

### 3. 预构建镜像
使用 Docker 自定义镜像（高级用户）

---

## 🆘 紧急联系

如果以上方法都无效：

1. **Streamlit 社区**：https://discuss.streamlit.io/
2. **GitHub Issues**：https://github.com/streamlit/streamlit/issues
3. **Discord**：https://discord.gg/streamlit

---

**现在就去提交并重新部署吧！** 🚀
