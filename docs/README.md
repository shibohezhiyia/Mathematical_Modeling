# Mathematical Modeling - 文档中心

## 文档索引

| 文档 | 内容 |
|------|------|
| `API.md` | 自动生成的 API 文档（从 docstring 提取） |
| `USAGE.md` | Python API 使用示例与 Web API 端点说明 |
| `generate_api_docs.py` | API 文档自动生成脚本 |

## 快速使用

### Python API 示例

参见 [USAGE.md](USAGE.md) 中的 14 个完整示例，涵盖：
- 端到端分类/回归/聚类
- 自动数据清洗
- 超参数优化（贝叶斯/RL）
- 深度学习（MLP/CNN/LSTM/GRU/NAS）
- 多模态（图像/文本）
- 模型解释（SHAP/LIME）
- 公平性分析
- 大文件分块读取
- 结果缓存
- GPU 加速

### Web 界面

```bash
python web/app.py
# 访问 http://localhost:5000
```

## API 文档生成

```bash
python docs/generate_api_docs.py
# 生成 docs/API.md
```

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 查看覆盖率
python -m pytest tests/ --cov=core --cov-report=html
```
