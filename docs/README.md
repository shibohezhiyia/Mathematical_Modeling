# 文档中心

本目录只保留面向使用者的文档，自动生成的大型 API 清单不再提交到仓库。

## 文档

- [USAGE.md](USAGE.md)：Python、Web 界面、多数据集和多模态调用示例。
- [项目主页](../README.md)：安装、功能概览和最小运行流程。
- [安全说明](../SECURITY.md)：API Key、公开部署和隐私文件处理规则。

## 本地运行

```bash
python web/app.py
```

打开 <http://127.0.0.1:5000>。

## 测试

```bash
python -m pytest -q
```

如果需要重新生成开发者 API 索引，可运行 `generate_api_docs.py`；生成文件属于本地产物，不应提交到公开仓库。
