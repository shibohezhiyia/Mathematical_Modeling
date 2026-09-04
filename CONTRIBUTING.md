# 参与贡献

欢迎提交问题和改进建议。

## 提交前检查

```bash
python -m py_compile web/app.py
node --check web/static/js/app.js
python -m pytest -q
```

请不要提交 `data/`、`workspace/`、日志、缓存、模型权重、真实数据或任何 API 密钥。涉及安全问题请按照 `SECURITY.md` 私下报告，不要直接发布利用细节。
