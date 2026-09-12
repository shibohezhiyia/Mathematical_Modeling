# 模块引用审计

运行 `python scripts/audit_module_usage.py` 或访问
`/api/research/module-catalog?audit=1`，可按模块目录输出源码、测试和 Web 引用数量，
并列出 `catalog_only_keys`。审计不导入模块、不执行代码，也不能证明运行时覆盖率或
模型质量；它用于避免“文件存在/登记了”被误报成“主流程已接入”。
