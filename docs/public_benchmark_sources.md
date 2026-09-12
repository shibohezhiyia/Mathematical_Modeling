# 公开题源目录

`examples/public_benchmark_sources.json` 是公开仓库中的题源索引，不是题面镜像，也不是答案库。每条记录包含官方页面、结构标签和（如已在本机封存）封存案例 ID；题面、附件和参考评分标准只以哈希形式保存在 `blind_benchmark` 的受控目录中。

## 设计边界

- 题面版权和竞赛期间的内容隔离：仓库不复制正文、附件或答案。
- 题源可复核：所有链接必须是 HTTPS，且限定在已登记的官方域名。
- 结构标签只用于分层抽样和报告，不是模型答案，不参与选模。
- `source_listed` 只表示官方页面已登记；`sealed_locally` 表示本机有封存哈希，仍不等于有独立评分标准。

验证命令：

```powershell
python scripts/validate_public_benchmark_sources.py
```

目前公开目录已登记 COMAP 2023–2025 题源和 2025 CUMCM 官方题源页；实际正确率必须在独立评分标准解封后报告，不能由“题面可访问”推导。
