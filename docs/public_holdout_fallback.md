# 无法获得封存评分时的替代评测

当赛事方不提供未公开题面和独立评审分数时，项目采用公开数据的“标签不进入拟合器”替代方案：

1. 固定公开数据源 SHA-256、随机种子和训练/测试切分；
2. 训练函数只接收 `X_train, y_train`，测试标签只写入临时封存文件，再交给独立评分器进程；
3. 对时间序列使用时间顺序切分，对普通表使用固定随机切分；
4. 先运行 Ridge 基线，再运行 ExtraTrees 处理；
5. 记录泄漏列、缺失标记、源哈希和切分哈希；
6. 使用每个任务相对 RMSE 的配对置换检验和 bootstrap 区间。

运行命令：

```powershell
python scripts/run_external_data_benchmark.py `
  --output data/reports/public_holdout_fallback.json
```

当前本机结果为 6/6 个案例完成（新增 UCI Airfoil Self-Noise），处理相对 Ridge 的平均相对 RMSE 变化为 `-0.263771`，bootstrap 95% CI=`[-0.431034,-0.098282]`，精确符号置换双侧 `p=0.0625`。这不是统计显著改进，也不是“真实未见题准确率”；数据和标签均公开，因此结果只能说明公开数据上的可复现预测差异。完整记录见 [`external_validation_20260911.md`](external_validation_20260911.md)。

评分器入口是 `python -m core.independent_holdout`，只读取真值/预测向量，不能导入模型或训练数据。它不是操作系统级沙箱，也不改变公开标签的性质。

该替代方案的用途是：在等待外部封存评审期间发现泄漏、切分、资源和回归错误。只有取得项目外保管的题面、参考材料和独立评审，才允许使用 `blind_statistics` 报告真实未见题准确率或显著性。
