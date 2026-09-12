# 评测冻结

在读取任何未见题、最终测试或外部评审材料前，先生成内容寻址的冻结清单。清单只保存源码、脚本、Web 和公开配置文件的哈希、依赖版本、随机种子、方法和预算；`data/`、`workspace/` 与题面正文不会进入清单。

```powershell
python scripts/create_evaluation_freeze.py `
  --protocol-id pilot-blind-v1 `
  --seed 20260911 `
  --method baseline `
  --method current-system `
  --max-seconds 1800 `
  --max-memory-mb 4096 `
  --output data/reports/evaluation_freeze.json
```

冻结文件应由题目保管者保存一份。评测前后可调用 `verify_evaluation_freeze` 检查源码是否改变；出现 `mismatch` 时，必须重新冻结并从头运行，不能把旧结果和新代码混合。

冻结证明的是运行条件一致性，不是模型正确性。真正的未见题结论仍需要题目在项目外保管、独立评分者解封以及预注册的评分量规。
