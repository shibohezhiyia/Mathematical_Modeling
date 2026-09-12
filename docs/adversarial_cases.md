# 对抗案例生成

`build_adversarial_cases` 按固定种子生成端点、退化、中点、近边界、随机和扰动输入。它只提供测试点，不自行声称违反约束；调用方应把这些案例交给 `run_counterexample_suite` 和受控模型评价器，才能形成反例证据。
