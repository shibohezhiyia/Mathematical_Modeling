# 轻量图消息传递交互筛查

`core.gnn_interaction_screen` 是一个可选的 PyTorch 后端：每个数值列作为节点，每行作为一个样本图，学习 1–3 层带 sigmoid 边门的消息传递，再用验证集评估预测误差。它用于发现“哪些变量组合有助于预测目标”，不等价于 GNN 因果发现，也不构成机制证明。

## Web 接口

`POST /api/research/gnn-interactions` 接收：

- `rows`：最多 2000 行 JSON 对象；
- `target`：目标列；
- 可选 `columns`、`epochs`、`hidden_dim`、`restarts`、`message_layers`、`group_column`、`time_column`、`dynamic_windows` 和 `random_state`。

返回结果会包含训练/验证行数、每次重启的验证 RMSE、边门强度和跨重启稳定性。缺少 PyTorch 时返回 `unavailable`，不会偷偷退化成偏相关并冒充 GNN。

研究助手主流程默认关闭该可选后端以避免大表任务突然增加训练成本；在研究请求中显式设置 `enable_gnn_screen=true`，并同时提供明确的 `target`（单表可写列名，多表写 `数据集.列名`），结果才会进入 `specialized_results.gnn_interaction_screen`。它只对明确目标运行，不会根据列名擅自猜目标。

主流程会把数据画像中满足重复实体数量条件的 ID 候选和可靠的 datetime 列传入
`group_column`/`time_column`；因此在实体或时序数据上不会因为调用方遗漏参数而默认随机行切分。
无法确认实体/时间语义时才保留随机切分，并在结果的 `split_policy` 中明确记录。

当显式提供 `time_column` 且 `dynamic_windows>=2` 时，系统会按时间切成最多五个滚动窗口，
在每个窗口独立训练并输出 `dynamic_edge_persistence`。该结果是时间局部的预测性交互稳定性，
用于发现结构漂移，不是动态图因果发现；窗口不足 60 行时安全返回 `not_assessed`。

## 证据边界

数据划分只在训练集估计标准化参数，验证集不参与训练；指定 `group_column` 时按实体组留出，指定 `time_column` 时按时间尾部留出，避免同一实体或未来记录泄漏。`stability` 只表示有限重启下的重复出现比例；`predictive message-passing interaction` 不能直接改写为因果箭头，仍需题面机制、干预或独立实验支持。
