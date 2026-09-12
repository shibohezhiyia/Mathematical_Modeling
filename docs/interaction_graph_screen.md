# 多变量交互图筛查

`core.interaction_graph_screen.discover_interaction_graph` 对单个数据表的数值变量计算带岭正则的偏相关，并用固定种子的重采样统计边稳定性。它用于发现“在控制其他变量后仍值得检查”的交互候选，结果会接入研究输出的 `specialized_results.interaction_graph_screen`。

该结果不是 GNN、因果发现或物理机理证明。边只能解释为有限样本下的条件关联；未通过稳定性阈值的关系不会进入候选边，样本不足返回 `not_assessed`。
