# 延迟嵌入潜状态候选

`discover_delay_latent_states` 将观测序列拼成有界延迟嵌入，并在训练时间段上做 SVD/PCA，返回低维坐标、基、解释方差和留出重建误差。它适合作为“状态空间可能未闭合”的一个候选分支。

```python
from core.latent_state_discovery import discover_delay_latent_states

candidate = discover_delay_latent_states(times, observations, ["x", "y"])
```

输出的坐标是数学上的重建坐标，不自动对应溶解氧、消费者信心等物理变量。`tested_not_falsified` 只表示当前延迟、训练/留出划分和重建阈值下没有被这项检查否定；它不证明存在真实隐变量，也不解决可辨识性、观测偏差或外生输入竞争。
