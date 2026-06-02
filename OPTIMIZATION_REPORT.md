# Mathematical Modeling 优化报告

## 项目概览

这是一个面向数据建模比赛的端到端智能分析引擎，涵盖数据加载、类型识别、缺失值智能分析、自动性能调度、自动编码、特征选择、K折交叉验证、多模型并行训练、模型融合等全链路能力。

**核心规模**：
- 核心代码：26,656 行
- 核心文件：62 个 Python 文件
- 最大文件：`core/modeling_engine.py` (2,865 行)

---

## 代码层面优化

### 1. 🔴 向量化优化（高优先级）

**问题**：发现 443 处 `apply_along_axis` / `iterrows` / `iteritems` / `itertuples` 使用，以及 54 处 `for...in range` 循环。

**影响文件**：
- `core/modeling_engine.py` - `np.apply_along_axis` 用于硬投票
- `core/modeling_engine.py` - `pd.apply(lambda...)` 用于编码
- 多处 `iterrows()` 遍历 DataFrame

**优化方案**：
- 硬投票：`np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr)` → `scipy.stats.mode(arr, axis=1).mode.flatten()`
- 编码 apply：预计算映射字典，用 `map()` 替代 apply
- iterrows：用 `df[col].values` 向量化操作替代

### 2. 🟡 数值稳定性优化（中优先级）

**问题**：
- `core/ot_reweighting.py:213`：`np.linalg.solve(K_ss_reg, kappa)` 未检查矩阵条件数
- `core/modeling_engine.py`：回归权重计算使用 `1.0 / (rmse + 1e-6)`，可能因 rmse=0 导致无穷大
- 多处 `np.unique()` 未使用 `return_counts=True` 缓存结果

**优化方案**：
- 添加矩阵条件数检查，条件数过高时回退到 `np.linalg.lstsq`
- 使用 `np.divide(1.0, rmse + 1e-6, out=np.zeros_like(rmse), where=rmse!=0)` 避免除零警告
- 缓存 `np.unique` 结果避免重复计算

### 3. 🟡 内存优化（中优先级）

**问题**：
- `pd.concat` / `pd.merge` 使用频繁（17处）
- DataFrame 类型转换可能产生不必要的副本

**优化方案**：
- 使用 `pd.concat(..., copy=False)` 减少内存复制
- 就地类型转换 `df.astype(dtype, copy=False)`

---

## 数学层面优化

### 1. 🔴 超参数搜索空间优化（高优先级）

**文件**：`core/search_space.py`

**问题**：
- `Parameter.sample()` 对 log scale 使用 `math.exp(rng.uniform(log_lo, log_hi))`
- 未利用对数均匀分布的数学性质，采样效率低
- `build_candidates()` 生成候选值时未考虑条件参数依赖

**优化方案**：
- 使用 `np.logspace` 替代手动 log-exp 转换
- 添加条件参数的联合采样策略
- 对离散参数使用 Sobol 序列替代随机采样，提高空间覆盖

### 2. 🟡 集成权重优化（中优先级）

**文件**：`core/modeling_engine.py` - `_compute_weights()`

**问题**：
- 当前使用简单归一化 CV 分数作为权重
- 未考虑模型间的相关性（ diversities ）
- 回归任务使用 `1.0 / (rmse + 1e-6)` 过于粗糙

**优化方案**：
- 添加负相关惩罚：权重与模型间预测相关性成反比
- 使用堆叠（stacking）元学习器作为默认集成策略
- 回归任务使用 `1.0 / (rmse + epsilon)` 的平滑版本

### 3. 🟡 核矩阵优化（中优先级）

**文件**：`core/modeling_engine.py` - `_create_kernel_approximation()`

**问题**：
- 核近似使用随机傅里叶特征或 Nystrom 方法
- 未根据数据维度自适应选择 n_components

**优化方案**：
- 根据有效秩（effective rank）自适应选择组件数
- 使用随机 SVD 加速核矩阵构建

---

## 算法层面优化

### 1. 🔴 早停策略增强（高优先级）

**文件**：`core/smart_early_stopper.py`

**优化方向**：
- 添加概率性早停（基于贝叶斯优化历史）
- 支持多目标早停（精度+速度权衡）

### 2. 🟡 交叉验证策略改进（中优先级）

**文件**：`core/modeling_engine.py` - `CrossValidator`

**优化方向**：
- 对时间序列数据使用 TimeSeriesSplit
- 对类别不平衡数据使用 StratifiedKFold 的改进版
- 添加重复交叉验证（Repeated K-Fold）支持

### 3. 🟡 特征选择数学优化（中优先级）

**文件**：`core/modeling_engine.py` - `AutoFeatureSelector`

**优化方向**：
- 互信息计算使用 k-NN 估计替代直方图法
- 添加特征间的条件互信息分析
- PCA 使用随机 SVD 加速

---

## 优化实施计划

1. **第一阶段**：向量化优化（直接影响性能）
2. **第二阶段**：数值稳定性（减少异常）
3. **第三阶段**：数学算法改进（提升精度）
4. **第四阶段**：内存优化（支持更大规模数据）

---

## 具体改进文件

1. `core/modeling_engine_optimized.py` - 核心引擎优化版
2. `core/search_space_optimized.py` - 搜索空间数学优化
3. `core/evaluation_engine_optimized.py` - 评估引擎优化
4. `utils/vectorization_utils.py` - 向量化工具集

*详细改进见具体代码文件*
