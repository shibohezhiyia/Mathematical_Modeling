# 受控数值执行

已有的通用 ODE (`adaptive_ode/v1`)、有界非线性优化 (`bounded_nlp/v1`) 与凸二次规划 (`quadratic_program/v1`) 现在默认在独立进程执行。不需要安装新依赖，也不需要开启模型 API；经过原有来源与契约验证的任务会自动使用这条链路。

这不是任意 Python 的安全沙箱。它只处理可信内置求解器的 JSON 数学契约，不接受程序、模块路径或可执行对象。N03–N04 的候选假设图仍不能因此自动获得执行权限。

N07 新增显式绑定的标量图执行入口 `scalar_graph/v1`，使用同一资源监督，但不经过题型模板。它只解释白名单原语，不执行生成代码；假设必须另行绑定实验，不能仅凭类型通过就开始研究。使用方法与限制见 [数学图搜索](graph_search.md)。

`quadratic_program/v1` 复用注册表中已核验的凸二次规划契约（线性目标项、对称半正定二次矩阵、线性等式/不等式和有限边界），worker 不接收回调或代码。多起点一致性、可行性和半正定性仍由求解器结果审计；执行成功不自动等于现实约束完整或全局最优已被证明。

冻结留出入口 `scalar_graph_confirm/v1` 也使用同一监督：结构、参数和阈值已经固定，不再拟合。默认 30 秒/1,000 次向量评估，包含数值异常后的逐点定位；调用方另行负责持久的样本使用登记。worker 不单独认证数据新颖性。

## 当前限制

| 项目 | 已实现行为 |
| --- | --- |
| 强制墙钟时限 | ODE 默认 30 秒，非线性优化默认 45 秒；包含排队、启动、导入、求解及响应读取；超时后终止进程并清理 |
| 内存 | 默认每个 worker 1024 MiB；限制建立失败时拒绝执行，不回退到主进程 |
| 数值评估 | ODE 最多 250,000 次 RHS 向量调用，包含容差复算；非线性优化最多 50,000 次目标/约束调用，包含可行性检查 |
| 标量图 | 通过显式实验入口调用；搜索默认单候选 500 次向量评估，包含拟合差分与反例重放；单次 worker 上限 30 秒并受会话剩余时间约束 |
| 并发 | 每个服务进程最多 2 个此类 worker；请求 BLAS 单线程。部署多个服务进程时可显式传入 `SharedResourceQuota`，用 SQLite 租约原子限制跨进程内存/槽位；默认不启用，不把本机路径当作集群隔离 |
| 输入/输出 | 仅普通 JSON，输入最多 1 MiB，结果最多 16 MiB，stderr 最多 64 KiB；超量即失败，不截断后当成完整答案 |
| 表达式 | 使用 `core.safe_expression.SafeNumericExpression` 的受限 AST 语言，无 `eval`；最多 4096 字符、256 个 AST 节点、32 层深度；限制函数参数个数 |
| 数值异常 | 除零、实数定义域错误、溢出、非有限中间值不能成为有效数值结果 |
| 生命周期 | 每个节点新建临时目录；完成、失败或取消后终止 worker 并删除本次目录；不写入仓库缓存 |
| 环境 | 不继承 API 密钥、`PYTHONPATH` 等环境变量；仅保留运行必需项，不将 stderr、原始堆栈或本地路径写入结果 |
| 文件/网络/子进程 | worker 安装 Python audit hook：路径只允许临时目录、可信项目/解释器目录，网络和子进程默认拒绝；结果元数据记录 `permission_isolated=true` |

墙钟检查由父进程执行，不依赖 SciPy 是否及时调用 Python 回调。进程创建、操作系统调度及终止回收本身仍有开销，不能解释为毫秒级实时保证；达到时限后不能保留迟到的候选结果。

独立进程会增加启动开销，当前没有声称提速。它首先解决已接入执行入口无法强制中断的问题；标量图有候选内部的有界中间量复用，跨候选复用与安全 worker 池仍待 N08。

## 操作系统边界

- Windows：使用 Job Object 限制提交内存，最多一个活动进程，关闭 job 时终止其进程。先挂接 job，再通过 stdin 释放契约；worker 在接收契约前不加载数值后端。内存限制是提交内存上限，不是任务管理器中的 RSS 数值。[Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)、[内存限制结构](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)
- Linux：worker 在数值库导入前设置 `RLIMIT_AS` 和禁止 core dump；父进程通过独立进程组终止任务。限制的是虚拟地址空间，不等同于 RSS，也没有提供 cgroup 级进程树总内存或进程数量限制。[Python resource 文档](https://docs.python.org/3/library/resource.html)
- 其他平台（包括当前 macOS 路径）：这些受控后端返回 `isolation_unavailable`，不会宣称已获得未实现的限制。其他既有功能不因此整体禁用。

Python audit hook 是防御性权限边界，不是容器、Windows restricted token 或 Linux namespace。它限制受审计的路径、网络和子进程事件；可信安装目录、Python 环境和数值依赖仍在信任边界内，恶意依赖、原生库漏洞和操作系统级绕过不在此保证内。磁盘检查只覆盖本次 worker 临时目录；跨服务内存/槽位配额只有在显式注入 `SharedResourceQuota` 时才生效，磁盘总额和容器级隔离仍未完成。

`SafeNumericExpression` 只规定“哪些数学语法可以被解释”，不是任意生成代码的安全沙箱。它不接受属性访问、下标、导入、字符串、容器、赋值或关键字参数，并只绑定普通有限实数。需要执行候选程序时仍必须走 `SolverProcessRunner`；没有操作系统隔离能力时，系统应拒绝开放式代码执行，而不是把 AST 通过当作安全证明。

现有其他求解器、模型训练、积分动力学发现及静态编译阶段没有全部迁入这条进程链路；报告仍区分强制时限和软预算，不把已接入后端外推为全项目沙箱。

## 失败如何解释

| 错误码 | 含义 |
| --- | --- |
| `timeout` / `queue_timeout` | 本次时间或资源等待预算耗尽，不证明模型错误 |
| `memory_limit` | worker 明确报告内存申请失败；无法确认原因的原生异常退出仍使用 `worker_exit` |
| `evaluation_limit` | 调用次数耗尽，不降低检查门槛后重试 |
| `numeric_domain` | 出现除零、开方/对数定义域或浮点溢出等问题；还不是带适用域证书的数学反例 |
| `invalid_contract` | worker 重新核验后拒绝契约，忽略调用方自报的 `machine_verified` |
| `upstream_failed` | 上游没有可用结果，下游没有使用占位数值启动 |
| `cancelled` | 核心执行接口收到取消信号，不形成数值结论 |
| `isolation_unavailable` | 无法安装必需的资源限制，本次不执行 |
| `permission_isolation_unavailable` | 无法安装 worker 权限审计，本次不执行 |
| `output_limit` / `worker_exit` | 输出超量或进程异常退出，不采纳半截结果 |

这些错误保留 `mathematical_verdict: not_assessed`。网页与报告显示原因及建议，机器可读执行证据保留错误码与运行监督元数据。成功 worker 返回后仍进入原有独立审计；非凸优化不会因为执行成功而升级成全局最优证明。

## 开发接口与测试

```python
from threading import Event
from core.solver_runtime import SolverLimits, SolverProcessRunner
from core.shared_resource_quota import SharedResourceQuota, SharedQuotaLimits

cancel = Event()
quota = SharedResourceQuota("/var/tmp/mathmodel-quota.sqlite3",
                            limits=SharedQuotaLimits(max_memory_mb=4096, max_slots=8))
# contract 必须是完整、已核验的 ode_system 契约；worker 会再次检查。
result = SolverProcessRunner().execute(
    "adaptive_ode/v1", contract,
    limits=SolverLimits(wall_seconds=30, memory_mb=1024, max_evaluations=250_000),
    cancel=cancel,
    shared_quota=quota,  # 可选；不传则只使用本服务进程内配额
)
```

取消信号是核心接口能力；研究助手异步入口已接入 `/api/research/cancel`，采用合作式检查并在提交结果前丢弃取消后的结果。它不强杀正在运行的 Python 线程，也不替代主服务多任务压力认证。时限、内存、临时目录磁盘配额和调用次数由服务端设置，模型响应不能修改它们。磁盘检查只覆盖本次 worker 临时目录，不等于全盘配额或操作系统容器隔离。

```bash
python -m pytest tests/test_solver_runtime.py tests/test_mechanistic_modeling.py tests/test_universal_math_solvers.py -q
```

测试夹具只通过测试代码替换私有启动函数制造挂起、超量输出和内存压力，不能通过求解器键或 JSON 请求调用。压力测试本身有上限；不会主动占满主机内存。

当前实测平台为 Windows / Python 3.12。Linux 分支仍需目标环境复验；当前 SciPy 1.17.1 超出仓库声明的 `<1.16.0` 范围，不能将本机通过视作支持版本矩阵认证。
