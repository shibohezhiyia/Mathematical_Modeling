# 数值 worker 的 OS 隔离

默认 `python_audit` 使用 Python 审计、进程和资源限制，**不是 OS 安全边界**。
可选 `SolverLimits(isolation_mode="strict_os")` 使用容器；配置、创建或启动失败时拒绝执行，
不回退到宿主 worker。本页描述代码契约，不是已完成的跨平台安全认证。

## 镜像与运行条件

从项目根目录构建专用 Linux worker 镜像：

```console
docker build -f deploy/solver/Dockerfile -t mathmodel-worker:local .
docker image inspect --format "{{.Id}}" mathmodel-worker:local
```

构建阶段允许下载基础镜像和数值依赖；构建完成后执行阶段只接受固定镜像摘要，
不会因请求参数自动拉取或替换镜像。
Dockerfile 专用忽略文件只允许核心 Python 源码、工具源码和构建文件进入上下文，
不传入 `.git`、环境文件、密钥配置、数据和研究缓存。仍应审查源码是否硬编码密钥。
源码修改后必须重建镜像；宿主源码不实时挂载到容器。

运行配置由可信部署者设置，不由题目、API 请求或模型生成：

- `MATHMODEL_OS_SANDBOX_RUNTIME`：`docker` 或 `podman`。
- `MATHMODEL_OS_SANDBOX_IMAGE`：上述实际镜像 ID `sha256:…`，或 `仓库@sha256:…`；不接受可变标签。
- `MATHMODEL_OS_SANDBOX_ATTESTATION_FILE`：受保护的绝对路径 JSON。
- `MATHMODEL_OS_SANDBOX_ATTESTED=1`：部署者已核验配置。

证明文件使用 `mathmodel.os-attestation/v2`，必须包含下列字段。
下面的摘要只是格式占位，必须换成实测镜像摘要，不能直接作为验收材料：

```json
{
  "schema_version": "mathmodel.os-attestation/v2",
  "runtime": "docker",
  "image": "sha256:<实际的64位小写十六进制摘要>",
  "network_none": true,
  "read_only_root": true,
  "non_root": true,
  "pids_limit": 64,
  "no_host_mounts": true
}
```

旧版 v1 的宿主挂载/PID 配置不再接受。JSON 是部署声明，不是密码学证明，也不能证明
守护进程确实执行了所有限制。CLI 探测到 Docker/Podman 仅表示命令存在，不表示服务可用。
`unshare` 没有独立实现的执行后端，不会开启严格模式。

## 执行与回收

每次请求先检查镜像没有声明额外 `VOLUME`，然后以随机唯一名称 `create`，再用
`start --attach --interactive` 传入有界 JSON。选项全部位于镜像名之前，默认镜像入口被显式替换。

容器使用 UID/GID `65532:65532`、禁网、只读根、丢弃全部 capabilities、
`no-new-privileges`、禁用 healthcheck；内存及 swap 总额相同、CPU 配额为 1、PID 上限 64。
PID 预算允许数值库线程，**不表示 OS 禁止所有子进程**；Python 审计仍禁止用户派生子进程。
仅 `/tmp` 提供按 `disk_bytes` 限定的 tmpfs，禁用 IPC 共享内存，禁用容器日志驱动，
不挂载宿主目录或 Docker socket。该配额不等于守护进程全局磁盘配额。

容器通过 `env -i` 获取最小数值运行环境，BLAS 线程为 1，不继承宿主 API key。
Docker CLI 在宿主只接收白名单上下文/TLS/系统环境；CLI 配置不作为容器环境传入。
Windows 原生 Job Object 不套在 Docker 客户端上，容器内存元数据标为 `container_cgroup`。

成功、超时、取消、创建/启动失败和异常路径均尝试按该请求唯一名称执行 `rm --force --volumes`。
删除返回失败时查询同名容器确认是否已经不存在；服务不可达不能冒充删除成功。
无法确认回收时记录 `container_cleanup=unverified`，成功数值也以
`container_cleanup_failed` 拒绝发布；原本失败的请求保留原错误并附回收状态。
同一服务进程随即阻止新严格任务（`container_recovery_required`），防止重复重试积累未知容器。
部署者确认指定任务容器已回收后才能重启服务恢复；重启本身不证明容器已消失。
该暂停机制仅限当前进程，不能替代多服务统一回收治理。
不会运行全局 prune，也不会清除其他任务容器。回收保留独立的有界时间，可能超过数值预算。
镜像检查和创建阶段也轮询取消信号；创建中断后守护进程可能稍后完成创建，此时仅查询到
当前无同名容器不能证明请求已撤销，会保留 `unverified`，由部署侧检查该任务名称。

## 验证范围

```console
python -m pytest tests/test_os_sandbox.py tests/test_solver_runtime.py -q
```

这些测试检查命令构造、准入、真实探针子进程的数据传入、资源/异常路径和模拟容器生命周期。
**模拟守护进程测试不能替代真实容器验收**。专用镜像的构建命令也不是构建成功的证据。

已有真实容器配置与数值执行冒烟测试。完成上面的环境配置后，在 PowerShell 中显式启用：

```powershell
$env:MATHMODEL_TEST_REAL_SANDBOX = "1"
python -m pytest tests/test_os_sandbox_integration.py -q
```

未启用时报告跳过，不冒充通过；启用后服务、镜像或配置缺失会失败，不以跳过掩盖。
这两项冒烟测试仍不是下面全部攻击场景的验收。

Linux 上可直接运行仓库脚本完成构建、摘要记录、v2 attestation 生成和三组测试：

```bash
MATHMODEL_SANDBOX_EVIDENCE_DIR="$PWD/data/runtime/linux-sandbox" \
  bash scripts/verify_linux_sandbox.sh
```

脚本不会上传数据或密钥；生成的 `data/runtime/linux-sandbox/` 只作为本地验收材料，
不要提交到 Git。Podman 用户可设置 `MATHMODEL_OS_SANDBOX_RUNTIME=podman`。

不想在本机安装 Linux 时，直接在 GitHub 仓库打开 **Actions → Linux sandbox verification → Run workflow**。
工作流使用 GitHub 托管的 Ubuntu 24.04 Runner，在云端构建 Docker 镜像并上传 14 天有效的
`linux-sandbox-evidence-<commit>` 工件。工作流只读取仓库内容，不需要 API key；仓库设置为公开后，
其他人也可以查看运行日志和下载验收材料。

部署验收还需在实际 Docker/Podman Linux 环境记录镜像摘要、运行时版本和容器 inspect，
测试禁止宿主文件读取、网络访问、根目录写入，验证内存/PID/tmpfs 限制及取消后的容器消失。
服务进程被强杀或宿主断电后的孤儿容器治理、守护进程访问控制、内核漏洞隔离、
跨主机配额及 Windows 原生受限令牌/ACL 不由本模块保证。
此 worker 只执行 `SUPPORTED_EXECUTORS` 白名单，不等于所有训练后端已被容器化。

动态编译、主研究和模型训练任务均由 `core/process_execution.py` 使用 spawn 进程运行，超时/取消会硬终止
该进程；任务账本默认位于 `data/runtime/tasks.sqlite3`，可通过 `MATHMODEL_TASK_STORE_PATH` 指定受控目录，服务重启时仅将超过恢复租约的未完成任务标记为
`interrupted`，不会伪造结果。SQLite WAL 与原子准入适合单机多进程；多主机部署仍应替换为具备租约、心跳和故障转移的共享任务服务，不能直接把 SQLite 放在无锁语义的网络盘。

生产模式下异步/同步主研究也通过 `core/research_process.py` 进入同一 spawn 服务：上传表格会先
压缩成有界 JSON 快照，worker 自行重建 DataFrame 并调用研究助手；状态从任务账本读取，取消
会终止该研究进程。模型训练通过 `core/training_process.py` 保存 joblib 结果工件后由父进程加载。
测试模式为保证替身可控仍保留线程路径。该机制是单机进程级监督，不等同于容器/内核级隔离。

选项及生命周期依据：[Docker create](https://docs.docker.com/reference/cli/docker/container/create/)、
[Docker start](https://docs.docker.com/reference/cli/docker/container/start/)、
[资源约束](https://docs.docker.com/engine/containers/resource_constraints/)。
