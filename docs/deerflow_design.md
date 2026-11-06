# DeerFlow 改造技术设计（草案）

状态：总体方向已确认；本文件中标注待批准的接入细节仍需用户审核。依据本地 DeerFlow `2a16b26960cb953cd3283702d018bd7dce6b80b7` 静态分析，尚未运行原型。

## 1. 已确认

- 以 DeerFlow Gateway 为主体，移植本项目的调度；保留主 Agent、独立后台 Worker，以及 README 2.1 的 Shared State + Message Bus 协作架构。
- 以 Agent Teams 示例为基础扩展协调器；沿用 OpenWorkBot 现有前端。成功、失败或停止任务的结果送达主聊天后清理调度数据、不归档，保留聊天结果、工作区文件及不含任务内容的请求 ID 去重标记；移除每团队累计 100 个请求和共享状态 20 条上限。提示 Worker 简短回复，超过单条消息 16 KiB 时截断并提示。
- 每个用户有独立主助手；主 Agent、Worker / 协助者使用独立 LangGraph 检查点，同一上下文同时只运行一个图，同一助手的 Agent 共享所属用户的工作区。
- 多用户；应用数据、重要消息与 LangGraph 检查点使用同一个 SQLite 文件。业务状态和图状态分别负责，消息按稳定 ID 去重；已开始但结果未知的工具操作不自动重跑。
- 工具记录按用户、运行 ID 和工具调用 ID 关联；执行前保存状态，完成后保存结果，重复调用复用已保存结果。任务状态与待发送总线消息在同一应用事务保存。
- 重置时停止并重启当前用户所有 Agent / SubAgent，删除该用户旧聊天、任务、检查点、长期记忆和模型日志，全部改用新会话；不影响其他用户。每次新 Run 刷新记忆快照。
- 使用 DeerFlow `LocalSandbox`，`allow_host_bash: false`；后续切 Docker 须另行确认。现有 RAG 服务及其 MCP 接入暂不改造，切 PostgreSQL 须另行确认；其余能力优先沿用 DeerFlow。
- 未获批准的具体设计或实现先与用户同步；不理解或未验证的内容如实列出。

## 2. 已核实的 DeerFlow 机制

| 能力 | 源码现状及与目标的差异 |
| --- | --- |
| Agent 运行 | Gateway 的 `AgentRuns` 可创建独立线程、启动 / 取消运行、发送 `steer` 输入并查询输入状态。`RunInputControl` 能在运行中处理新指令，并在边界暂停 / 恢复原生子 Agent；其子任务生命周期仍与父 Run 关联。 |
| Agent Teams | 可选示例插件使用独立 Agent Run，提供按用户隔离、成员定向请求、结果回传和共享状态；默认无固定主 Agent，成员请求按会话排队，不是完整发布订阅总线。 |
| Teams 限制 | 示例使用单独的 SQLite 文件；Gateway 重启后团队须重新连接。附件捕获 / 导入要求 POSIX，不能未经验证假定适用于当前 Windows 环境。 |
| 批量任务 | 默认关闭的 `batch_task` 可持久化任务，但租约过期后可能重试；源码建议用于只读或幂等操作。 |
| 存储 | DeerFlow 的应用数据、LangGraph Store 和检查点默认共用 SQLite；`run_events.backend` 默认是 `memory`，需独立判断哪些事件必须持久化。 |
| 记忆 | DeerMem 默认 middleware 模式：按用户保存 JSON 摘要、按 Agent 保存 Markdown 事实，FTS 使用独立 SQLite 索引；默认会话首次注入记忆快照，轮次后后台提取。默认防抖 30 秒、最多 100 条事实、置信度阈值 0.7、注入预算 2,000 token。待提取队列在内存中，异常退出可能丢失未处理内容；`clear_memory(user_id, agent_name=None)` 清空该用户摘要和全部 Agent 事实。 |
| 沙箱 / 认证 | `LocalSandbox` 默认禁用宿主命令；主 Agent 与原生子 Agent 共享线程沙箱。本地邮箱 / 密码允许自助注册，OIDC 和可选细粒度授权默认关闭。 |

更正：DeerFlow 并非只能等子 Agent 结束后处理用户输入；它已有运行中 `steer`。独立后台 Worker、双向消息和重启后的自动恢复仍需适配与验证。

来源：[运行中输入](../.github_deerflow/backend/packages/harness/deerflow/runtime/runs/input_control.py)、[AgentRuns](../.github_deerflow/backend/packages/extension-api/deerflow_extension_api/agent_runs.py)、[Agent Teams](../.github_deerflow/examples/deerflow-extension-agent-teams/README.md)、[批量任务](../.github_deerflow/backend/packages/harness/deerflow/tools/builtins/batch_task_tool.py)、[记忆清除](../.github_deerflow/backend/packages/harness/deerflow/agents/memory/backends/deermem/deer_mem.py)、[配置](../.github_deerflow/config.example.yaml)。

## 3. 候选接入细节（待批准）

### P1：协调器和通信

- 以 Agent Teams 示例插件为协调器基础，复用其完整 `AgentRuns`、成员身份、定向请求和共享状态；扩展固定主 Agent、后台 Worker 独立生命周期、发布订阅和主聊天调度。
- 候选流程：用户消息进入主 Agent；主 Agent 受理任务后，协调器记录任务并为 Worker 创建独立线程，以稳定请求 ID 启动 Run；Worker 通过总线上报进度 / 结果，主 Agent 继续处理用户消息。Worker Run 不以主回复结束为结束条件。
- 主 Agent 和各 Worker 均显式读写可恢复的共享任务、进度与结果；通过 Message Bus 双向传递请求、事件和通知，支持定向传递与发布订阅。运行时负责调度、生命周期、并发和状态。
- 建议 Message Bus 在 SQLite 保存消息和每个目标的投递状态；发布时确定接收者。进程内信号只用于唤醒，不能作为唯一消息来源。消息来源、用户和目标均由服务端校验。
- 正在运行的目标可研究通过 `AgentRuns.send_input(..., mode="steer")` 接收消息；以检查点中的已应用输入确认投递，而非仅以接口受理为成功。目标结束或拒绝时由持久队列处理；是否满足现有 Worker 消息顺序与恢复要求尚未验证。
- Teams 示例使用独立数据库并要求重连；其默认最多八个活动任务，当前项目的 Worker 上限为三。统一 SQLite、自动恢复及 Worker 容量需要适配，具体实现未批准。

### P2：重置与记忆清除

- 已确认删除当前用户旧聊天、任务、检查点和模型日志，以及共享摘要和全部 Agent 事实；所有 Agent / SubAgent 改用新会话，不影响其他用户。具体接入方式待批准。
- 建议先持久化重置状态，停止该用户所有旧 Agent / SubAgent、阻止旧代新写入，处理正在进行的记忆提取，再删除旧记录及清除记忆；成功后才允许新会话运行。数据库与记忆文件无法假定同一事务，失败 / 重启时需安全重试。
- DeerMem 清除前后会取消排队项，但 `cancel_by_agent` 明确不取消已经开始的提取；按用户等待或阻止旧代写入的具体机制尚未找到可直接复用的实现。
- DeerFlow 删除线程会删除线程目录，并对检查点、运行及事件记录采取尽力清理；为满足保留工作区和确认旧记录删除，不能直接照搬该接口。

### P3：数据、身份和沙箱

- 应用表保存 Shared State、任务、审批、重要消息和工具记录；LangGraph 检查点保存 Agent 图状态。统一文件中的不同连接仍是不同事务。DeerMem 文件和 FTS 索引由其自身管理。
- Teams 示例默认独立 SQLite 且直接使用 `sqlite3`；如采用，如何接入已确认的统一文件及未来 PostgreSQL 需要专项适配。`run_events` 暂按 DeerFlow 默认的内存模式，重要总线消息单独落库；如恢复依赖修改该配置，先同步用户。
- 每个请求按服务端认证身份校验助手、线程、任务、消息、审批和文件的归属；SSE 只发送该用户数据。主线程的沙箱工作区与 Worker 检查点线程分开映射，具体接口与租约处理需验证。
- 不向 Agent 暴露宿主命令工具，包括现有 PowerShell 路径。`allow_host_bash: false` 只约束 DeerFlow 相应工具，MCP / 自定义工具需另行审查。LocalSandbox 只提供应用层边界，不提供系统级用户隔离。

## 4. 尚未理解或验证

1. Teams 的 `AgentRuns` 能力句柄是进程内、绑定用户授权的；重启后示例须用户重连。如何在不保存用户凭据、且重新校验权限的前提下自动恢复后台任务，尚未确定。
2. Worker 独立检查点与共享主工作区的映射、沙箱租约、Windows 路径及 Teams 附件机制是否兼容，尚未验证。
3. `steer`、广播和定向消息与总线确认如何衔接；检查点压缩、目标运行结束和进程崩溃时如何避免漏投 / 重投，尚未通过故障注入验证。
4. 正在执行的 DeerMem 提取如何按用户安全排空或拦截；记忆文件、FTS 和 SQLite 重置状态失败时如何恢复，尚未确定。
5. 重置时删除全部旧检查点并阻止迟到的旧运行写回，须覆盖所有 Agent 的线程；删除失败及后台回调如何安全收口尚未验证。
6. 已确定每次新 Run 刷新记忆；替换旧快照及读取失败的具体处理尚未确定。
7. 已确定结束任务在结果送达后清理；结果截断、去重标记与依赖 / 回执的具体收口方式待设计。现有模型日志按会话记录、没有用户标识，如何只清除当前用户日志待设计。
8. 以 Gateway 为主体时源码的正式目录、现有前端及模型配置如何接入，尚未确定；不会把未入库的 `.github_deerflow` 临时参考目录当作正式运行依赖。

具体候选流程、失败处理和待确认的取舍见 [deerflow_runtime_design.md](deerflow_runtime_design.md)。

## 5. 验收与实施边界

- 在独立空库验证两名用户的聊天、审批、SSE、文件、检查点和记忆隔离；关闭页面或主回复中断后，后台 Worker 保持独立运行。
- 验证忙碌 Agent 的定向消息和发布订阅、重复请求、检查点与总线之间的崩溃恢复，以及未知工具结果不重跑。
- 验证重置时排队和正在提取的旧记忆无法重现；失败 / 重启可重试，其他用户记忆不受影响。
- 验证 LocalSandbox 禁用宿主命令、共享工作区和 Windows 路径。应用表按项目规则从空库建立，不做历史数据迁移；LangGraph 初始化自己的检查点表。
- 具体方案批准后再实施。RAG 不纳入本次改造或隔离验收；当前共享的 RAG MCP 不能视为已完成多用户隔离。
