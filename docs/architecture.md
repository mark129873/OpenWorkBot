# architecture.md 架构与技术设计

本文第 1 节描述现有实现；第 2～10 节为主聊天与多助手改造草案，尚未实施，须经用户整体审批后开发。

## 1. 当前基线

React / TypeScript → FastAPI 会话接口 → 主 Agent Loop → 权限与工具 → SQLite → SSE 或历史回放。主循环默认每轮最多 40 个模型步骤；现有子助手在父轮内串行执行、最多 30 步，内部历史未独立持久化。当前没有本设计的后台多任务调度和重启续跑。

主要入口：`frontend/src/App.tsx`、`backend/app/api/sessions.py`、`agent/loop.py`、`agent/subtask.py`、`sessions/recorder.py` 和 `sessions/replay.py`。Agent 与 sessions 路径相对 `backend/app/`。

后端使用 Python 3.12、FastAPI、Anthropic 兼容客户端、SQLAlchemy、SQLite；前端使用 React、Vite、SSE。Skills、Hooks、todo 和现有 build / edit / yolo 权限继续沿用。根目录 `uv run --no-project start.py` 启动前后端（8100 / 5173）；`/api/health` 不依赖模型配置。当前无登录或多租户，不能直接暴露公网。

## 2. 目标与产品交互（待审批）

- 进入直接显示唯一主聊天，无会话侧栏。沿用左右消息、Markdown、空聊天引导、助手名称、用量和权限模式；移除任务详情、工具过程、复制、重新生成及停止按钮。
- 主助手负责持续对话、简单问答、判断、协调和分配；调查、业务分析、编写、文件和命令等实际任务交给 Worker。关键进展、结果及确认回到主聊天并标明任务；默认隐藏 Worker。回复默认简短，用户要求时可展开，不设固定字数。
- 用户可随时发新消息，追问、修改、停止或提出其他任务。新消息仅打断主助手当前回复，已输出文字保存并标记；Worker 继续运行。后台报告等新消息处理后，按实际发送顺序追加。
- 主对话与各任务上下文独立保存和压缩，不将所有 Worker 工作细节塞进主对话。
- 只使用单个 FastAPI 进程、`asyncio`、现有模型、SQLite 与 SSE。最多三个 Worker 执行槽位；不引入 Redis、Celery、LangGraph、任务表、Kanban 或长期记忆模块。现有 todo 不负责调度。

## 3. 目标架构与代码结构

    主聊天 ── HTTP / SSE ── FastAPI
                              │ lifespan / app.state
                           Runtime
             ┌────────────────┼──────────────────┐
          主助手 Loop       调度器 / MessageBus    Worker Loop × 0..3
             └────────────────┴──────────────────┘
                              │
                      SQLite：聊天、任务及上下文

`Runtime` 属于 FastAPI `lifespan`，由 `app.state` 持有主循环及各 Worker 的 `asyncio.Task` 强引用；SSE 连接只订阅事件，不拥有执行任务。主助手使用单个消费循环，用户输入优先于后台报告；Worker 独立执行。同一数据目录只允许一个后端进程：启动时获取进程锁，已有进程持锁则明确报错，退出或异常时释放锁。

| 拟改动 | 职责 |
| --- | --- |
| `backend/app/main.py` | 启动数据库和 Runtime、恢复状态；关闭时收口任务。 |
| `backend/app/api/chat.py`（新） | 主聊天快照、消息、审批和 SSE；调用 Runtime，替换旧会话入口。 |
| `backend/app/agent/runtime.py`（新） | 任务、依赖、三个槽位、调度、停止信号及后台任务监督。 |
| `backend/app/agent/agent_bus.py`（新） | 内部消息、持久化核对和重试、等待及等待环检测。 |
| `backend/app/agent/task_tools.py`（新） | 主 / 子助手任务与通信工具，调用方身份由运行时注入。 |
| `backend/app/agent/loop.py`、`subtask.py`、`system_prompt.md` | 主助手对话和判断；独立 Worker 循环与角色提示词。主助手不直接执行业务工具。 |
| `backend/app/agent/llm.py`、`compact.py`、`tools.py` | 结构化路由、独立上下文压缩、文件分页及命令输出限制。 |
| `backend/app/sessions/store.py`、`replay.py`、`recorder.py`、`approvals.py`、`session_permissions.py` | 短事务、上下文投影、记录及多任务审批；保持现有授权范围。 |
| `backend/app/models.py`、`agent/model_call.py`、`modelio.py` | 上下文及未处理消息查询索引；并发模型日志与轮转串行化，关联任务 / 运行 ID。 |
| `frontend/src/api/client.ts`、`hooks/useSessionStream.ts`、`App.tsx` 及组件 | 接入主聊天、按消息 ID / 版本更新、关联任务审批。 |

现有 `turn_manager.py` 的单会话单轮和 `subtask.py` 的父轮同步等待不再用于后台任务调度。Runtime 捕获 Worker 循环异常并持久化失败通知；主循环异常写入错误并恢复后续消息处理，不能把异常当成功，也不能自动重跑结果未知的外部工具。模型快照写失败仅告警；核心状态未写入时不能假报已受理。

## 4. 主聊天接口与事件

| 接口 | 契约 |
| --- | --- |
| `GET /api/chat` | 返回 `{revision, messages, approvals, permission_mode, context_usage}`。 |
| `POST /api/chat/messages` | 接收 `{client_message_id, text, permission_mode?}`；非空校验，保存后返回 202，不等待模型。ID、文本和权限模式相同的重试幂等，复用 ID 且参数不同时返回 409。 |
| `GET /api/chat/events` | SSE 先推送一致快照，再推送 `{revision, type, data}`，事件包括消息、审批和用量。 |
| `POST /api/chat/approvals/{request_id}` | 接收现有 `option_id`、可选拒绝理由；落盘后按现有权限规则处理，不等待主助手判断。失效请求或不同决定重复提交返回冲突。 |

保留 `/api/permission/state`。订阅注册、取快照与事件发布由同一写锁及持久递增版本协调；每个连接独立队列，慢连接溢出时断开并重新取快照。流式文本分批先落库再广播；新输入推进回复代号，旧流迟到片段不得覆盖新回复。已接收的后台任务不因页面或主回复中断而取消。

用户消息与审批选择统一分配持久递增的 `user_seq`。同一页面按发送 / 点击顺序提交，无须等待模型；多页面并发以服务端落盘顺序为准。模型判断携带来源序号，不能让较早消息的迟到判断覆盖同一操作较新的用户决定。

## 5. 调度与助手通信

- 每个任务有稳定 `task_id`、创建顺序、目标及独立上下文；一次执行有 `run_id`。执行槽位可复用，但上下文不能串用；同一上下文同时只有一个模型调用。
- 按用户最初交代顺序领取，续跑沿用原序。旧任务进入 `reviewing` 等主助手判断时，较晚任务先等待；已运行任务不抢占。用户任务优先于额外协助，无空槽时协助请求立即返回未分配。
- Worker 一轮最多 30 个模型步骤，每步模型调用前保存步数；第 30 步正常结束可以成功，需要继续时交主助手判断。主助手可立即新开一轮，重复且无进展时可停止。暂停同一轮保留 `run_id` 与已用步数；新一轮才重置。主助手默认单轮上限沿用现有 40 步。
- 主助手确认 A 依赖较晚的 B 时，记录依赖并可先安排 B；A 在模型步或工具结束的安全边界保存上下文、让出槽位，B 完成后主助手判断并恢复 A。B 的领取和待判断顺序继承最早等待任务的序号；B 已运行或等待用户时不重复领取。B 失败或停止通知主助手；已停止的 A 不恢复。等待用户审批的执行单元仍占槽位。
- 根任务可处于 `queued / running / waiting_user / waiting_dependency / reviewing / completed / failed / stopped`；负责人和每个协助者另存运行状态。用户审批与任务依赖可同时存在；完成任务前须收口相关协助。
- 文件与命令工具经权限检查后使用 `asyncio.to_thread`，不阻塞聊天；线程不共享数据库 Session 或直接操作 `asyncio.Queue`。命令合并 stdout / stderr 并持续排空，读取和等待均有界。

模型工具为草案：主助手使用 `task_create / task_inspect / task_update / task_stop / task_review / task_wait_for / route_input`；Worker 使用 `agent_list / agent_send / agent_wait / agent_help / task_report`。主助手和所属任务可用 `read_tool_output` 查看已保存输出。工具返回“已受理”不等于执行完毕。

Worker 可直接给忙碌同伴发信息，对方在当前模型步或工具结束后处理；请求额外工作只能分配空槽。停止或修改同时送负责人及相关协助者。等待回复登记前后均检查已保存回复，等待期间监听停止、变更和依赖暂停，避免漏唤醒；被打断的工具调用补齐结果。

已知等待关系持久化为有向边，例如 `task-A → task-B → ctx-B → task-A` 表示 A 等 B、B 的执行者等 A 占用的文件。边变化时检测环并通知主助手判断，不自动终止；未声明的依赖与 PowerShell 内部等待无法完整识别。

## 6. 内部消息、状态与上下文

使用固定内部主会话 `Session.id = "main"`，复用现有 `Message / Part`，不新增表。隐藏控制 Message 存储以下 Part，用户消息和各上下文的模型消息各自独立：

开发阶段不做数据库迁移，旧多会话历史不迁移；测试仅使用独立空库。不得自动清空 `backend/data/`，其中可能包含工作区和日志。

| 记录类型 | 最小内容 |
| --- | --- |
| `task_state` | 任务 ID、目标、标题、创建序号、状态、负责人、依赖、停止意图及文件占用。 |
| `agent_context / agent_run` | 上下文 ID、任务 / 角色、原始模型消息、摘要边界、运行 ID、步数和运行状态。 |
| `agent_message` | 消息 ID、来源 / 目标、任务 / 上下文 / 运行 ID、类型、内容、关联请求、投递状态和重试信息。 |
| `approval / user_event / file_wait` | 审批请求及参数快照；用户消息 / 审批决定、先后序号和操作身份；文件路径、占用者和等待关系。 |

运行时锁和短 SQLite 事务串行化状态变更；每次事务独立创建 Session，不持锁等待模型、工具或用户。任务创建与工具受理结果、状态与待投递消息、目标上下文写入与投递标记分别原子提交。模型不得自报发送方或伪造授权；内部操作按上下文、运行及 `tool_call_id` 去重，参数不同的同 ID 重试报冲突。

消息先入库再通过 `asyncio.Queue` 唤醒：`pending → delivered → processed`。消息 ID 对目标上下文稳定；启动、空闲 / 中断时及定时按索引核对未处理记录，投递失败按有界退避重试。已投递但未处理的报告恢复时不重复插入上下文；只重试内部消息和报告，不重跑外部文件或命令。迟到的旧 `run_id` 结果不能覆盖新运行；修改已受理但在结束前未送达时保留记录并通知主助手判断是否续跑。关键报告到主助手，普通同伴消息不自动拼进主对话。

主对话和各任务分别组装、压缩；`load_history` 按上下文取模型历史，`load_replay` 只投影用户可见内容。原始聊天、助手消息及已采集工具结果保留；摘要 / 裁剪只作用于模型输入副本，摘要失败不推进边界，工具调用与结果保持配对。主回复被打断时，丢弃未完整解析的调用，保留已完成结果或补入中断结果。

- `read_file` 参考 Hermes：按行分页，默认第 1 行、最多 2,000 行 / 10 万字符，按整行给后续行号。单行超过 2,000 字符则截断，该行余下部分不能按行号续读；原文件不变，此限制已确认。
- 命令输出流式采集最多 1 MiB，超出保留首尾并标记，保持字符编码完整，继续排空进程输出；已采集内容随任务保存，不设置 7 天清理。每次向模型返回最多 3 万字符，先给首尾预览，可按输出 ID 分段读取已保存部分；超出 1 MiB 的中段不可恢复。

## 7. 授权、用户修改与文件冲突

沿用现有权限优先级：硬拒 → yolo 模式 → deny → ask → allow → edit / build 默认规则。`allowSession` 只覆盖 `ask`，不能覆盖已判定的硬拒或 deny；授权仍存内存，在主会话内由所有任务共享，后端重启后失效。文件按工具和真实路径匹配，命令按完整参数精确匹配；`read_file` 新增的分页参数不改变同一文件原有的会话授权范围。

主助手判断用户消息期间，已有审批继续展示、可点击，新审批也照常提出；已获授权的操作继续执行。消息和审批按用户操作顺序记录，对同一有效操作以较新的决定为准。例如先说“不要删除 A”，判断未结束时又批准同一删除，则可执行删除，较早消息的迟到判断不能覆盖该批准；顺序相反时，已经开始的删除可能无法撤回。已结束任务、过期请求或参数变化仍按现有规则拒绝旧审批。

主助手判断用户消息时，结构化关联来源 `user_seq`、任务、受影响操作和意图；Runtime 校验后对同一操作比较较新的用户决定，过时的冲突意图不执行，其他不冲突要求仍处理。判断期间若有较新消息或审批，而冲突关系尚不明确，须结合新事件重新判断或询问用户，不能当作无冲突；审批不暂停。修改 / 停止意图及其去重记录原子保存。

`route_input` 草案包含 `{message_id, user_seq, intents: [{type, task_id, instruction, approval_ids?}], needs_clarification}`，意图类型为 `chat / create / query / update / stop / unclear`。使用指定工具调用并校验结果；接口错误或结果无效时报告错误，不执行不确定的意图，审批仍可按现有规则处理。首版不增加 JSON 备用分支。

首版 Worker 共享现有 `data_dir/workspace`。文件工具沿用真实路径限制；`write_file / edit_file / delete_file` 以规范化路径协调：不同任务等待首次写入该文件的任务结束；同任务协助者的写操作逐次串行。等待过或读取后被其他助手改动的文件，拒绝旧写入并要求重读；重新调用重新检查权限。暂停让出槽位不释放文件占用；PowerShell 仅设置工作目录，不是系统沙箱，其内部修改范围无法可靠识别。

## 8. 后台运行与恢复

- 消息已保存并返回 202 后，关闭网页只断开订阅；电脑、后端进程及所需网络正常时 Worker 继续，重开页面获取快照。等待审批的工作等用户处理。电脑休眠、关机或后端进程退出会中断，不能在进程停止期间继续。
- 启动先恢复用户消息 / 审批决定的顺序、任务 / 上下文及未处理通知，再提供快照并异步恢复工作；页面启动不等待模型。缺少模型配置时仍可打开聊天，未完成任务保留并报告配置问题。等待中的旧审批以新 ID 发起，旧 ID 失效；操作以 `task_id + run_id + tool_call_id` 关联历史顺序，已批准但未开始的操作重新检查权限。
- 恢复排队、运行、等待及待判断任务；已完成、失败或用户已请求停止的任务不自动重跑。已完成但未转达的结果继续投递。关闭服务与用户停止任务分别记录。
- 工具已开始但结果未提交时，补入“结果未知”供 Worker 检查，不自动重跑，也不视为成功；任意外部命令不保证恰好执行一次。已开始的 PowerShell 命令按现有协作式停止规则收口。

## 9. 验收与未决风险

开发前保持草案状态；实现后使用假模型、独立临时工作区与空 SQLite 验证：

1. 进入主聊天、持续追问及新任务并行；三个槽位上限、30 步续跑 / 停止、原序调度及依赖让位。
2. 跨 Worker 通信、忙碌边界投递、已知等待环、文件竞争、审批共享与按任务关联；连续消息及消息 / 审批的先后顺序均以最新有效决定为准，不丢弃无关要求；不同分页沿用同一文件授权。测试指定工具调用的无效输入及 API 错误。
3. 消息落盘后通知失败、主回复被打断、去重及进程内重试；模拟后台重启、未知工具结果、审批重发和任务创建幂等。
4. 启动本地 HTTP / SSE，接收任务后关闭页面，验证后台完成、重连快照和结果；验证同一数据目录不能启动两个后端，以及快照 / 增量交界断线、慢连接和日志失败。
5. 检查上下文隔离与工具调用配对，文件分页、超长行提示、1 MiB 命令采集和分段读取；执行前端构建、检查和浏览器验收。

本机能力检查：SQLite 3.53.1 支持 `json_extract`；当前 DeepSeek 官方 `deepseek-flash` 在非思考模式下，单次无副作用请求成功强制调用指定工具，本项目客户端尚未暴露该参数。GPT-6 官方 Responses API 文档支持指定工具；原生接入需后续适配，尚未实测。

真实模型的任务归属、结果判断与重复检测尚未充分验证。DeepSeek 的 Anthropic 兼容接口在思考模式下不支持强制指定工具；当前代码关闭思考模式，将来启用时需重新处理。模型服务不可用时，聊天中的停止 / 修改无法及时判断，已授权的后台操作仍按约定继续；旧任务反复续跑或等待判断可能让较晚任务久等。命令输出及超长单行有已确认的截断限制，持久记录长期累积。现有 `backend/tests/test_reliability.py` 有 10 项测试；此前基线曾通过，本次设计整理未运行测试，不能作为新方案验收证据。日志及干净库要求见 [reliability.md](reliability.md)。

## 10. 参考

- Hermes `9a76ac8e4f`：`tools/async_delegation.py`、`delegate_tool_registry.py`、`file_tools.py`、`file_operations.py`，参考后台委派、安全边界及文件分页；不照搬 Kanban。
- Codex `e72da2b538`：`codex-rs/core/src/unified_exec/`，参考命令输出采集。
- [DeepSeek `6337fc823ce`](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/effects.ts)：参考 Router / Worker 分工、待处理记录和失败重试；不照搬其授权或额外槽位。
- [Anthropic 工具选择](https://platform.claude.com/docs/en/api/http/messages)：指定 `tool_choice` 可要求调用特定工具；[部分思考模式存在限制](https://platform.claude.com/docs/en/build-with-claude/thinking)，当前客户端默认关闭该模式。
- [DeepSeek 接口](https://api-docs.deepseek.com/guides/anthropic_api/)与 [OpenAI GPT-6 工具调用](https://developers.openai.com/api/docs/guides/function-calling)：前者非思考模式支持指定工具，后者通过 Responses API 接入。
