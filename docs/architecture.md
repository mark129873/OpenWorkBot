# architecture.md 架构与技术设计

本文记录主聊天、多助手与已批准的 RAG 技术设计和运行边界；实现状态见进度记录。

第 1–12 节记录改造前的历史基线；当前架构见第 13–15 节，TODO 与主助手改造见第 16 节。

## 1. 改造前基线

React / TypeScript → FastAPI 主聊天接口 → 主助手与 Runtime → 最多三个独立 Worker → 权限与工具 → SQLite / SSE。主助手每轮最多 40 步，Worker 每轮最多 30 步，任务和上下文独立持久化并支持恢复。

主要入口：`frontend/src/App.tsx`、`backend/app/api/chat.py`、`agent/runtime.py`、`agent/worker.py` 和 `sessions/chat_store.py`。旧会话代码保留供回归，不再注册旧接口。Agent 与 sessions 路径相对 `backend/app/`。

后端使用 Python 3.12、FastAPI、Anthropic 兼容或 OpenAI 客户端、SQLAlchemy、SQLite；前端使用 React、Vite、SSE。Skills、Hooks、todo 和现有 build / edit / yolo 权限继续沿用。根目录 `uv run --no-project start.py` 启动前后端（8100 / 5173）；`/api/health` 不依赖模型配置。当前无登录或多租户，不能直接暴露公网。

Skill 从 `backend/skills/<name>/SKILL.md` 发现：主助手和 Worker 的系统提示词只包含 YAML frontmatter 的 `name / description`；名称须符合规范且与目录一致，无效条目跳过。适用时用 `load_skill` 读取完整说明，引用的文本资源按需分页读取；资源路径不得越出该技能目录。脚本通过现有命令工具及权限执行，加载 Skill 不授予额外权限。

## 2. 产品交互

- 进入直接显示唯一主聊天，无会话侧栏。沿用左右消息、Markdown、空聊天引导、助手名称、用量和权限模式；移除任务详情、工具过程、复制、重新生成及停止按钮。
- 主助手判断任务规模：简单、预计可快速完成的任务直接使用工具；复杂、耗时或适合并行的任务交给 Worker。关键进展、结果及确认回到主聊天；默认隐藏 Worker。回复默认简短，用户要求时可展开，不设固定字数。
- 用户可随时发新消息，追问、修改、停止或提出其他任务。新消息仅打断主助手当前回复，已输出文字保存并标记；Worker 继续运行。后台报告等新消息处理后，按实际发送顺序追加。
- 设置中的“删除”需确认；先停止任务并归档聊天和已保存结果，再新建空白助手。保留工作区、项目规则、模型配置及原模型日志；执行模式恢复 build，临时授权清空。
- 主对话与各任务上下文独立保存和压缩，不将所有 Worker 工作细节塞进主对话。
- Bot 使用单个 FastAPI 进程、`asyncio`、现有模型、SQLite 与 SSE。最多三个 Worker 执行槽位；Bot 不引入 Redis、Celery、LangGraph、任务表、Kanban 或长期记忆模块。独立 RAG 服务可使用 LangGraph。

## 3. 改造前架构与代码结构

    主聊天 ── HTTP / SSE ── FastAPI
                              │ lifespan / app.state
                           Runtime
             ┌────────────────┼──────────────────┐
          主助手 Loop       调度器 / MessageBus    Worker Loop × 0..3
             └────────────────┴──────────────────┘
                              │
                      SQLite：聊天、任务及上下文

`Runtime` 属于 FastAPI `lifespan`，由 `app.state` 持有主循环及各 Worker 的 `asyncio.Task` 强引用；SSE 连接只订阅事件，不拥有执行任务。主助手使用单个消费循环，用户输入优先于后台报告；Worker 独立执行。同一数据目录只允许一个后端进程：启动时获取进程锁，已有进程持锁则明确报错，退出或异常时释放锁。

| 代码 | 职责 |
| --- | --- |
| `backend/app/main.py`、`process_lock.py` | 启动数据库和 Runtime、获取进程锁、恢复及关闭。 |
| `backend/app/api/chat.py` | 主聊天快照、消息、审批和 SSE。 |
| `backend/app/agent/runtime.py`、`worker.py` | 主助手、任务调度、三个槽位及独立 Worker 循环。 |
| `backend/app/agent/agent_bus.py`、`task_tools.py` | 持久化内部消息、重试、等待环及助手通信工具。 |
| `backend/app/agent/system_prompt.md` | 主助手角色、直接执行与分配原则及简短回复要求。 |
| `backend/app/agent/llm.py`、`compact.py`、`tools.py` | 结构化路由、独立上下文压缩、文件分页及命令输出限制。 |
| `backend/app/sessions/chat_store.py`、`approvals.py`、`session_permissions.py` | 短事务、上下文投影、多任务审批及现有授权范围。 |
| `backend/app/models.py`、`agent/model_call.py`、`modelio.py` | 上下文及未处理消息查询索引；并发模型日志与轮转串行化，关联任务 / 运行 ID。 |
| `frontend/src/api/client.ts`、`App.tsx` 及组件 | 主聊天、按版本更新及任务审批。 |

现有 `turn_manager.py` 的单会话单轮和 `subtask.py` 的父轮同步等待不再用于后台任务调度。Runtime 捕获 Worker 循环异常并持久化失败通知；主循环异常写入错误并恢复后续消息处理，不能把异常当成功，也不能自动重跑结果未知的外部工具。模型快照写失败仅告警；核心状态未写入时不能假报已受理。

## 4. 主聊天接口与事件

| 接口 | 契约 |
| --- | --- |
| `GET /api/chat` | 返回 `{revision, assistant_id, reset, messages, approvals, permission_mode, context_usage}`。 |
| `POST /api/chat/messages` | 接收 `{assistant_id, client_message_id, text?, attachments?, permission_mode?}`；正文和附件至少有一项，保存后返回 202，不等待模型。相同请求幂等；复用 ID 参数不同时或该消息已被删除中断时返回 409。 |
| `POST /api/chat/uploads` | 每次上传一个文件；前端支持多文件，全部成功后发送消息。单文件最多 100 MiB，写入临时文件后原子替换工作区 `uploads/` 中的同名文件；助手 ID、上传 ID 和文件结果持久化，重复请求不得覆盖更新后的文件。 |
| `GET /api/chat/events` | SSE 先推送一致快照，再推送 `{revision, type, data}`，事件包括消息、审批和用量。 |
| `POST /api/chat/approvals/{request_id}` | 接收 `assistant_id`、`option_id` 和可选拒绝理由；失效请求或不同决定重复提交返回冲突。 |
| `POST /api/chat/reset` | 接收 `{assistant_id, reset_id, confirm_stopped?}`，幂等受理删除及必要的重试。 |
| `POST /api/chat/reset/cancel` | 接收 `{assistant_id, reset_id}`，取消未完成的删除；已停止的任务不自动恢复。 |

保留 `/api/permission/state`。订阅注册、取快照与事件发布由同一写锁及持久递增版本协调；每个连接独立队列，慢连接溢出时断开并重新取快照。流式文本分批先落库再广播；新输入推进回复代号，旧流迟到片段不得覆盖新回复。已接收的后台任务不因页面或主回复中断而取消。

助手 ID 随删除更换；旧页面的消息、审批和删除请求不可作用于新助手。删除状态持久化并通过快照广播，等待已启动工具及任务收口后，在同一事务中归档旧聊天、任务状态 / 报告 / 已保存输出并建立空白上下文；版本和用户操作序号持续递增。服务重启后，如工具结果未知，删除须待用户检查并确认操作已结束；不得自动重跑。归档仅保存数据，不在当前页面展示。

用户消息与审批选择统一分配持久递增的 `user_seq`。同一页面按发送 / 点击顺序提交，无须等待模型；多页面并发以服务端落盘顺序为准。模型判断携带来源序号，不能让较早消息的迟到判断覆盖同一操作较新的用户决定。

## 5. 调度与助手通信

- 每个任务有稳定 `task_id`、创建顺序、目标及独立上下文；一次执行有 `run_id`。执行槽位可复用，但上下文不能串用；同一上下文同时只有一个模型调用。
- 按用户最初交代顺序领取，续跑沿用原序。旧任务进入 `reviewing` 等主助手判断时，较晚任务先等待；已运行任务不抢占。用户任务优先于额外协助，无空槽时协助请求立即返回未分配。
- Worker 一轮最多 30 个模型步骤，每步模型调用前保存步数；第 30 步正常结束可以成功，需要继续时交主助手判断。主助手可立即新开一轮，重复且无进展时可停止。暂停同一轮保留 `run_id` 与已用步数；新一轮才重置。主助手默认单轮上限沿用现有 40 步。
- 主助手确认 A 依赖较晚的 B 时，记录依赖并可先安排 B；A 在模型步或工具结束的安全边界保存上下文、让出槽位，B 完成后主助手判断并恢复 A。B 的领取和待判断顺序继承最早等待任务的序号；B 已运行或等待用户时不重复领取。B 失败或停止通知主助手；已停止的 A 不恢复。等待用户审批的执行单元仍占槽位。
- 根任务可处于 `queued / running / waiting_user / waiting_dependency / waiting_agent / reviewing / completed / failed / stopped`；负责人和每个协助者另存运行状态。用户审批与任务依赖可同时存在；完成任务前须收口相关协助。
- 文件与命令工具经权限检查后使用 `asyncio.to_thread`，不阻塞聊天；线程不共享数据库 Session 或直接操作 `asyncio.Queue`。命令合并 stdout / stderr 并持续排空，读取和等待均有界。

主助手使用 `task_create / task_inspect / task_search / task_update / task_stop / task_review / task_wait_for / route_input`；Worker 使用 `agent_list / agent_send / agent_wait / agent_help / task_report`。主助手和所属任务可用 `read_tool_output` 查看已保存输出。工具返回“已受理”不等于执行完毕。

主助手可调用现有文件、命令等工具直接完成快速任务，不启动 Worker 模型或占用 Worker 槽位；直接工具最多三组并发，每个任务最多 40 个工具步骤，需要等待任务依赖时转交后台。工具执行沿用任务记录、权限、文件协调及结果持久化；等待工具或审批时仍可处理新消息，完成后由主助手继续回复。重启时已开始但结果未知的操作不重跑，待执行操作重新检查权限。

主助手首轮使用主助手提示词及对话上下文，输出完整意图和首个动作。快速工具动作在意图受理时一并保存并执行；简单命令通常只需首轮调用工具、结果返回后再调用模型回复。未提供动作时沿用主助手循环；待澄清的动作不执行，明确的独立意图仍可处理；创建、修改或停止任务时先按完整意图处理；普通文字回复保持流式输出。

Worker 可直接给忙碌同伴发信息，对方在当前模型步或工具结束后处理；请求额外工作只能分配空槽。停止或修改同时送负责人及相关协助者。等待回复时持久化等待并让出槽位；目标继承最早等待者的顺序，回复、对方结束、停止或变更时按原序恢复；重启后继续等待，被打断的工具调用补齐结果。

最终报告送达后才开放 `task_review`；模型调用当前未开放工具时不执行，最多重试一次。Worker 不得追加用户明确禁止的命令。

已知等待关系持久化为有向边，例如 `task-A → task-B → ctx-B → task-A` 表示 A 等 B、B 的执行者等 A 占用的文件。边变化时检测环并通知主助手判断，不自动终止；未声明的依赖与 PowerShell 内部等待无法完整识别。

## 6. 内部消息、状态与上下文

使用固定内部主会话 `Session.id = "main"`，复用现有 `Message / Part`，不新增表。隐藏控制 Message 存储以下 Part，用户消息和各上下文的模型消息各自独立：

开发阶段不做数据库迁移，旧多会话历史不迁移；测试仅使用独立空库。不得自动清空 `backend/data/`，其中可能包含工作区和日志。

| 记录类型 | 最小内容 |
| --- | --- |
| `task_state` | 任务 ID、目标、标题、创建序号、状态、负责人、依赖、停止意图及文件占用。 |
| `agent_context / agent_run / agent_summary` | 上下文 ID、任务 / 角色、原始模型消息、摘要边界、运行 ID、步数和运行状态。 |
| `agent_message` | 消息 ID、来源 / 目标、任务 / 上下文 / 运行 ID、类型、内容、关联请求、投递状态和重试信息。 |
| `approval / user_event / file_wait` | 审批请求及参数快照；用户消息 / 审批决定、先后序号和操作身份；文件路径、占用者和等待关系。 |
| `tool_execution / tool_group / tool_output` | 工具执行状态、成组调用及已采集输出；恢复时未知结果不重跑。 |

运行时锁和短 SQLite 事务串行化状态变更；每次事务独立创建 Session，不持锁等待模型、工具或用户。任务创建与工具受理结果、状态与待投递消息、目标上下文写入与投递标记分别原子提交。模型不得自报发送方或伪造授权；内部操作按上下文、运行及 `tool_call_id` 去重，参数不同的同 ID 重试报冲突。

消息先入库再通过 `asyncio.Queue` 唤醒：`pending → delivered → processed`。消息 ID 对目标上下文稳定；启动、空闲 / 中断时及定时按索引核对未处理记录，投递失败按有界退避重试。已投递但未处理的报告恢复时不重复插入上下文；只重试内部消息和报告，不重跑外部文件或命令。迟到的旧 `run_id` 结果不能覆盖新运行；修改已受理但在结束前未送达时保留记录并通知主助手判断是否续跑。关键报告到主助手，普通同伴消息不自动拼进主对话。

主对话和各任务分别组装、压缩；按上下文读取模型历史，用户只看到可见内容。原始聊天、助手消息及已采集工具结果保留；摘要 / 裁剪只作用于模型输入副本，摘要失败不推进边界，工具调用与结果保持配对。主回复被打断时，丢弃未完整解析的调用，保留已完成结果或补入中断结果。

模型每次只接收最多 50 个未完成任务及 5 个最近结束任务的摘要，总长度不超过 12,000 字符；省略时提示通过 `task_search` 查找，不确定时向用户确认。原始任务保留，`task_inspect` 按需分页查看详情和报告。

- `read_file` 参考 Hermes：按行分页，默认第 1 行、最多 2,000 行 / 10 万字符，按整行给后续行号。单行超过 2,000 字符则截断，该行余下部分不能按行号续读；原文件不变，此限制已确认。
- 沙箱命令输出最多保留 2 万字符，超出保留首尾并标记；按退出码、超时或执行异常标记失败。

## 7. 授权、用户修改与文件冲突

沿用现有权限优先级：硬拒 → yolo 模式 → deny → ask → allow → edit / build 默认规则。`allowSession` 只覆盖 `ask`，不能覆盖已判定的硬拒或 deny；授权仍存内存，在主会话内由所有任务共享，后端重启后失效。文件按工具和真实路径匹配，命令按完整参数精确匹配；`read_file` 新增的分页参数不改变同一文件原有的会话授权范围。

主助手判断用户消息期间，已有审批继续展示、可点击，新审批也照常提出；已获授权的操作继续执行。消息和审批按用户操作顺序记录，对同一有效操作以较新的决定为准。例如先说“不要删除 A”，判断未结束时又批准同一删除，则可执行删除，较早消息的迟到判断不能覆盖该批准；顺序相反时，已经开始的删除可能无法撤回。已结束任务、过期请求或参数变化仍按现有规则拒绝旧审批。

主助手判断用户消息时，结构化关联来源 `user_seq`、任务、受影响操作和意图；Runtime 校验后对同一操作比较较新的用户决定，过时的冲突意图不执行，其他不冲突要求仍处理。具体操作与停止整个任务分开判断；较新审批覆盖旧指令时，按受影响操作的审批事实回复。冲突关系不明确时重新判断或询问用户；审批不暂停。修改 / 停止意图及其去重记录原子保存。

`route_input` 包含 `{message_id, user_seq, intents: [{type, task_id, instruction, approval_ids?}], needs_clarification, first_action?}`，意图类型为 `chat / create / query / update / stop / unclear`；首个动作关联具体意图并携带工具及参数。使用指定工具调用并校验结果；意图受理与首个动作登记原子提交，较早修改遇到新的用户消息时持久化并重新判断。接口错误或结果无效时报告错误，不执行不确定的意图，审批仍可按现有规则处理。首版不增加 JSON 备用分支。

同用户 Worker 共享工作区；文件工具按容器内真实目标协调占用和版本，不同任务等待文件占用释放后须重读。沙箱命令可直接改文件，由 Worker 自行协调；具体边界见第 15 节。

## 8. 后台运行与恢复

- 消息已保存并返回 202 后，关闭网页只断开订阅；电脑、后端进程及所需网络正常时 Worker 继续，重开页面获取快照。等待审批的工作等用户处理。电脑休眠、关机或后端进程退出会中断，不能在进程停止期间继续。
- 启动先恢复用户消息 / 审批决定的顺序、任务 / 上下文及未处理通知，再提供快照并异步恢复工作；页面启动不等待模型。已路由但未完成的主回复重启后重新生成，不重复路由或创建任务；较新用户消息取代旧回复。缺少模型配置时仍可打开聊天，未完成任务保留并报告配置问题。等待中的旧审批以新 ID 发起，旧 ID 失效；操作以 `task_id + run_id + tool_call_id` 关联历史顺序，已批准但未开始的操作重新检查权限。
- 恢复排队、运行、等待及待判断任务；已完成、失败或用户已请求停止的任务不自动重跑。已完成但未转达的结果继续投递。关闭服务与用户停止任务分别记录。
- 工具已开始但结果未提交时，补入“结果未知”供 Worker 检查，不自动重跑，也不视为成功；任意外部命令不保证恰好执行一次。已开始的 PowerShell 命令按现有协作式停止规则收口。

## 9. 验收与未决风险

使用假模型、独立临时工作区与空 SQLite 验证：

1. 进入主聊天、持续追问及新任务并行；三个槽位上限、30 步续跑 / 停止、原序调度及任务 / 助手等待让位。
2. 跨 Worker 通信、忙碌边界投递、已知等待环、文件竞争、审批共享与按任务关联；连续消息及消息 / 审批的先后顺序均以最新有效决定为准，不丢弃无关要求；不同分页沿用同一文件授权。测试指定工具调用的无效输入及 API 错误。
3. 消息落盘后通知失败、主回复被打断、去重及进程内重试；模拟后台重启、未知工具结果、审批重发和任务创建幂等。
4. 启动本地 HTTP / SSE，接收任务后关闭页面，验证后台完成、重连快照和结果；验证同一数据目录不能启动两个后端，以及快照 / 增量交界断线、慢连接和日志失败。
5. 检查上下文隔离与工具调用配对，文件分页、超长行提示、1 MiB 命令采集和分段读取；执行前端构建、检查和浏览器验收。

本机验证：后端全量 90 项、首轮动作专项 16 项通过；此前前端 2 项测试、构建、lint 及浏览器主聊天验证通过。隔离环境使用 DeepSeek 验证 `pwd` 和文件写入各两次模型调用、混合意图及命令限制；此前验证并行、审批、回复恢复和历史查询。OpenAI 接入的验证状态见进度记录。

真实模型仍可能误解要求或误调工具；DeepSeek 思考模式不支持当前强制指定工具方式。模型服务不可用时停止 / 修改无法及时判断，已授权操作仍继续。PowerShell 无法完整识别外部副作用；Windows PowerShell 读取无 BOM 的 UTF-8 文件时可能需要 `-Encoding UTF8`。命令输出及超长单行按约定截断，持久记录持续累积。日志及干净库要求见 [reliability.md](reliability.md)。

## 10. 参考

- Hermes `9a76ac8e4f`：`tools/async_delegation.py`、`delegate_tool_registry.py`、`file_tools.py`、`file_operations.py`，参考后台委派、安全边界及文件分页；不照搬 Kanban。
- Codex `e72da2b538`：`codex-rs/core/src/unified_exec/`，参考命令输出采集。
- [DeepSeek `6337fc823ce`](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/effects.ts)：参考 Router / Worker 分工、待处理记录和失败重试；不照搬其授权或额外槽位。
- [Anthropic 工具选择](https://platform.claude.com/docs/en/api/http/messages)：指定 `tool_choice` 可要求调用特定工具；[部分思考模式存在限制](https://platform.claude.com/docs/en/build-with-claude/thinking)，当前客户端默认关闭该模式。
- [DeepSeek 接口](https://api-docs.deepseek.com/guides/anthropic_api/)与 [GPT-6 Luna 接口](https://developers.openai.com/api/docs/models/gpt-6-luna)：均可在关闭额外推理时指定工具。

## 11. 受控 Agentic RAG

- `rag/` 是独立 Python 环境及 stdio MCP 服务，仅 Worker 注册查询、导入、列表和移出；运行时另用内部取消接口。主助手将知识库请求交后台，不阻塞普通聊天；单纯问答可直接返回 Worker 答案。入库和移出核对用户原话中的明确指令，硬拒及显式权限规则仍有效；不确定的表达可能需重述。
- 检索子图按计划并行生成查询，结果去重后检索；转换均失败或为空时使用原问题兜底，已检索过的查询不重复。证据不足最多补检一轮，只评估证据，不复核生成后的事实；检索故障报告错误。
- RAG 模型及评测裁判请求由 SDK 处理重试：连接错误按指数退避最多重试 3 次，其他临时错误按 SDK 规则重试；超出后报告失败。
- 回答只覆盖问题及影响结论的必要条件；询问办理方式时说明直接相关的条件、限制和允许方式，数值随未知条件变化时按条件分别说明。保留原文的逻辑关系、适用范围、日期和例外，不用一般规则放宽特别要求；暂不输出引用编号和来源列表，兼容字段 `sources` 返回空数组。已知纯引用从回答中清除，证据原文仍供回答、忠实度评分及按需追踪使用。
- 资料支持可提取文本的 PDF、TXT、Markdown，单文件最多 100 MiB；PDF 使用 pdfplumber，Markdown 保留标题层级，TXT 保留段落。结构化结果为 `doc_id / source / tree`，保留可识别的标题、段落、表格、图片、页码、位置和顺序；PDF 的 `bbox` 为页面左上角原点的 `[x0, top, x1, bottom]`，单位为点。清洗前 JSON 保存在 `RAG_DATA_DIR/documents/<document_id>/<version>.json`，更新后清理旧版本，失败时警告；不做 OCR，不提示部分空文本页。
- 清洗多页重复页眉页脚和页码，业务警示至少保留一次；按结构切块，超限按字符拆分，上限 500、重叠 50，长度包含标题路径。块保留原文、标题路径和页码；标题与正文一起参与 Embedding、BM25 和重排，每批最多 32 块。无法可靠识别的结构不猜造；全部无文本时入库失败。
- 父子检索复用现有子块和结构 JSON：每轮最多 6 个查询，稠密和 BM25 各取 20，RRF 保留 10（常数 60）；合并去重后最多 20 块重排。按最高子块重排顺序去重，回填最多 5 份完整文档供证据评估和回答；读取失败时报错。全文可能超过模型上下文，不静默截断。
- Milvus Cloud 使用稠密/BM25 混合检索并本机重排，规划、评估与回答复用 Bot 模型配置。银行批量重建脚本会清除绑定集合、本地目录记录和旧结构化 JSON；保留原始工作区文件及评测资料。
- RAG 元数据独立保存在 `rag/data/`；同一工作区文件重新入库时先写新版本，成功后切换可检索版本，失败保留旧版。覆盖工作区文件不自动更新索引；移出仅删除知识库内容，删除助手保留知识库。
- 用户自行删除 law 旧 Milvus 集合；新服务不得自动清空或接管未识别的集合。测试仅用独立临时工作区、空库和隔离集合。上传逐项反馈，任一文件失败时保留草稿、不发送消息；纯文件消息无需正文。
- 首次执行 `uv sync --project rag`，参考 `rag/.env.example` 配置 law 的 Milvus 连接、本机模型路径及设备；安装并将 `llama` 加入 PATH。根启动脚本自动启动 Embedding / 重排服务，服务故障不停止 Bot。结果未知的操作需用户明确要求重试；已开始的云端请求可能无法撤回。
- RAG 和模型不记录密钥；现有模型快照可能包含检索内容。无独立知识库页面、通用 MCP 管理或 OCR；证据评估及回答仍可能出错，生成后不复核事实和引用支持关系。中断或云端清理失败可能留下不可检索的旧分块，暂不自动清理。
- 银行基准评测按最终前 5 份父文档统计文档 Recall@5 / NDCG@5，并计算回答质量和 MCP 延迟；保留有效数及失败耗时。Langfuse 保留追踪及证据读取；评测分数仅保存本地，不调用其评分接口。文档命中不代表条款覆盖。
- 独立检索评测直接复用 RAG 检索图，在回答前停止；按 `LANGFUSE_ENABLED` 上传逐题追踪和完整候选，本地保留题目、最终证据、警告、参数、耗时、指标和追踪标识，供独立生成评测使用。开启时先验证 Langfuse 配置与连接，运行中检测到的上传异常单独记录；关闭追踪或上传失败时不保留过程和完整候选。按目标文件编号计分，不代表条款覆盖；失败题单独记录。
- 独立生成和端到端评测使用去掉已知纯引用的正文作为展示和两项评分的唯一回答；旧追踪先核对原答案再清理，保留必要事实和警告，评测结果不另存原始回答。既有模型及历史追踪仍按原策略保留。生成评测从 JSON 读取固定问题、参考答案和原文证据，不依赖端到端脚本、Milvus、重排或 Langfuse；评分需要 Embedding 服务。新口径与旧分数需区分，固定评判模型和评分 Embedding 后才能比较。
- 独立检索和生成结果分别保存为 `rag/evaluations/result/retrieval_<月日时分>.json` 和 `generation_<月日时分>.json`；端到端批次保存在 `result/runs/`，评分结果保存在 `result/`。JSON 结果逐题原子更新；银行评测报告和建议分别写入 `result/report/` 和 `result/analysis/`。结果可能包含资料正文，提交或分享前检查敏感信息。
- 端到端生成和独立检索通过 `--dataset` 指定题集，当前选用原始题库前 50 题的 `bank-rag-v2-50.jsonl`；题目未提供 `behavior` 时记为空字符串。独立生成使用传入的固定证据，旧 v1 结果保留。
- `rag/evaluations/` 保留四个命令：`utils/run_bank_rag.py`、`utils/evaluate_bank_rag.py`、`evaluate_retrieval.py`、`evaluate_generation.py`。前者及独立检索指定 `--dataset .\rag\evaluations\benchmarks\bank-rag-v2-50.jsonl --limit 50`；端到端评分默认使用新目录中的最新批次，可用 `--run` 指定其他批次；独立生成指定 `--evidence <检索结果.json> --limit 50`。路径相对当前目录，默认只运行 1 题。
- 人工证据 JSON 可为题目数组或含 `cases` 数组的对象；每题至少包含 `id`、`question`、`reference`、`evidence`。`evidence` 为含 `filename` 和 `content` 的数组，允许空数组表示无证据。生成时只把问题和证据交给 RAG，参考答案仅供评分。

## 12. OpenAI 模型接入

- `LLM_PROVIDER` 默认为 `anthropic`，可选 `openai`；OpenAI 使用 `OPENAI_API_KEY`、可选 `OPENAI_BASE_URL` 和现有 `MODEL_ID`，修改配置后重启。RAG 复用同一配置。
- Bot 和评测裁判的 OpenAI 调用使用 Chat Completions；RAG 的 OpenAI 文本生成使用 Responses，均设置 `store=false`。Bot 固定 `reasoning_effort=none`；RAG 和裁判可另行配置，验收模型为 `gpt-6-luna`。
- 除 RAG 文本生成和评测裁判外，模型调用固定 `temperature=0`；追踪及新报告记录实际参数，历史结果不改写。
- RAG 文本生成在 `rag/.env` 用 `RAG_TEMPERATURE` 配置温度，默认 `0`，`none` 或留空时不传；可设 0–2，具体范围依模型接口。OpenAI 推理强度由 `RAG_REASONING_EFFORT` 配置，默认 `none`；GPT-6 Luna 的非 `none` 推理需不传温度。环境变量优先，修改后重启；追踪及评测记录实际配置。
- 评测裁判在 `rag/.env` 用 `EVAL_REASONING_EFFORT` 和 `EVAL_TEMPERATURE` 配置，默认 `medium` 和 `none`；温度 `none` 或留空时不传。RAGAS 参数映射不得覆盖设置，评测结果记录配置。
- 保留现有历史格式和 SQLite 表；请求时转换消息、工具调用及结果，流式聚合完整参数，支持指定工具与用量。参数无效、流中断或输出截断时不得执行不完整工具调用。
- 延续现有授权、上下文压缩和恢复语义，密钥不进入日志；用假服务及独立空库测试。用户填写真实配置并通知后再做真实调用。

## 13. DeerFlow 参考改造

参考本地 DeerFlow `2a16b26960cb953cd3283702d018bd7dce6b80b7`，在本项目自行实现；未批准或不清楚的设计和实现先与用户同步。

### 13.1 已确认目标

    用户 ↔ 现有前端 / Gateway ↔ 主 Agent（独立上下文）
                                     ↕ 提交任务 / 运行反馈
                            协调器 / Runtime
                                     ↕ 调度 / 生命周期 / 并发
                        后台 Worker × 多个（各自独立上下文）

    主 Agent + 各 Worker ↔ Shared State（任务 / 进度 / 结果）
    主 Agent + 各 Worker ↔ Message Bus（请求 / 事件 / 通知）
    应用状态和重要消息 + 各 Agent 检查点 → 同一 SQLite 文件

保留 README 的主 Agent、独立后台 Worker 和 2.1 Shared State + Message Bus；沿用 Vite 前端及现有调度规则。

### 13.2 DeerFlow 已核实机制

| 能力 | 源码现状与差异 |
| --- | --- |
| Agent 运行 | `AgentRuns` 提供独立线程、启动 / 取消、`steer` 和输入查询；`RunInputControl` 可在运行中处理输入，原生子 Agent 生命周期仍与父 Run 关联。`task` 不能直接代替独立 Worker；默认关闭的 `batch_task` 可能在租约过期后重试，适合只读或幂等任务。 |
| Agent Teams | 示例提供用户隔离、独立 Agent Run、定向请求、结果回传和共享状态，但无固定主 Agent、完整发布订阅或自动重连；成员请求按会话排队，示例最多八个活动任务。附件捕获 / 导入要求 POSIX，Windows 兼容性未验证。 |
| 存储 | Gateway 的应用数据、LangGraph Store 和检查点默认共用 SQLite；Teams 示例另用 SQLite，`run_events.backend` 默认是 `memory`。DeerMem 以用户 JSON 摘要和 Agent Markdown 事实为主存储，FTS 使用独立 SQLite 索引。 |
| 记忆 | 默认 middleware 模式，首次运行注入会话快照，轮次后异步提取；防抖 30 秒、最多 100 条事实、置信度阈值 0.7、注入预算 2,000 token。待提取队列在内存中，异常退出可能丢失待处理内容；`clear_memory(user_id, agent_name=None)` 清除该用户摘要和全部 Agent 事实。 |
| 沙箱 / 认证 | LocalSandbox 默认关闭宿主命令，不提供系统级用户隔离；`allow_host_bash` 不约束 MCP / 自定义工具。本地邮箱密码允许自助注册，OIDC 和可选细粒度授权默认关闭。 |

### 13.3 协调、消息与任务

- Gateway 按认证用户绑定独立 Runtime、主助手、Worker、任务、消息、审批及工作区，拒绝跨用户访问和伪造发送者。主助手和 Worker 分别保存上下文；Shared State 支持按任务或类型分页、版本冲突校验，Message Bus 支持定向发送及发布订阅，后加入者不回放旧消息。
- 业务数据和 LangGraph 异步检查点使用同一 SQLite 文件的不同表，关键检查点同步写入；应用事务与检查点事务协调写入以避免相互阻塞。任务、待发送消息在业务事务中持久化，进程内事件只作唤醒。消息在模型步骤或工具执行后的安全边界处理，消息和投递 ID 先写入目标检查点，再确认总线送达；无法确认时暂停并提示核查。目标已结束而消息未能处理时通知主助手，不自动重发。
- 每用户最多六个后台 Worker，保留任务顺序、依赖让位、审批占槽、主助手 40 步和 Worker 30 步的单轮限制；退出登录、切换账号、断线和主回复被打断不会取消后台工作。Worker 最终结果超过 16 KiB 时按 UTF-8 字符边界截断并提示。
- 成功、失败及停止的任务，在结果送达且依赖和通知处理后清理；先记录清理状态、删除 Worker 检查点，再在业务事务中删除任务数据并保留仅含请求 ID 的去重标记。中断后继续清理；未确认消息保留。主聊天结果、用户文件、长期记忆和模型日志保留。
- 不设置团队累计 100 个请求、共享状态 20 条 / 16,000 字节总量、最近消息 100 条、任务查询偏移 200、每 Run 16 条 / 64 KiB 或单根任务 12 个关联请求的示例限制；按需分页和现有上下文压缩仍限制单次模型输入。

### 13.4 恢复与授权

- 用户显式切换 build / edit / yolo 后，运行中和重启恢复的任务使用最新有效模式；文件审批继续沿用现有规则，重启后临时会话授权失效。已完成的工具结果复用，已开始但结果未知的工具不得自动重跑。
- 启动时按用户恢复安全的未完成工作，停止或等待中的任务遵循现有调度规则；检查点与业务状态不是同一事务，未知投递或外部操作必须等待核查。只允许启用的账号登录和创建运行时。

### 13.5 长期记忆、重置与工作区

- 每用户一份记忆，仅主 Agent 使用；分类摘要为 `memory.json`，事实按 DeerFlow 的版本、来源和 YAML 元数据逐条保存为 Markdown，主会话水位保存于 `watermarks.json`。首次无水位时提取当前主会话的用户原文和可见最终回复，过滤工具、内部消息、审批、纯上传及纯确认；按范围、持续性、描述性与置信度校验，识别明确纠错；新事实格式有误且涉及删除时整批拒绝。单条长消息保留首尾各 500 字，成功写入后推进水位。
- 提示词要求摘要和事实非常简洁且不失信息；摘要沿用 DeerFlow 的句数建议，七类正文合计不超过 1,000 token，每条事实正文不超过 1,000 token。长度由模型遵循，程序不强制截断。
- 各用户共用串行提取队列，静默 30 秒、最长防抖 300 秒；正常关闭等待 30 秒后取消未完成的提取，已开始的文件写入等待结束；异常退出后由下一次主助手处理事项触发重试。多文件更新按 DeerFlow 的日志、备份和版本检查恢复，重置代号阻止旧提取回写。主 Agent 每个 Run 以用户消息或报告所属任务目标召回，按词汇相关性与置信度各 50% 排序；纠错事实保障最多 500 token，普通预算 2,000 token。记忆以隐藏用户消息注入，不进入提取输入；Worker 不读取记忆。读取失败或超时提示用户并继续；不使用 FTS 索引。旧记忆不迁移，按用户要求删除。
- 重置停止该用户所有 Agent、使旧记忆提取代号失效，删除其聊天、任务、检查点、记忆及可确定归属的模型日志和轮转文件；新助手使用空白会话。旧版无法确认归属的日志保留并提示；其他用户和工作区不受影响。删除开始前可取消，开始后仅可等待或失败重试；重启继续未完成的删除。
- 用户工作区为 `backend/workspaces/<用户 ID>/`，与 `backend/data/` 分开且不入 Git；本机旧工作区已移至 `C:/Users/nnnnnn/Desktop/OpenWorkBot_旧工作区`，不导入新账号。Docker 沙箱见第 15 节，宿主命令关闭。

### 13.6 接口与配置

- 保留 Vite 前端、`backend/app/main.py` Gateway 入口、`start.py` 启动方式以及聊天、上传、审批、重置和 SSE 接口。聊天默认兼容原完整历史返回，前端使用分页上翻；账号名和密码自行注册登录，Cookie 保存登录态，写请求校验 CSRF。首版仅监听和允许本机回环访问。
- 主助手、Worker 和记忆提取共用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MODEL_ID` 指定的 OpenAI 兼容 Chat Completions；上下文压缩沿用 `CONTEXT_WINDOW` 等配置。`langgraph`、`langgraph-checkpoint-sqlite`、`jieba` 和 `tiktoken` 用于检查点及记忆；首次使用 token 词表可能需要联网。
- 应用与检查点使用 `backend/data/app.db`，记忆文件放在 `backend/data/memory/`。不做旧数据库迁移，测试使用独立空库；重建前核对数据目录，不能删除用户工作区。Docker 改造见第 15 节，PostgreSQL 待后续确认。
- 共享 RAG MCP 接入见第 14 节；RAG 服务及现有知识库保持独立。

### 13.7 验证与边界

- 隔离空库及模拟模型测试覆盖用户隔离、检查点同步、消息状态不确定、任务清理、重置重试与重启、记忆代号屏障、中文检索、分页、认证和每用户六个 Worker。前端执行构建、lint、现有测试及本机登录页面检查；完整结果见 [progress.md](progress.md)。
- 尚未使用真实模型复测本次改造，也未执行真实服务崩溃或断电试验；SQLite 与记忆文件不构成跨存储原子事务。参考实现不复制 DeerFlow 源码，应用当前定位为本机使用。

参考源码：[AgentRuns](../.github_deerflow/backend/packages/extension-api/deerflow_extension_api/agent_runs.py)、[Agent Teams](../.github_deerflow/examples/deerflow-extension-agent-teams/deerflow_extension_agent_teams/service.py)、[LocalSandbox](../.github_deerflow/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py)、[记忆队列](../.github_deerflow/backend/packages/harness/deerflow/agents/memory/backends/deermem/deermem/core/queue.py)、[记忆存储](../.github_deerflow/backend/packages/harness/deerflow/agents/memory/backends/deermem/deermem/core/storage.py)。

## 14. 共享 RAG MCP 接入

- Gateway 管理一个共享 stdio MCP 进程；后台 Worker 按现有权限与用户明确指令调用查询、列表、入库和移出，结果通过 Message Bus 送主助手及主聊天。保留 RAG 生成的答案、原有检索及 Responses 接口；最终结果按 16 KiB 截断并完成任务清理。
- 共用现有 `rag/data/` 目录、文档 ID 和 Milvus 集合；不迁移、重建或清空旧知识库。入库仅从当前用户工作区取文件，复制到受控临时目录后调用原 MCP；正常结束后清理，结果未知时保留临时文件待核查。按完整文件名（含扩展名、忽略大小写）查找，命中时使用旧目录路径调用入库以复用 ID；新版本成功后切换，失败保留旧版本，工作区原文件不变。共享库变更串行处理，查询可并发。
- 停止或重置时取消尚未发出的操作，已发出的入库或移出等待返回，再停止当前用户任务；结果未知不得自动重试。共享知识库、其他用户任务与现有 Langfuse 追踪保留；固定 MCP 服务由 Gateway 启动，宿主命令仍禁用。
- 优先只改 Gateway、Worker、MCP 客户端和项目文档；修改 `rag/` 代码前必须取得用户批准。隔离测试覆盖旧库、同名更新、权限、并发、取消和恢复；未授权真实模型或云端写入。评测重建脚本不得指向共享库。

## 15. 已批准的 AIO Docker 沙箱改造

- 主 Agent 和后台 Worker 保留每用户 Runtime 共用的 LangGraph `create_agent` 图、现有回调、调度、消息、审批、停止和恢复。
- 对齐 DeerFlow 的 `SandboxMiddleware`、`SandboxProvider`、`Sandbox`、`AioSandboxProvider`、`AioSandbox` 和 `LocalContainerBackend` 接口及职责，只实现 Docker AIO；不保留 Lark 参数，非空额外挂载明确报错。Skills 为所有 Agent 共享，不做单 Agent 隔离。
- 每用户共享一个 AIO 容器和一个网络代理，按 Agent / Run 管理独立租约与命令会话。无用户身份使用独立的 `default` 沙箱；无线程的底层调用使用独立逻辑 ID，多个逻辑 ID 指向同一用户容器。等待同伴或依赖时保留租约和 shell，直到 Run 结束。
- 静态 allowlist 位于 `backend/.env` 的 `SANDBOX_ALLOW_DOMAINS`，默认 `pypi.org`、`files.pythonhosted.org`、`registry.npmjs.org`、`github.com`，修改后重启。代理只转发允许的 HTTP / HTTPS 公网目标；未列域名在所有模式直接拒绝，不产生网络审批或临时授权。Docker Desktop 代理仅解析 IPv4。
- 文件工具使用容器内绝对路径及 DeerFlow AIO 文件接口；不再用 `openat2` 拦截符号链接。保留真实目标文件占用、版本检查和既有审批；写已有文件须先读当前版本，写后须重读。对齐 DeerFlow：写前读取报错或正文以 `Error:` 开头时放行检查，由 AIO 写接口和容器权限决定结果，可能绕过先读要求。命令可直接修改文件，由 Worker 协调，不设置工作区级锁。
- `str_replace` 替代 `edit_file`，默认替换第一处，可指定全部替换；新增 `ls` / `grep`，搜索过滤对齐 DeerFlow。`read_file` 使用 `start_line` / `end_line`，保留现有分页与截断。`write_file` 支持追加，非追加限 80 KiB，AIO 自动创建缺失的父目录。移除 `delete_file`，删除走 `bash`，不自动移入回收站。
- build / edit 每次 `bash` 调用独立审批，yolo 自动放行，均受网络 allowlist 限制。命令输出最多 2 万字符，默认超时 120 秒。底层接口支持单次 `env`：使用临时 bash 会话及工作区初始目录，不继承持久 shell 的目录或变量；模型工具不暴露 `env`。
- AIO 限制 1 GiB / 1 CPU / 512 PIDs，代理限制 256 MiB / 1 CPU / 128 PIDs，沿用 `seccomp=unconfined`。空闲 600 秒回收，3 个物理沙箱为软上限。进程清理无法确认时暂停该用户复用，待其他运行租约结束后回收；结果未知不自动重跑，Docker 不可用不回退宿主执行。
- 删除未挂载的 `/api/sessions` 旧会话链路及其专属测试、前端组件；当前聊天仍使用的模型、任务板、审批和记录能力保留。开发阶段不迁移旧数据。
- 使用假模型、独立空库和临时工作区验证接口、权限、文件版本、租约、停止、网络 deny 与恢复；专用 Docker 容器验证 AIO 文件、代理和命令会话，前端运行构建及测试。

## 16. TODO 与主助手改造

- 主助手、Worker 和协助助手各自在 LangGraph state 的 `todos` 中维护清单；通常由模型在三步及以上任务或用户明确要求时使用 `write_todos`，顺序执行建议只保留一项进行中，并行可多项，均不做硬性校验。复核报告的首步固定调用 `task_review`。基于 `TodoListMiddleware` 全量更新，允许空内容、多个进行中事项和空列表清空，同一模型轮最多调用一次。上下文丢失工具调用时注入隐藏的当前清单，不设置结束提醒或等待信号。旧 TODO 数据清空，保留空表，不兼容旧工具。
- 移除 `route_input` 模型路由，主助手直接判断并使用现有任务和工作工具。同一轮调用先整体登记，再按模型给出的顺序执行；出错后停止后续调用并交主助手判断。必要时可先查询任务再执行。用户消息和审批按序处理，同一操作较新的决定优先，相关性由模型判断；不新增审批编号字段。
- 沿用现有权限、异步聊天和后台任务边界；不新增 TODO 的重启恢复流程，模型根据可用上下文自行判断，结果未知的外部操作不自动重跑。使用独立空库和模拟模型验证状态隔离、压缩提醒、工具顺序、失败停止、消息与审批顺序和任务清理。
