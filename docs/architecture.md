# OpenWorkBot 架构


## 1. 组成与数据流

React/TypeScript → FastAPI 会话 API → Agent Loop → 权限/工具 → 记录器 → SQLite → 回放或模型历史。
前端通过 SSE 接收实时事件，断线后查询已保存结果。

| 模块 | 入口 | 职责 |
| --- | --- | --- |
| 界面 | frontend/src/App.tsx、frontend/src/hooks/useSessionStream.ts | 会话、执行过程、审批、子助手面板 |
| 会话入口 | backend/app/api/sessions.py | 装配执行依赖、后台任务、SSE、停止和审批 |
| 主循环 | backend/app/agent/loop.py | 模型往返、工具分发、步数与停止检查 |
| 模型适配 | agent/llm.py、agent/model_call.py | Anthropic 兼容流、工具参数、用量记录 |
| 工具与扩展 | agent/tools.py、todo.py、subtask.py、skills.py、hooks.py | 文件/命令、任务板、子助手、技能、钩子 |
| 权限 | agent/permission_service.py、sessions/approvals.py | allow/deny/ask、授权范围与等待 |
| 存储投影 | sessions/recorder.py、store.py、replay.py | 执行事实、界面回放、模型历史 |
| 配置与日志 | config.py、obs.py、modelio.py | 环境配置、工程日志与模型快照 |

agent/、sessions/ 路径均相对 backend/app/；界面文件位于 frontend/。

## 2. 执行边界

- 单后端进程；同会话单 turn，不同会话可并行；工具与子助手在单 turn 内串行。
- 主循环最多 40 模型步，子助手最多 30 步；子助手只提供 7 个同步工具，无嵌套 subtask。
- 工具失败以 Error: 结果回填；模型异常失败收口。仅明确输入上下文超限且未输出时允许一次安全恢复。
- 无工具且 Stop hook 无注入时结束；Web 未实现独立目标验收器。UserPromptSubmit 仅预留。
- todo 最多 20 项、1 项进行中；全量保存任务板，不自动调度任务。
- Skills 按目录加载全文；目录枚举未注入 Web 系统提示词。

## 3. 状态与持久化

| 位置 | 内容 | 重启后 |
| --- | --- | --- |
| session | 会话、项目归属、标题、时间 | 保留 |
| message | 角色、轮次、模型、用量等元信息 | 保留 |
| part | 正文、工具状态/结果、压缩边界 | 保留 |
| local_setting | 项目权限模式与规则 | 保留 |
| todo | 当前任务板 | 保留 |
| turn_usage | 轮次状态、总耗时、主/子用量 | 保留 |
| 进程内登记 | 活跃 turn、停止信号、审批槽位、会话授权 | 丢失 |

SQLite 开启 WAL、外键与 busy_timeout。记录可更新，不是不可变事件日志。
子助手内部历史不独立保存为消息序列；父工具保存目标、状态和输出，模型快照另记 subtask_id。
刷新不取消后台执行；后端重启不自动续跑。重新生成替换最后一轮，但不撤销文件/命令副作用。

## 4. 上下文与事件

- load_history 为模型组装消息；load_replay 为界面组装执行块，两者共用存储而非共用展示内容。
- 每个模型步检查预算；先替换旧工具输出，再按需摘要旧轮前缀，保留当前用户原文与活跃工具配对。
- 压缩阈值为窗口减输出预留与缓冲；空摘要、未缩减或失败不提交新边界。
- 工具执行、审批、子助手事件进入共享队列；SSE 只消费，不提供持久事件重放。
- SSE 心跳用于连接保活，不是主动任务调度。

## 5. 权限与已知限制

build/edit/yolo 模式与项目规则决定 allow/deny/ask；审批支持本次、会话范围、拒绝。子助手沿用父级约束。
文件工具限制真实解析路径；PowerShell 仅设置工作目录，不提供系统级隔离。停止为协作式信号，不保证强杀已运行命令。

- 普通 tool_started 在工具完成后才发；审批弹窗缺独立停止入口。
- 压缩摘要调用未进入常规模型快照/用量；入口未继承上一轮真实用量。
- 无长期记忆、通用 MCP、并行子任务调度、跨进程执行恢复、认证或多租户。
- 端口默认值存在分歧：Settings=8100、load_settings=8000，启动器与前端代理使用 8100。
- 旧测试已移除，package.json、pyproject.toml 与 uv.lock 仍有测试入口/依赖残留。
- 部分界面名称及提示词仍沿用旧项目；提示词称“沙箱”不代表具备系统隔离。

## 6. 配置与启动

Python 3.12+、uv、Node.js/npm；命令工具需要 PowerShell。使用 uv.lock 和 package-lock.json
后端在 backend/ 安装依赖，前端在 frontend/ 安装依赖
根目录 `uv run --no-project start.py` 启动前后端：127.0.0.1:8100 与 127.0.0.1:5173；/api 代理至后端。
健康接口为 /api/health，不依赖真实模型配置。当前无鉴权，不直接暴露公网。
