# OpenWorkBot

<br />

**项目介绍**

OpenWorkBot 旨在打造类 OpenAI Dots 的自主智能员工,面向个人与企业提供主动执行、多智能体协同、长期记忆与工作流自动化能力,构建可自主规划、调用工具、根据反馈调整执行的 Agent 协作系统。

**系统架构**

项目采用主Agent与后台 SubAgent Worker协作的架构：
- 主助手负责与用户交互、理解意图、协调以及简单任务。
- SubAgent Worker 负责执行复杂或耗时的工作，并将进度和结果反馈给主助手。

将对话从一问一答的半双工模式转变为全双工模式
![全双工模式](./readme_docs/subagentWorker.png)


## 通过 MCP 接入 LangGraph RAG 知识库

RAG 作为独立 MCP 服务，由后台 Worker 调用。LangGraph 按问题规划查询，通过 Milvus 稠密向量和 BM25 混合检索、本地重排及证据评估生成回答；证据不足时并通过 Grounding 反馈加入了 Reflection/自我纠错进行补检。

### 离线构建

1. **结构化解析**：提取 带层级关系的 Structured Document JSON: 保留来源,名称,格式,标题层级、段落、表格、页码和顺序。
2. **清洗**：去除重复的页眉页脚和多余空白，保留数字、日期、否定词及表格关系。
3. **切块**：先按标题、章节和段落切分；超过上限时再拆分500/50重叠。每个 Chunk 保留标题层级关系。
4. **建立索引**：标题与正文参与 Embedding，写入 Milvus 稠密向量和 BM25 索引。

### 在线检索

![LangGraph RAG 检索流程](./readme_docs/rag_langgraph.png)

1. 根据问题重写, 多查询, 查询扩展, 问题拆分等RAG增强策略, 每轮最多 6 个子查询；
2. 稠稠密检索和 BM25 各取 20 条，经 RRF 融合后每个查询保留 10 条；
3. 共 60 条合并去重后Top 20 条重排，聚合取 Top5 父Chunk供证据评估和回答。

### 回退降级

比如LLM的APIConnectionError

1. 指数退避策略 重试3-5次
2. 查询转换全部失败或为空时使用原问题
3. 使用规则进行降级

### RAG 测评结果

| 指标                 |                                    基线结果 | 作用    |
| ------------------ | --------------------------------------: | ----- |
| 文档 Recall\@10      |                          0.880 --→ 1.00 | 找全了没  |
| 文档 NDCG\@10        |                         0.776 --→ 0.886 | 排序好不好 |
| Faithfulness       |                         0.710 --→ 0.970 | 有没有编造 |
| Answer Correctness |                         0.473 --→ 0.802 | 回答对不对 |
| MCP 平均延迟 / P95     | 28.09 秒 / 35.47 秒 --→ 32.25 秒 / 43.27 秒 | 响应时间  |

- 生成为 gpt-6-luna；测评为 gpt-6.1-terra

### RAG 测评脚本

```bash
# 1.启动模型
cd C:\Users\nnnnnn\Desktop\OpenWorkBot
.\rag\.venv\Scripts\python.exe .\rag\run_models.py
# 2.上传知识库
.\rag\.venv\Scripts\python.exe .\rag\evaluations\utils\upload_bank_rag.py
# 3.检索评测
.\rag\.venv\Scripts\python.exe .\rag\evaluations\evaluate_retrieval.py --dataset .\rag\evaluations\benchmarks\bank-rag-v2-50.jsonl --limit 50
# 4.生成评测
.\rag\.venv\Scripts\python.exe .\rag\evaluations\evaluate_generation.py --evidence .\rag\evaluations\result\retrieval_11-05-18-50.json --limit 50
```

# 项目实现

## 1. 九个机制

主链路：用户输入 → 压缩 → LLM → 权限 → 工具 → tool\_result → 再调 LLM。

| 机制                 | 本项目实现                                                                                                        | 补充说明                                                                                         | 代码文件                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Agent Loop      | run\_turn 进行While True循环；无工具则结束；                                                                             | 错误也作为 `tool_result` 喂回，不抛异常打断循环                                                              | [agent/loop.py](./backend/app/agent/loop.py)、[agent/llm.py](./backend/app/agent/llm.py)、[agent/model\_call.py](./backend/app/agent/model_call.py) |
| 2. Tool Use        | `TOOLS` 定义工具及参数 schema；`TOOL_HANDLERS` 为名字 → 函数的 dispatch map；todo\_write 和 subtask 由主循环单独处理。                | 文件操作全部先过 `safe_path` 沙箱, `execute_tool` 统一拦截 → 分发 → 兜异常                                      | [agent/tools.py](./backend/app/agent/tools.py)、[agent/loop.py](./backend/app/agent/loop.py)                                                       |
| 3. Permission      | 主循环check\_permission根据权限模式build/edit/yolo, 返回allow/deny/ask；命中危险命令清单或越界时deny，批准后把工具结果给模型；拒绝后把拒绝原因给模型，让它决定下一步 | `permission_hook` 三道闸门：禁止清单硬拒 / 越界与 shell 删除硬拒 / 高危命令 `[y/N]` 确认；拒绝原因作为 `tool_result` 喂回模型   | [agent/permission\_service.py](./backend/app/agent/permission_service.py)、[sessions/approvals.py](./backend/app/sessions/approvals.py)            |
| 4. Hooks           | `UserPromptSubmit / PreToolUse / PostToolUse / Stop` 四个事件点挂回调                                                | `Pre` = 权限 + 日志；`Post` = 大输出告警；`Stop` = 计数；`Submit` 暂无注册                                     | [agent/hooks.py](./backend/app/agent/hooks.py)、[agent/loop.py](./backend/app/agent/loop.py)                                                       |
| 5. Task System     | `TodoManager`任务看板, todo\_write整板更新；持久化sqlite3；                                                               | 单 `in_progress`串行进行                                                                          | [agent/todo.py](./backend/app/agent/todo.py)                                                                                                      |
| 6. Subagents       | 独立历史、最多30步、串行、防递归；当前7个同步工具，权限同父级，结果回填父轮                                                                      | `subtask(prompt)` 用全新 `messages` 跑 30 轮独立循环，最终文本作 `tool_result` 返回父级；同一响应里的多个 `subtask` 严格串行 | [agent/subtask.py](./backend/app/agent/subtask.py)                                                                                                |
| 7. Context Compact | turn开始执行microcompact → LLM摘要；阈值与恢复边界见§3                                                                      | <br />                                                                                       | [agent/compact.py](./backend/app/agent/compact.py)                                                                                                |
| 8. Skill           | 启动扫描 `skills/*/SKILL.md`，system prompt 只放“名称 + 描述”目录，`load_skill` 按需取全文                                      | frontmatter 只取 `name / description` 两字段                                                      | [agent/skills.py](./backend/app/agent/skills.py)                                                                                                  |
| <br />             | <br />                                                                                                       | <br />                                                                                       | <br />                                                                                                                                            |

<br />

- 工具：bash/read\_file/write\_file/edit\_file/glob/delete\_file + todo\_write/load\_skill/subtask。
  - 子助手仅同步工具：6个基础工具 + load\_skill；不含todo/subtask。
  - 文件读/写/改/删经safe\_path限制于workspace
- check\_permission权限判断:

| 判断结果  | 含义                      |
| ----- | ----------------------- |
| allow | 允许，直接执行工具               |
| deny  | 拒绝，不执行，将拒绝原因回填模型        |
| ask   | 等待用户审批；批准后执行，拒绝则将原因回填模型 |

| 权限模式        | build 变更前确认 | edit 自动编辑 | yolo 完全访问 |
| ----------- | ----------- | --------- | --------- |
| 读取工作区文件     | allow       | allow     | allow     |
| 写入、编辑工作区文件  | ask         | allow     | allow     |
| 删除文件、执行一般命令 | ask         | ask       | allow     |
| 命中硬拒规则      | deny        | deny      | deny      |

| ask选项                                        | 含义                       | 是否保存规则                    |
| -------------------------------------------- | ------------------------ | ------------------------- |
| Allow once（允许一次）                             | 只批准当前这次工具调用              | 不保存长期规则                   |
| Always allow in this conversation（在此对话中始终允许） | 批准当前调用，后续在当前会话中匹配的操作不再询问 | 只保存在内存，不写入项目规则；重启或新建会话不保留 |
| Deny（拒绝）                                     | 不执行当前调用，拒绝原因回填模型         | 不添加永久拒绝规则                 |

## 2. 对话数据存储

### 2.1 SQLite：会话事实

session表

- 会话id、project\_id、工作区完整路径、标题、时间、归档

message表

- 消息id, session\_id, 顺序sequence, 时间
- &#x20;data(身份、模型、轮次、用量、错误等元信息)

part表

- 部件 id, message\_id, session\_id, sequence, 时间
- data(正文text、工具tool或压缩信息compaction)

local\_setting表

- 项目权限模式:{"mode":"build/edit/yolo"}
- 规则ruleset:{"allow":\[{"toolName":"bash","ruleContent":"git status"}],"deny":\[{"toolName":"delete\_file"}],"ask":\[{"toolName":"write\_file"}]}

todo表

- 联合主键(session\_id, 顺序position), 任务内容content,  时间
- 状态status: pending 待办 / in\_progress 进行中 / completed 完成

turn\_usage表

- 联合主键(session\_id, turn\_id), user\_message\_id, 时间, 总耗时, 输入/输出及缓存 token
- status: running 运行 / completed 完成 / error 失败 / cancelled 停止

### 2.2 model-io：模型调用快照

实现：[modelio.py](./backend/app/modelio.py)；主/子循环共用model\_call记录每次调用，成功/失败均写入。

文件

- 项目根`log/model-io-<session_id>.jsonl`；每会话一个文件，一调用一行JSON追加

标识

- time、session\_id、turn\_id、subtask\_id、model

request

- system/messages全文、tool\_names（不保存工具schema）

response

- text、tool\_calls（id/name/input）

usage

- input/output tokens、cache\_read/cache\_creation tokens

诊断

- duration\_ms、error

目录/轮转

- MODELIO\_DIR可覆盖；MODELIO\_MAX\_MB默认10；追加前检查，超限以纳秒后缀归档

用途/失败

- 调试审计，不参与回放；写失败吞异常、记WARNING，不阻断对话

## 3. 上下文压缩

触发

- 历史token > 窗口 − 输出预留 − 安全buffer；默认预留32K + 13K

microcompact

- 直接将库内旧工具输出改为占位符；默认保留最近5条完整输出，重载历史后再判断预算

compact

- 仍超预算则LLM摘要此前历史；当前用户请求保留原文，不参与摘要

## 4. 项目启动

需要 Python 3.12+、uv 和 Node.js, 配置 `backend/.env`，然后启动：

```bash
uv run --no-project start.py
```

### 初版最小harness实现(仅参考) `scripts_mini_harness/mini_harness.py`

```text
用户输入 -> 压缩 -> LLM -> 有 tool_use ? -> 权限 hooks -> 工具 -> 回填
                                | 否: /goal 模式过目标闸门,否则结束

Tool:
bash / read_file / write_file / edit_file / glob /
delete_file(移入 rubbish/) / todo_write / load_skill / subtask(子助手)
```

- Context Compact
- 四级压缩：新结果落盘留预览 → 中间历史归档 → 旧结果缩短 → LLM 摘要重写；真实 token 计量，占窗口 80% 触发，压到 40%, 归档；重试 1 次

