# AGENTS.md

OpenWorkBot, 目标打造类OpenAI Dots 的自主智能员工, 面向个人与企业提供主动执行, 多智能体协同, 长期记忆, 工作流自动化的AI原生产品, 构建可自主规划, 自动调用工具, 自主迭代的超级Agent舰队

## 开始工作
1. 读取 scripts_mini_harness/mini_harness.py, 作为此项目harness思想的参考
2. 阅读 docs/prd.md了解功能需求与用户侧交互
3. 阅读 docs/architecture.md，了解项目架构
4. 读取 docs/progress.md，了解最新会话进度日志
5. 读取 docs/feature_list.json，确认开发进度
6. 读取 docs/reliability.md，了解运行与验证要求

## 结束会话前：
- 更新 docs/progress.md
- 更新 docs/feature_list.json
- 更新 docs/progress.md, 记录仍未解决的风险或 blocker
- 在工作处于安全状态后，用简洁的中文提交

## 工作规则
- 新增功能时，请**先更新对应文档，再编写代码**。
- 优先依赖仓库里的持久化文件，而不是聊天记录。
- 除非为了消除当前 blocker 的窄范围修复，否则不要扩大到其他功能。
- 数据库不做 schema 迁移（本项目仍在开发阶段, 不涉及到数据库的保留以及历史数据迁移）：改 models.py 表结构后，删除 backend/data/ 重启即全新建表；测试一律从空库开始。
- 如果涉及到修改docs目录下的文件以及AGENTS.md，要求内容非常简洁, 但不失信息

## 代码规范
- 代码要求简洁精炼, 避免使用复杂的语法或模式, 保持代码结构清晰

## 项目目录
一级目录:
- backend/ — 后端代码
- frontend/ — 前端代码
- docs/ — 项目文档
- scripts_mini_harness/ — harness 思想的参考实现: mini_harness.py
- tmp/ — 临时文件目录(不入库, 可随时清空)

项目开发参考目录(.开头):
- .github_codex/ — OpenAI Codex CLI 源码仓库克隆, agent harness工程实现参考: https://github.com/openai/codex
- .github_ZCode/ — ZCode 源码仓库克隆, agent harness工程实现参考: https://github.com/zai-org/ZCode
- .github_learn-claude-code/ — learn-claude-code 教程仓库克隆, agent harness思想参考: https://github.com/shareAI-lab/learn-claude-code


## 特别规定
- README.md 暂不修改
