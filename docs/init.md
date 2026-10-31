# OpenWorkBot — 环境与启动

以下命令来自当前代码配置，本轮未执行验证。无需恢复 law2 的数据库或日志。

## 环境

- Python 3.12+、uv、Node.js 与 npm；版本兼容性以锁文件及实际安装结果为准。
- 命令工具当前调用 PowerShell，使用该工具前需要相应可执行程序；启动服务不等于命令工具可用。
- 后端依赖在 backend/，前端依赖在 frontend/；根目录不是 Python 包。

## 准备

1. backend/ 执行 `uv sync --frozen`。
2. frontend/ 执行 `npm ci`。
3. 按需将 backend/.env.example 复制为 backend/.env，已有文件不要覆盖。
4. 真实对话需要配置 ANTHROPIC_API_KEY、MODEL_ID，兼容接口可设置 ANTHROPIC_BASE_URL；健康检查不需要真实模型凭据。

## 启动

根目录执行 `uv run --no-project start.py`，或分别启动：

- backend/：`uv run uvicorn app.main:app --host 127.0.0.1 --port 8100`
- frontend/：`npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`

前端：http://127.0.0.1:5173；健康接口：http://127.0.0.1:8100/api/health。
前端 /api 代理固定指向 8100。start.py 同时管理前后端，Ctrl+C 关闭其创建的进程。
当前 Settings、load_settings 与启动器端口默认值不完全一致，先使用上述显式端口，不依赖隐含默认值。

## 验证入口

frontend/ 保留 `npm run lint`、`npm run build`。这是检查入口，不是本次通过结论。
旧测试已删除，不运行 npm test 或 pytest 作为有效验收；配置残留见 [RELIABILITY.md](RELIABILITY.md)。

## 数据

backend/data/ 保存数据库、工作区与工程日志，log/ 保存模型调用快照；均不入库。
只监听本机，不通过改为 0.0.0.0 直接对外提供服务。
