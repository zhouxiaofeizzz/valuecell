# ValueCell 工作区架构与二次开发指南

## 1. 总体架构

ValueCell 是一个前后端分离 + 多智能体内核的系统，核心由以下层组成：

- 前端应用：`frontend/`，提供 Web UI，负责交互与展示。
- 后端服务：`python/valuecell/server/`，基于 FastAPI 提供业务 API 与模型/智能体调用入口。
- 智能体内核：`python/valuecell/core/`，包含编排器、规划器、任务执行、事件路由与会话存储。
- 智能体实现：`python/valuecell/agents/`，内置研究、新闻、策略、交易等智能体。
- 配置体系：`python/configs/`，提供模型与智能体的 YAML 配置与 agent cards。
- 适配器层：`python/valuecell/adapters/`，封装市场数据与模型提供方。
- 桌面客户端：`frontend/src-tauri/`，Tauri 桥接前端与系统能力。

## 2. 目录结构速览

仓库根目录：

- `frontend/`：前端应用（页面、组件、API 调用封装、资源）。
- `python/`：后端与智能体核心代码。
- `docs/`：架构与配置说明（可作为扩展阅读）。
- `start.ps1` / `start.sh`：一键启动脚本。

后端核心：

- `python/valuecell/server/`：API 层、服务层、数据库模型。
- `python/valuecell/core/`：编排器、规划器、任务执行、事件路由、会话与存储。
- `python/valuecell/agents/`：具体智能体实现与工具逻辑。
- `python/valuecell/adapters/`：市场数据与模型提供方的接入封装。
- `python/valuecell/config/`：配置加载与常量。

配置与资源：

- `python/configs/`：`config.yaml`、`agents/*.yaml`、`providers/*.yaml`、`agent_cards/*.json`。
- `assets/`：产品图片与展示资源。

## 3. 运行链路（核心流程）

启动脚本 `start.ps1` 的主要流程：

1. 检查并安装 `bun`、`uv`。
2. `uv sync` 同步 Python 依赖并初始化数据库。
3. 启动前端：`bun run dev`（默认端口 1420）。
4. 启动后端：`uv run python -m valuecell.server.main`。

后端启动入口：

- `python/valuecell/server/main.py` 启动 Uvicorn，创建 FastAPI 应用。
- `python/valuecell/server/api/app.py` 组装路由与中间件。

智能体运行链路（内核）：

- 编排器：`python/valuecell/core/coordinate/orchestrator.py`
- 超级智能体：`python/valuecell/core/super_agent/`
- 规划器：`python/valuecell/core/plan/`
- 任务执行：`python/valuecell/core/task/`
- 事件路由：`python/valuecell/core/event/`
- 会话与存储：`python/valuecell/core/conversation/`

## 4. 配置体系与加载优先级

配置优先级从高到低：

1. 环境变量
2. `.env`（系统路径）
3. YAML 配置（`python/configs/`）

关键文件：

- `python/configs/config.yaml`：模型与智能体的全局配置入口。
- `python/configs/providers/*.yaml`：模型提供方配置。
- `python/configs/agents/*.yaml`：智能体配置。
- `python/configs/agent_cards/*.json`：A2A 能力描述。

`.env` 的系统路径（启动脚本会提示）：

- Windows: `%APPDATA%\ValueCell\.env`
- macOS: `~/Library/Application Support/ValueCell/.env`
- Linux: `~/.config/valuecell/.env`

`APP_ENVIRONMENT` 会影响是否加载 `config.<env>.yaml`（如存在），否则回退到 `config.yaml`。

## 5. 二次开发流程

### 5.1 环境准备

- Python 3.12+
- 包管理：`uv`
- 前端工具：`bun`

### 5.2 启动开发环境

一键启动：

```powershell
.\start.ps1
```

分离启动：

```powershell
# 后端
cd python
uv run python -m valuecell.server.main

# 前端
cd frontend
bun run dev
```

### 5.3 后端二次开发

常见改动位置：

- 新增 API：`python/valuecell/server/api/routers/`
- 数据模型：`python/valuecell/server/db/models/`
- 业务逻辑：`python/valuecell/server/services/`
- 响应结构：`python/valuecell/server/api/schemas/`

### 5.4 智能体二次开发

推荐流程：

1. 在 `python/valuecell/agents/` 下新增智能体目录与核心逻辑。
2. 在 `python/configs/agents/` 中添加对应 YAML 配置。
3. 在 `python/configs/agent_cards/` 中添加能力描述 JSON。
4. 如需接入新模型提供方，在 `python/configs/providers/` 增加配置。

### 5.5 交易与数据适配

- 交易执行：`python/valuecell/agents/common/trading/execution/`
- 数据源适配：`python/valuecell/adapters/assets/`

### 5.6 前端二次开发

主要目录：

- 页面：`frontend/src/app/`
- API 封装：`frontend/src/api/`
- 组件库：`frontend/src/components/`
- 资产与图标：`frontend/src/assets/`

### 5.7 测试与校验

后端测试：

```powershell
cd python
uv run pytest
```

前端目前未在仓库中提供统一测试命令，可按业务需要补充。
