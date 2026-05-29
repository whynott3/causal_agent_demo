# 因果领域智能助手

一个基于 FastAPI + LangChain / LangGraph 的因果推断与因果发现助手。项目当前支持因果概念问答、DAG 路径分析、CSV 数据上传与画像、因果发现算法启发式推荐，以及单页前端的 Markdown / KaTeX / Mermaid / 数据卡片渲染。

## 当前能力

- **因果知识问答**：优先检索本地 Chroma 知识库，必要时调用 Tavily 网络搜索。
- **DAG 文本分析**：解析 `A -> B` 形式的 DAG，识别因果路径、后门路径、调整集合，并输出 Mermaid 图。
- **CSV 上传与数据画像**：上传 `.csv` 后自动生成 `profile.json`，展示行列规模、列类型、数值列分布、分类取值、相关性与警告。
- **数据驱动工作流**：上传数据后，Agent 给出概览并询问下一步意向。
- **因果发现算法推荐**：基于 profile 启发式推荐 PC / GES / FCI / NOTEARS / LiNGAM / PCMCI，并以前端算法选择卡片展示。
- **多会话记忆**：使用 LangGraph `SqliteSaver` 保存对话历史，支持读取和清空会话。

当前 Sprint 3 的定位是“推荐算法，不实际运行发现算法”。真实运行 PC / FCI / GES 等因果发现算法、输出 DAG 和逐边解释留给后续 Sprint。

## 技术栈

- 后端：FastAPI、Pydantic、Uvicorn
- Agent：LangChain 1.x、LangGraph、DashScope OpenAI 兼容接口
- 检索：Chroma、LangChain Chroma、Tavily Search
- 图分析：NetworkX
- 数据画像：pandas、numpy、scipy
- 前端：单文件 `app/static/index.html`，CDN 引入 marked、KaTeX、Mermaid
- 测试：pytest、FastAPI TestClient / httpx

## 目录结构

```text
app/
  main.py                         # FastAPI 入口，挂载 API 与静态前端
  api/v1/
    chat.py                       # /chat/stream, GET/DELETE /chat/messages
    upload.py                     # CSV 上传、profile/status API、uploads 清理
    oss.py                        # OSS 预签名接口（沿用）
  agents/
    causal_agent.py               # Agent、系统提示词、工具挂载、会话记忆
  common/
    logger.py                     # 日志配置
    runtime_context.py            # 当前 thread_id 的 ContextVar 运行时上下文
  data/
    textload.py                   # 本地知识库构建与检索
    causal基本概念.txt
  tools/
    dag_utils.py                  # DAG 解析、路径、调整集合、Mermaid
    data_profile.py               # CSV profile 与数据画像摘要工具
    algo_recommend.py             # 因果发现算法推荐启发式工具
  static/
    index.html                    # 前端 SPA
  tests/
    test_dag_utils.py
    test_data_profile.py
    test_upload_api.py
    test_algo_recommend.py
  uploads/
    .uploads_guard                # 占位文件，真实上传数据不入库

requirements.txt
```

## 环境变量

项目通过 `.env` 或系统环境变量读取模型与搜索配置。

```env
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=your_dashscope_api_key
TAVILY_API_KEY=your_tavily_api_key
```

可选测试变量：

```env
APP_UPLOADS_DIR=/absolute/path/to/uploads
```

`APP_UPLOADS_DIR` 主要用于测试隔离；正常运行时上传目录默认为 `app/uploads/`。

## 安装与启动

建议在已有 conda 环境中安装依赖：

```powershell
pip install -r requirements.txt
```

启动后端和前端：

```powershell
cd app
python main.py
```

注意：`python main.py` 默认使用 `127.0.0.1:8001`。

## 测试

```powershell
cd app
python -m pytest tests/ -q
```

当前测试覆盖：

- DAG 工具：解析、环检测、前门/后门路径、调整集合、Mermaid
- 数据画像：类型识别、缺失、相关性、GBK 编码、摘要工具
- 上传 API：CSV 上传、编码兜底、大小限制、status/profile、清理
- 算法推荐：PC/GES/FCI/NOTEARS/LiNGAM/PCMCI 推荐规则

```

`POST /chat/stream` 请求体：

```json
{
  "message": "从数据中自动学习因果结构（DAG）",
  "image_url": "",
  "thread_id": "causal-..."
}
```

### CSV 上传与 profile

```text
POST /api/v1/upload/csv
GET  /api/v1/upload/{thread_id}/status
GET  /api/v1/upload/{thread_id}/profile
```

上传接口为 `multipart/form-data`：

- `thread_id`: 当前前端会话 ID
- `file`: `.csv` 文件

限制与行为：

- 仅接受 `.csv`
- 单文件上限 100 MB
- 编码依次尝试 `utf-8`、`utf-8-sig`、`gbk`
- 文件保存到 `app/uploads/<thread_id>/data.csv`
- profile 保存到 `app/uploads/<thread_id>/profile.json`
- `DELETE /chat/messages` 会同步清理对应会话的上传目录

## 前端交互流程

1. 打开首页，点击输入框左侧回形针上传 CSV。
2. 前端调用 `POST /upload/csv`。
3. 后端生成 `profile.json`。
4. 前端调用 `GET /upload/{thread_id}/profile`，直接渲染数据画像卡片。
5. 前端向 Agent 发送隐藏的 profile 摘要消息。
6. Agent 输出数据概览与下一步建议。
7. 用户选择“从数据中自动学习因果结构（DAG）”。
8. Agent 调用算法推荐工具，输出 `causal-card type=algo-choice`。
9. 前端渲染算法选择卡片，点击后自动发送“我选择 XX 算法进行因果发现”。

## Agent 工具

当前 Agent 挂载的主要工具：

- `causal_knowledge_search`：本地因果知识库检索
- `web_search`：Tavily 网络搜索
- `dag_parse`
- `dag_frontdoor_paths`
- `dag_backdoor_paths`
- `dag_adjustment_set`
- `dag_to_mermaid`
- `summarize_uploaded_dataset`
- `recommend_causal_discovery_algorithms`

其中涉及上传数据的工具通过 `common/runtime_context.py` 中的 `ContextVar` 自动读取当前会话 `thread_id`。LLM 不需要、也不应该手动传入 `thread_id`，避免出现工具读错会话目录的问题。

```
````

当前前端支持：

- `type=profile`：历史兼容；现在 profile 主要由 API 直出渲染
- `type=algo-choice`：因果发现算法选择卡片

## 已知限制

- 当前不会实际运行 PC / FCI / GES / NOTEARS / LiNGAM / PCMCI，只做启发式推荐。
- `LiNGAM` 推荐基于数值列峰度启发式，不等价于真正的非高斯独立性检验。
- `PCMCI` 推荐仅基于时间字段与样本量，未验证严格的时间序列 / 面板结构。
- profile 卡片由前端 API 直出，Agent 文字概览依赖 profile 摘要工具；如服务未重启或上下文异常，需重新上传或刷新会话。
- 生产部署前需要收紧 CORS、上传大小策略、日志脱敏与密钥管理。

## 后续路线

建议后续 Sprint：

1. 接入 `run_causal_discovery`，真实运行 PC / FCI / GES，并输出 Mermaid DAG。
2. 对发现出的边做逐边解释，结合相关性信号与本地知识库来源。
3. 接入 DoWhy / EconML 做因果效应估计与敏感性分析。
4. 增加评测集与自动化回归评估。
5. 优化前端：工具调用面板、暗色模式、可点击引用来源。

