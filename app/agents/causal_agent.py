"""
因果领域智能助手 Agent。

复用原 FastAPI + LangChain Agent 框架：
- 模型：DashScope OpenAI 兼容接口
- 工具：本地因果知识库检索 + Tavily Web Search
- 记忆：LangGraph SqliteSaver（db/causal_agent.db）
- 输出：流式 AIMessageChunk

对外暴露：
- causal_chat(prompt, image, thread_id)：异步流式生成器，供 FastAPI StreamingResponse 使用
- clear_messages(thread_id)：清空指定会话历史
- get_messages(thread_id)：读取指定会话历史
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from common.logger import logger

load_dotenv()


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

web_search = TavilySearch(
    max_results=5,
    topic="general",
)


@tool
def causal_knowledge_search(query: str) -> str:
    """检索本地因果知识库（含 causal基本概念.txt 以及 data/ 下的所有论文 PDF）。

    适用场景：因果推断/因果发现的基础概念、术语、经典算法（PC、FCI、GES、NOTEARS、
    LiNGAM、PCMCI 等）、SCM、do-calculus、DAG、d-separation、混杂/中介/碰撞变量、
    隐藏变量、识别性等问题，应优先调用该工具获取本地资料。

    Args:
        query: 用自然语言描述的检索问题，例如"什么是混杂变量"或"FCI 如何处理隐藏混杂"。

    Returns:
        命中的本地知识片段（按相关性排序）。每个片段前会标注 [来源: 文件名 · p.页码]，
        回答时请把这些来源整理到"参考资料"小节里引用给用户。
        未命中时返回提示信息。
    """
    from data.textload import search_causal_knowledge

    try:
        hits = search_causal_knowledge(query, k=4)
        if not hits:
            return "本地因果知识库未命中，建议改用 web_search 或基于通用知识谨慎回答。"
        parts = []
        for hit in hits:
            src = hit.get("source") or "unknown"
            page = hit.get("page")
            if isinstance(page, int):
                header = f"[来源: {src} · p.{page + 1}]"
            else:
                header = f"[来源: {src}]"
            parts.append(f"{header}\n{hit.get('content', '').strip()}")
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        logger.error(f"本地因果知识检索失败: {exc}")
        return f"本地因果知识库检索失败：{exc}"


# ---------------------------------------------------------------------------
# 模型与记忆
# ---------------------------------------------------------------------------

model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

os.makedirs("db", exist_ok=True)
connection = sqlite3.connect("db/causal_agent.db", check_same_thread=False)
checkpointer = SqliteSaver(connection)
checkpointer.setup()


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

system_prompt = """你是一名因果推断（Causal Inference）与因果发现（Causal Discovery）领域的智能助手，服务对象包括因果学习者、研究人员、数据分析师。

# 核心目标
帮助用户完成以下任务：
1. 解释因果相关概念；
2. 把业务/研究问题建模为因果问题（识别 Treatment、Outcome、Confounder、Mediator、Collider、Instrument）；
3. 解释或诊断用户提供的 DAG，分析路径与偏差；
4. 根据数据特征推荐因果发现算法；
5. 设计因果效应估计方案（识别假设 + 调整集合 + 估计方法）；
6. 在复杂问题上生成结构化的"因果分析报告"。

# 工具使用策略
- 涉及因果推断/因果发现的概念、定义、算法对比、经典理论时，优先调用 `causal_knowledge_search` 检索本地知识库（其中包含 *Causation, Prediction, and Search*、FCI / PC / 隐变量发现等若干篇论文）。
- 当本地知识库不足、用户明确要求最新资料或需要查找具体论文/案例时，调用 `web_search`。
- 概念性回答应优先基于检索到的资料。
- 信息不足时，先追问再调用工具，避免无意义检索。

# 引用规范
- `causal_knowledge_search` 返回的每条片段都带有形如 `[来源: 文件名 · p.页码]` 的标记，必须忠实保留。
- 在回答的最后用 **参考资料** 小节列出本次使用到的来源，格式：`- 文件名 · p.页码`；同一来源多页可合并写。
- 网络搜索结果若有 URL，同样在参考资料里列出（保留原标题与链接）。
- 没有调用任何检索工具时不要凭空捏造文献，可直接说明"基于经典理论 / 经验回答"。

# 行为准则
- 永远不要把相关性直接表述为因果性。
- 当用户问"X 对 Y 的影响"时，必须先识别 Treatment、Outcome、潜在 Confounders，再讨论估计方法。
- 涉及方法推荐时，必须说明该方法的前提假设、适用条件和潜在风险。
- 涉及因果发现时，必须提醒用户：观测数据通常只能给出马尔可夫等价类，不能在没有额外假设下唯一确定真实因果方向。
- 信息不足时，主动追问：变量列表、样本量、数据类型（连续/离散/混合）、是否时间序列、是否存在隐藏混杂、是否有实验或自然实验。
- 对不确定的结论必须明确标注"在 XX 假设下成立"。
- 使用中文回答，专业但易懂，避免无信息量的客套。

# 回答结构（按问题类型选择对应模板）

A. 因果概念解释
1. 简要定义
2. 直观例子
3. 与相近概念的区别
4. 在因果分析中的作用
5. 常见误区

B. 因果问题建模
1. 问题重述
2. 变量角色识别（Treatment / Outcome / Confounder / Mediator / Collider / Instrument）
3. 可能的 DAG 假设（用 "A -> B" 文本形式列出）
4. 应该控制的变量
5. 不应该控制的变量
6. 还需要补充的信息

C. DAG 分析
1. 重述图结构
2. 因果路径
3. 后门路径
4. 是否存在混杂偏差
5. 推荐的调整集合
6. 不应控制的变量及原因

D. 因果发现方法推荐
1. 数据特征判断
2. 候选方法
3. 推荐优先级
4. 每种方法的适用前提
5. 不推荐方法及原因
6. 实施步骤

E. 因果效应估计方案
1. Treatment 与 Outcome
2. 潜在混杂变量
3. 识别假设（如条件可忽略性、SUTVA、正性、平行趋势、工具有效性等）
4. 推荐调整集合
5. 推荐估计方法（如回归调整 / 倾向得分匹配 / IPW / Doubly Robust / DID / IV / RDD / DML）
6. 方法前提与风险

F. 复杂问题输出完整"因果分析报告"
## 1. 问题定义
## 2. 变量角色
## 3. 因果图假设
## 4. 识别策略
## 5. 推荐方法
## 6. 实施步骤
## 7. 假设与风险
## 8. 下一步需要的数据

## 参考资料（如果调用了检索工具）
- 文件名 · p.页码
- ...

请严格遵守以上准则，结构化输出回答。"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[causal_knowledge_search, web_search],
    checkpointer=checkpointer,
    system_prompt=system_prompt,
)


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

async def causal_chat(prompt: str, image: str, thread_id: str):
    """因果领域助手流式对话。

    Args:
        prompt: 用户文本输入。
        image: 可选的图片 URL（例如手绘 DAG 截图）。为空字符串或 None 时按纯文本处理。
        thread_id: 会话 ID，用于 LangGraph SqliteSaver 维持多轮上下文。

    Yields:
        模型增量输出的 token 字符串。
    """
    logger.info(f"[用户] prompt={prompt!r}, image={image!r}, thread_id={thread_id!r}")
    try:
        if not image or image.strip() == "":
            message = HumanMessage(content=prompt)
        else:
            message = HumanMessage(content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt},
            ])

        for chunk, _metadata in agent.stream(
            {"messages": [message]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content
    except Exception as exc:
        logger.error(f"\n[错误] {exc}")
        yield "因果分析助手暂时无法响应，请稍后再试或换种描述方式。"


def clear_messages(thread_id: str) -> None:
    """清空指定会话的历史记录。"""
    logger.info(f"清空会话历史，thread_id={thread_id!r}")
    checkpointer.delete_thread(thread_id)


def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取指定会话的历史消息，返回 [{role, content}, ...]。"""
    logger.info(f"获取会话历史，thread_id={thread_id!r}")
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if not checkpoint:
        return []

    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    result: list[dict[str, str]] = []
    for msg in messages:
        if not msg.content:
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result
