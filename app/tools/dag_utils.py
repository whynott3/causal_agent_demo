"""基于 networkx 的轻量 DAG 推理工具集。

本模块提供两层 API：

1. 底层纯函数（便于单元测试）：
   - parse_dag_text(text)
   - find_backdoor_paths(nodes, edges, treatment, outcome)
   - find_frontdoor_paths(nodes, edges, treatment, outcome)
   - suggest_adjustment_set(nodes, edges, treatment, outcome)
   - to_mermaid(nodes, edges, direction, highlight)

2. LangChain @tool 包装（直接挂到 Agent，给 LLM 调用）：
   - dag_parse
   - dag_frontdoor_paths
   - dag_backdoor_paths
   - dag_adjustment_set
   - dag_to_mermaid

设计原则：
- 所有底层函数同步、纯函数，不打网络、不起子进程；
- @tool 层把入参 / 出参规整为字符串 / Markdown，便于 LLM 与前端 mermaid 联动。
"""

from __future__ import annotations

import re
from typing import Optional

import networkx as nx
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 常量与正则
# ---------------------------------------------------------------------------

_PATH_LIMIT = 200
_CYCLE_LIMIT = 5
_ARROW_RE = re.compile(r"\s*(?:->|→|-->)\s*")
_LINE_SEPS_RE = re.compile(r"[，,；;]")


# ---------------------------------------------------------------------------
# parse_dag_text
# ---------------------------------------------------------------------------

def parse_dag_text(text: str) -> dict:
    """把自由文本 DAG 解析为图结构。

    支持：
    - 边写法：``A -> B`` / ``A→B`` / ``A->B`` / ``A --> B``，前后空白随意；
    - 行内多段：``A -> B -> C`` 会被拆成 ``(A,B)`` 与 ``(B,C)`` 两条边；
    - 行分隔：换行、``，``、``,``、``；``、``;``；
    - 忽略空行、以 ``#`` 开头的注释行；
    - 节点名保留原始字符串（含中文），仅 strip 首尾空白。

    Returns:
        ``{
            'nodes': list[str],                # 出现顺序去重
            'edges': list[tuple[str, str]],    # 解析顺序
            'errors': list[str],               # 无法解析的原始行
            'has_cycle': bool,
            'cycles': list[list[str]],         # 若有，列出前 5 个环
        }``
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    normalized = _LINE_SEPS_RE.sub("\n", text)
    raw_lines = normalized.split("\n")

    nodes_order: list[str] = []
    nodes_seen: set[str] = set()
    edges: list[tuple[str, str]] = []
    errors: list[str] = []

    def _add_node(name: str) -> None:
        if name and name not in nodes_seen:
            nodes_seen.add(name)
            nodes_order.append(name)

    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in _ARROW_RE.split(line)]
        if len(parts) < 2 or any(not p for p in parts):
            errors.append(raw)
            continue

        for name in parts:
            _add_node(name)
        for a, b in zip(parts, parts[1:]):
            edges.append((a, b))

    # 兜底：删除自环（A -> A）并记录为错误
    cleaned_edges: list[tuple[str, str]] = []
    for a, b in edges:
        if a == b:
            errors.append(f"自环已忽略: {a} -> {b}")
            continue
        cleaned_edges.append((a, b))
    edges = cleaned_edges

    g = nx.DiGraph()
    g.add_nodes_from(nodes_order)
    g.add_edges_from(edges)

    has_cycle = not nx.is_directed_acyclic_graph(g)
    cycles: list[list[str]] = []
    if has_cycle:
        try:
            for cyc in nx.simple_cycles(g):
                cycles.append(list(cyc))
                if len(cycles) >= _CYCLE_LIMIT:
                    break
        except Exception:
            cycles = []

    return {
        "nodes": nodes_order,
        "edges": edges,
        "errors": errors,
        "has_cycle": has_cycle,
        "cycles": cycles,
    }


# ---------------------------------------------------------------------------
# 图构建辅助
# ---------------------------------------------------------------------------

def _build_dag(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("图中存在环路，无法在非 DAG 上分析因果路径与调整集合。")
    return g


def _check_endpoints(g: nx.DiGraph, treatment: str, outcome: str) -> None:
    if treatment not in g:
        raise ValueError(f"Treatment 节点不存在于图中: {treatment!r}")
    if outcome not in g:
        raise ValueError(f"Outcome 节点不存在于图中: {outcome!r}")
    if treatment == outcome:
        raise ValueError("Treatment 与 Outcome 不能相同。")


# ---------------------------------------------------------------------------
# find_backdoor_paths
# ---------------------------------------------------------------------------

def find_backdoor_paths(
    nodes: list[str],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
) -> list[list[str]]:
    """返回从 treatment 到 outcome 的所有后门路径。

    后门路径定义：treatment 与 outcome 之间在底层无向图上的简单路径，
    且首边方向指向 treatment（即 ``path[1] -> treatment`` 在 DiGraph 中存在）。
    """
    g = _build_dag(nodes, edges)
    _check_endpoints(g, treatment, outcome)

    u = g.to_undirected()
    result: list[list[str]] = []
    truncated = False
    for path in nx.all_simple_paths(u, treatment, outcome):
        if len(path) < 2:
            continue
        if g.has_edge(path[1], treatment):
            result.append(list(path))
            if len(result) >= _PATH_LIMIT:
                truncated = True
                break

    if truncated:
        # 用约定的「截断标记」字符串作为最后一项，便于上层提示
        result.append([f"[TRUNCATED] 路径数已达 {_PATH_LIMIT} 条上限，剩余未列出"])
    return result


# ---------------------------------------------------------------------------
# find_frontdoor_paths
# ---------------------------------------------------------------------------

def find_frontdoor_paths(
    nodes: list[str],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
) -> list[list[str]]:
    """返回从 treatment 到 outcome 的所有前门（因果）路径，即所有有向简单路径。"""
    g = _build_dag(nodes, edges)
    _check_endpoints(g, treatment, outcome)

    result: list[list[str]] = []
    truncated = False
    for path in nx.all_simple_paths(g, treatment, outcome):
        result.append(list(path))
        if len(result) >= _PATH_LIMIT:
            truncated = True
            break

    if truncated:
        result.append([f"[TRUNCATED] 路径数已达 {_PATH_LIMIT} 条上限，剩余未列出"])
    return result


def _collect_mediators(
    g: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> list[str]:
    """收集所有前门路径上的中介节点（排除 treatment 与 outcome）。"""
    mediators: set[str] = set()
    try:
        for path in nx.all_simple_paths(g, treatment, outcome):
            for node in path[1:-1]:
                mediators.add(node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return [n for n in g.nodes if n in mediators]


# ---------------------------------------------------------------------------
# suggest_adjustment_set
# ---------------------------------------------------------------------------

def _is_d_separator(
    g: nx.DiGraph,
    x: set[str],
    y: set[str],
    z: set[str],
) -> bool:
    """networkx 跨版本的 d-separation 检查。"""
    try:
        return bool(nx.is_d_separator(g, x, y, z))
    except AttributeError:
        try:
            return bool(nx.d_separated(g, x, y, z))  # type: ignore[attr-defined]
        except Exception:
            return False
    except Exception:
        return False


def _is_backdoor_sufficient(
    g: nx.DiGraph,
    treatment: str,
    outcome: str,
    z: set[str],
) -> bool:
    """后门准则验证：在删除 treatment 全部出边后，T 与 Y 在给定 Z 时 d-分离。"""
    g_bar = g.copy()
    g_bar.remove_edges_from(list(g_bar.out_edges(treatment)))
    return _is_d_separator(g_bar, {treatment}, {outcome}, set(z))


def suggest_adjustment_set(
    nodes: list[str],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
) -> dict:
    """基于后门准则启发式给出充分调整集合。

    候选集合：
        ``(ancestors(T) ∪ ancestors(Y)) - {T, Y} - descendants(T)``
    再用 d-separation 在 ``G̅_T``（删除 T 全部出边）中验证 T ⊥ Y | Z。

    若启发式不充分，尝试 ``parents(T)`` 兜底；都失败则返回空集 + warning。

    Returns:
        ``{
            'adjustment_set': list[str],          # 排序稳定（按 nodes 顺序）
            'is_sufficient': bool,
            'blocked_backdoor_paths': int,
            'remaining_backdoor_paths': list[list[str]],
            'forbidden_nodes': list[str],
            'mediators': list[str],
            'warnings': list[str],
        }``
    """
    g = _build_dag(nodes, edges)
    _check_endpoints(g, treatment, outcome)

    warnings: list[str] = []
    node_order = {n: i for i, n in enumerate(nodes)}

    descendants_t = set(nx.descendants(g, treatment))
    ancestors_t = set(nx.ancestors(g, treatment))
    ancestors_y = set(nx.ancestors(g, outcome))

    candidate = (ancestors_t | ancestors_y) - {treatment, outcome} - descendants_t

    mediators = [n for n in _collect_mediators(g, treatment, outcome) if n in descendants_t]

    forbidden = set(descendants_t) | set(mediators)
    forbidden.discard(outcome)
    forbidden_sorted = sorted(forbidden, key=lambda n: node_order.get(n, 1 << 30))

    chosen: set[str] = candidate
    is_sufficient = _is_backdoor_sufficient(g, treatment, outcome, chosen)

    if not is_sufficient:
        parents_t = set(g.predecessors(treatment))
        parents_t -= {outcome}
        if parents_t and _is_backdoor_sufficient(g, treatment, outcome, parents_t):
            chosen = parents_t
            warnings.append(
                "祖先并集启发式未通过 d-分离验证，已退化为 treatment 的父节点集合。"
            )
            is_sufficient = True
        else:
            warnings.append(
                "无法用启发式找到充分调整集合，请检查是否存在未观测混杂或前门 / IV 路径。"
            )
            chosen = set()
            is_sufficient = False

    adjustment_sorted = sorted(chosen, key=lambda n: node_order.get(n, 1 << 30))

    all_backdoor = find_backdoor_paths(nodes, edges, treatment, outcome)
    real_backdoor: list[list[str]] = [p for p in all_backdoor if not (p and isinstance(p[0], str) and p[0].startswith("[TRUNCATED]"))]

    remaining: list[list[str]] = []
    if chosen:
        for path in real_backdoor:
            middle = set(path[1:-1])
            if not (middle & set(chosen)):
                remaining.append(path)
    else:
        remaining = list(real_backdoor)

    blocked = len(real_backdoor) - len(remaining)

    return {
        "adjustment_set": adjustment_sorted,
        "is_sufficient": bool(is_sufficient),
        "blocked_backdoor_paths": blocked,
        "remaining_backdoor_paths": remaining,
        "forbidden_nodes": forbidden_sorted,
        "mediators": [n for n in nodes if n in set(mediators)],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# to_mermaid
# ---------------------------------------------------------------------------

_ALLOWED_DIRECTIONS = {"LR", "TD", "TB", "RL", "BT"}


def to_mermaid(
    nodes: list[str],
    edges: list[tuple[str, str]],
    direction: str = "LR",
    highlight: Optional[dict] = None,
) -> str:
    """渲染为 Mermaid ``graph {direction}`` 文本块（不含 ``` 围栏）。

    - 节点 id 使用稳定索引（``n0``、``n1``、…），显示名走 ``["显示名"]`` 语法，
      规避中文 / 空格 / 特殊字符在 mermaid id 里报错；
    - highlight 可选：
        ``{'treatment': str, 'outcome': str,
           'adjustment': list[str], 'mediators': list[str]}``
      不同角色用不同 classDef 高亮（黄 / 绿 / 蓝 / 紫）。
    """
    direction = direction if direction in _ALLOWED_DIRECTIONS else "LR"

    lines: list[str] = [f"graph {direction}"]

    name_to_id: dict[str, str] = {}
    for i, name in enumerate(nodes):
        nid = f"n{i}"
        name_to_id[name] = nid
        safe = name.replace('"', "'")
        lines.append(f'    {nid}["{safe}"]')

    for a, b in edges:
        if a in name_to_id and b in name_to_id:
            lines.append(f"    {name_to_id[a]} --> {name_to_id[b]}")

    if highlight:
        treat = highlight.get("treatment")
        outc = highlight.get("outcome")
        adj = highlight.get("adjustment") or []
        med = highlight.get("mediators") or []

        used = False
        treat_id = name_to_id.get(treat) if isinstance(treat, str) else None
        outc_id = name_to_id.get(outc) if isinstance(outc, str) else None
        adj_ids = [name_to_id[n] for n in adj if isinstance(n, str) and n in name_to_id and n not in (treat, outc)]
        med_ids = [name_to_id[n] for n in med if isinstance(n, str) and n in name_to_id and n not in (treat, outc)]

        if treat_id:
            lines.append("    classDef cTreat fill:#fde68a,stroke:#f59e0b,color:#0f172a;")
            lines.append(f"    class {treat_id} cTreat;")
            used = True
        if outc_id:
            lines.append("    classDef cOut fill:#bbf7d0,stroke:#10b981,color:#0f172a;")
            lines.append(f"    class {outc_id} cOut;")
            used = True
        if adj_ids:
            lines.append("    classDef cAdj fill:#bae6fd,stroke:#0ea5e9,color:#0f172a;")
            lines.append(f"    class {','.join(adj_ids)} cAdj;")
            used = True
        if med_ids:
            lines.append("    classDef cMed fill:#e9d5ff,stroke:#a855f7,color:#0f172a;")
            lines.append(f"    class {','.join(med_ids)} cMed;")
            used = True
        if not used:
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 渲染辅助：把节点序列还原为带方向的 Markdown 字符串
# ---------------------------------------------------------------------------

def _render_directed_segment(
    edge_set: set[tuple[str, str]],
    a: str,
    b: str,
) -> str:
    """根据原图中 (a, b) / (b, a) 的方向决定渲染为 ``a -> b`` 还是 ``a <- b``。"""
    if (a, b) in edge_set and (b, a) not in edge_set:
        return f"{a} -> {b}"
    if (b, a) in edge_set and (a, b) not in edge_set:
        return f"{a} <- {b}"
    if (a, b) in edge_set and (b, a) in edge_set:
        return f"{a} <-> {b}"
    return f"{a} -- {b}"


def _render_path(edge_set: set[tuple[str, str]], path: list[str], directed_only: bool) -> str:
    if not path:
        return ""
    if directed_only:
        return " -> ".join(path)
    parts = [path[0]]
    for a, b in zip(path, path[1:]):
        seg = _render_directed_segment(edge_set, a, b)
        # seg 已经是 "a -> b" / "a <- b"，我们只需要把 "<arrow> b" 拼到末尾
        arrow_part = seg[len(a):].lstrip()
        parts.append(arrow_part)
    return " ".join(parts)


def _fence_mermaid(body: str) -> str:
    return "```mermaid\n" + body + "\n```"


def _format_node_list(items: list[str], empty: str = "—") -> str:
    return "、".join(items) if items else empty


# ---------------------------------------------------------------------------
# @tool 包装层
# ---------------------------------------------------------------------------

@tool
def dag_parse(text: str) -> str:
    """解析自然语言 DAG 描述并返回结构化摘要 + Mermaid 代码块。

    支持的边写法：``A -> B`` / ``A→B`` / ``A->B`` / ``A -> B -> C``。
    支持的分隔：换行、``，``、``,``、``；``、``;``。

    Args:
        text: 用户给出的 DAG 文本，例如
            "教育水平 -> 收入；家庭背景 -> 教育水平；家庭背景 -> 收入"。

    Returns:
        Markdown 文本：节点列表、边列表、是否存在环、解析失败的行，
        以及一个 ```mermaid``` 代码块（前端会自动渲染成图）。
    """
    info = parse_dag_text(text)
    lines: list[str] = []
    lines.append(f"- 节点（{len(info['nodes'])} 个）：{_format_node_list(info['nodes'])}")
    if info["edges"]:
        edge_strs = [f"{a} -> {b}" for a, b in info["edges"]]
        lines.append(f"- 边（{len(info['edges'])} 条）：{ '; '.join(edge_strs) }")
    else:
        lines.append("- 边：—")
    if info["errors"]:
        lines.append(f"- 解析失败的行（{len(info['errors'])} 条）：{ '; '.join(info['errors']) }")
    if info["has_cycle"]:
        cyc_strs = [" -> ".join(c + [c[0]]) for c in info["cycles"]]
        lines.append(f"- ⚠️ 检测到环路（前 {len(info['cycles'])} 个）：{ '; '.join(cyc_strs) }")
        lines.append("- 由于图中存在环，无法继续做因果路径 / 调整集合分析，请先修正图。")
    else:
        lines.append("- 是否 DAG：是")

    mermaid_body = to_mermaid(info["nodes"], info["edges"], direction="LR")
    lines.append("")
    lines.append(_fence_mermaid(mermaid_body))
    return "\n".join(lines)


@tool
def dag_frontdoor_paths(text: str, treatment: str, outcome: str) -> str:
    """在用户给出的 DAG 文本上查找 treatment 到 outcome 的全部前门路径
    （即所有有向简单路径，对应系统提示词模板 C 的「前门路径 / 因果路径」小节）。

    Args:
        text: DAG 文本。
        treatment: 处理变量名（需与 DAG 文本中的节点名完全一致）。
        outcome: 结果变量名（需与 DAG 文本中的节点名完全一致）。

    Returns:
        Markdown 列表，每条路径以 "A -> B -> C" 形式渲染，并附一行
        "本图共识别 N 条前门路径；中介节点：{...}"。
    """
    info = parse_dag_text(text)
    if info["has_cycle"]:
        return "图中存在环路，无法分析前门路径，请先修正 DAG。"
    try:
        paths = find_frontdoor_paths(info["nodes"], info["edges"], treatment, outcome)
    except ValueError as exc:
        return f"前门路径分析失败：{exc}"

    truncated_msg: Optional[str] = None
    real_paths: list[list[str]] = []
    for p in paths:
        if p and isinstance(p[0], str) and p[0].startswith("[TRUNCATED]"):
            truncated_msg = p[0]
            continue
        real_paths.append(p)

    g = nx.DiGraph()
    g.add_nodes_from(info["nodes"])
    g.add_edges_from(info["edges"])
    mediators: list[str] = []
    seen: set[str] = set()
    for p in real_paths:
        for n in p[1:-1]:
            if n not in seen:
                seen.add(n)
                mediators.append(n)

    if not real_paths:
        return f"本图共识别 0 条前门路径；treatment={treatment} 到 outcome={outcome} 无有向路径。"

    lines = [f"本图共识别 {len(real_paths)} 条前门路径；中介节点：{_format_node_list(mediators)}"]
    for p in real_paths:
        lines.append(f"- {' -> '.join(p)}")
    if truncated_msg:
        lines.append(f"- ⚠️ {truncated_msg}")
    return "\n".join(lines)


@tool
def dag_backdoor_paths(text: str, treatment: str, outcome: str) -> str:
    """在用户给出的 DAG 文本上查找 treatment 到 outcome 的全部后门路径。

    Args:
        text: DAG 文本。
        treatment: 处理变量名。
        outcome: 结果变量名。

    Returns:
        Markdown 列表，每条路径用 "A <- B -> C" 形式渲染（每段边按原始
        方向选用 ``->`` / ``<-``），并附一行 "本图共识别 N 条后门路径"。
    """
    info = parse_dag_text(text)
    if info["has_cycle"]:
        return "图中存在环路，无法分析后门路径，请先修正 DAG。"
    try:
        paths = find_backdoor_paths(info["nodes"], info["edges"], treatment, outcome)
    except ValueError as exc:
        return f"后门路径分析失败：{exc}"

    truncated_msg: Optional[str] = None
    real_paths: list[list[str]] = []
    for p in paths:
        if p and isinstance(p[0], str) and p[0].startswith("[TRUNCATED]"):
            truncated_msg = p[0]
            continue
        real_paths.append(p)

    if not real_paths:
        return f"本图共识别 0 条后门路径；treatment={treatment} 到 outcome={outcome} 无后门偏差。"

    edge_set = set(info["edges"])
    lines = [f"本图共识别 {len(real_paths)} 条后门路径"]
    for p in real_paths:
        lines.append(f"- {_render_path(edge_set, p, directed_only=False)}")
    if truncated_msg:
        lines.append(f"- ⚠️ {truncated_msg}")
    return "\n".join(lines)


@tool
def dag_adjustment_set(text: str, treatment: str, outcome: str) -> str:
    """在用户给出的 DAG 文本上推荐一个满足后门准则的调整集合，并说明：
    是否充分（is_sufficient）、未被阻断的后门路径、不可控制的节点
    （treatment 的后代 + 中介节点；必须明确指出哪些是中介）。

    Args:
        text: DAG 文本。
        treatment: 处理变量名。
        outcome: 结果变量名。

    Returns:
        Markdown 报告 + 高亮 treatment / outcome / adjustment / mediators 的
        ```mermaid``` 代码块。第一版采用 ancestor-based 启发式：
        ``(ancestors(T) ∪ ancestors(Y)) - {T,Y} - descendants(T)``，
        d-separation 验证不通过时退化为 ``parents(T)`` 兜底。
    """
    info = parse_dag_text(text)
    if info["has_cycle"]:
        return "图中存在环路，无法推荐调整集合，请先修正 DAG。"
    try:
        result = suggest_adjustment_set(info["nodes"], info["edges"], treatment, outcome)
    except ValueError as exc:
        return f"调整集合推荐失败：{exc}"

    edge_set = set(info["edges"])
    lines: list[str] = []
    lines.append(
        f"- 推荐调整集合：{_format_node_list(result['adjustment_set'])}"
        f"（is_sufficient={result['is_sufficient']}）"
    )
    lines.append(f"- 已阻断后门路径数：{result['blocked_backdoor_paths']}")
    if result["remaining_backdoor_paths"]:
        lines.append("- ⚠️ 仍未阻断的后门路径：")
        for p in result["remaining_backdoor_paths"]:
            lines.append(f"  - {_render_path(edge_set, p, directed_only=False)}")
    else:
        lines.append("- ✅ 所有后门路径均被阻断。")

    if result["mediators"]:
        lines.append(
            f"- 中介节点（不应控制，否则会阻断因果效应）：{_format_node_list(result['mediators'])}"
        )
    else:
        lines.append("- 中介节点：—（本图无中介）")

    if result["forbidden_nodes"]:
        lines.append(
            f"- 不可控制的节点（treatment 的后代 ∪ 中介）：{_format_node_list(result['forbidden_nodes'])}"
        )

    if result["warnings"]:
        lines.append("- 提示：")
        for w in result["warnings"]:
            lines.append(f"  - {w}")

    highlight = {
        "treatment": treatment,
        "outcome": outcome,
        "adjustment": result["adjustment_set"],
        "mediators": result["mediators"],
    }
    mermaid_body = to_mermaid(info["nodes"], info["edges"], direction="LR", highlight=highlight)
    lines.append("")
    lines.append(_fence_mermaid(mermaid_body))
    return "\n".join(lines)


@tool
def dag_to_mermaid(text: str, direction: str = "LR") -> str:
    """把用户给出的 DAG 文本转成一个 ```mermaid``` 代码块，方便前端画图。

    Args:
        text: DAG 文本。
        direction: 仅接受 ``"LR"``（左→右）或 ``"TD"``（上→下），其它值会被纠正为 LR。

    Returns:
        一个 ```mermaid``` 代码块；若图中有环，会在代码块前附环路提示。
    """
    info = parse_dag_text(text)
    direction = direction if direction in ("LR", "TD") else "LR"
    body = to_mermaid(info["nodes"], info["edges"], direction=direction)
    prefix = ""
    if info["has_cycle"]:
        cyc_strs = [" -> ".join(c + [c[0]]) for c in info["cycles"]]
        prefix = f"⚠️ 检测到环路：{'; '.join(cyc_strs)}\n\n"
    return prefix + _fence_mermaid(body)
