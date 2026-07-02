"""因果发现执行工具（Sprint 4 / P2.T4）。

本模块提供两层 API：

1. 纯函数：
   - run_causal_discovery(csv_path, algorithm, columns=None, options=None, timeout_s=60)
     读取本地 CSV，执行因果发现并统一输出 nodes / edges / mermaid。

2. LangChain @tool 包装：
   - run_uploaded_causal_discovery(algorithm, columns=None, options=None, timeout_s=60)
     自动读取当前会话上传目录中的 data.csv 与 profile.json，执行发现并落盘 dag.json。

设计要点：
- 不把原始 CSV 行暴露给 LLM，只返回聚合后的图结构与告警；
- 会话 thread_id 由 runtime_context 注入，不依赖 LLM 传参；
- 算法在工作线程中执行并支持 timeout，避免阻塞主流程太久。
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from langchain_core.tools import tool

from common.runtime_context import get_current_thread_id
from tools.data_profile import _uploads_root

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SUPPORTED_ALGOS = {"PC", "FCI", "GES"}


def _safe_thread_dir(thread_id: str) -> Optional[Path]:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        return None
    return _uploads_root() / thread_id


def _safe_float(x: Any, n: int = 4) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except Exception:
        return None


def _escape_mermaid_node(name: str) -> str:
    # 保留可读性：ID 用稳定 hash 方案避免特殊字符；label 保留原文
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not safe_id:
        safe_id = "node"
    return f'{safe_id}["{str(name).replace(chr(34), chr(39))}"]'


def _edge_to_key(edge: dict) -> tuple:
    return (
        edge.get("from"),
        edge.get("to"),
        edge.get("type"),
        _safe_float(edge.get("confidence")),
    )


def _merge_edges(edges: list[dict]) -> list[dict]:
    # 1) 先做完全重复去重
    seen = set()
    uniq: list[dict] = []
    for e in edges:
        k = _edge_to_key(e)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    # 2) 冲突消解：同一对节点若同时存在无向/有向，优先保留无向，避免 Mermaid 与边表不一致
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for e in uniq:
        a = str(e.get("from"))
        b = str(e.get("to"))
        pair = tuple(sorted([a, b]))
        by_pair.setdefault(pair, []).append(e)

    out: list[dict] = []
    for _pair, group in by_pair.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        undirected = [g for g in group if g.get("type") == "undirected"]
        if undirected:
            # 只保留第一条无向边，压掉同对节点的定向冲突边
            out.append(undirected[0])
            continue
        # 无无向边时保持原状（例如双向/不确定组合）
        out.extend(group)
    return out


def _normalize_node_name(raw: str, columns: list[str]) -> str:
    """causallearn 默认节点名常为 X1/X2...，映射回真实列名。"""
    m = re.fullmatch(r"X(\d+)", str(raw))
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(columns):
            return str(columns[idx])
    return str(raw)


def _edge_objects_to_unified(edge_objects: list[Any], columns: list[str]) -> list[dict]:
    """把 causallearn 的 Edge 对象统一转换为 {from,to,type,confidence}。"""
    result: list[dict] = []
    for e in edge_objects:
        n1 = _normalize_node_name(str(e.get_node1().get_name()), columns)
        n2 = _normalize_node_name(str(e.get_node2().get_name()), columns)
        ep1 = str(e.get_endpoint1())  # TAIL / ARROW / CIRCLE
        ep2 = str(e.get_endpoint2())
        edge_type = "uncertain"
        a = n1
        b = n2
        if ep1 == "TAIL" and ep2 == "ARROW":
            edge_type = "directed"
            a, b = n1, n2
        elif ep1 == "ARROW" and ep2 == "TAIL":
            edge_type = "directed"
            a, b = n2, n1
        elif ep1 == "TAIL" and ep2 == "TAIL":
            edge_type = "undirected"
            a, b = n1, n2
        elif ep1 == "ARROW" and ep2 == "ARROW":
            edge_type = "bidirected"
            a, b = n1, n2
        else:
            # FCI 常见 CIRCLE-ARROW / CIRCLE-CIRCLE / ARROW-CIRCLE，记作 uncertain
            edge_type = "uncertain"
            a, b = n1, n2
        result.append({
            "from": a,
            "to": b,
            "type": edge_type,
            "confidence": None,
        })
    return _merge_edges(result)


def _edges_to_mermaid(nodes: list[str], edges: list[dict]) -> str:
    if not nodes:
        return "graph LR\n  Empty[\"无可展示的因果图\"]"
    lines = ["graph LR"]
    # 先保证所有节点出现（即使无边）
    for n in nodes:
        lines.append(f"  {_escape_mermaid_node(n)}")
    for e in edges:
        a = str(e.get("from"))
        b = str(e.get("to"))
        edge_type = e.get("type")
        left = re.sub(r"[^A-Za-z0-9_]", "_", a) or "node"
        right = re.sub(r"[^A-Za-z0-9_]", "_", b) or "node"
        if edge_type == "directed":
            conn = "-->"
        elif edge_type == "undirected":
            conn = "---"
        elif edge_type == "bidirected":
            conn = "<-->"
        else:
            conn = "-.->"
        lines.append(f"  {left} {conn} {right}")
    return "\n".join(lines)


def _prepare_numeric_data(
    csv_path: str,
    columns: Optional[list[str]],
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """读取 CSV 并做最小清洗。

    Returns:
      (numeric_df, used_columns, dropped_non_numeric, warnings)
    """
    warnings: list[str] = []
    df = pd.read_csv(csv_path)
    if columns:
        req = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            warnings.append(f"以下列名在数据中不存在，已忽略：{', '.join(missing)}")
        if not req:
            raise ValueError("指定的 columns 全部不存在，无法运行因果发现。")
        df = df[req]

    non_numeric_cols = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric_cols:
        warnings.append(
            "检测到非数值列，已自动排除："
            + ", ".join(non_numeric_cols)
        )
        df = df.drop(columns=non_numeric_cols)

    if df.shape[1] < 2:
        raise ValueError("可用于因果发现的数值列不足 2 个。")

    # 缺失值处理：先用列均值填补，再 drop 仍缺失的行
    if df.isna().any().any():
        warnings.append(
            "检测到缺失值：先使用列均值填补；若仍存在缺失行则已删除。"
        )
        df = df.fillna(df.mean(numeric_only=True))
        before = len(df)
        df = df.dropna(axis=0)
        dropped = before - len(df)
        if dropped > 0:
            warnings.append(f"缺失值处理后额外删除了 {dropped} 行。")

    used = [str(c) for c in df.columns]
    return df, used, non_numeric_cols, warnings


def _guard_data_shape(n_rows: int, n_cols: int) -> Optional[str]:
    if n_rows < 100:
        return f"样本量过小（n={n_rows}<100），不建议直接做因果发现。"
    if n_cols > n_rows:
        return f"变量数大于样本量（cols={n_cols}>rows={n_rows}），不建议直接做因果发现。"
    if n_cols > 50:
        return f"变量数过多（cols={n_cols}>50），建议先做特征筛选后再运行。"
    return None


def _execute_algorithm(
    data: np.ndarray,
    columns: list[str],
    algorithm: str,
    options: dict,
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    algo = algorithm.upper().strip()
    alpha = float(options.get("alpha", 0.05))

    if algo == "PC":
        cg = pc(data, alpha=alpha, show_progress=False)
        graph = cg.G
        edge_objects = graph.get_graph_edges() if hasattr(graph, "get_graph_edges") else []
        return _edge_objects_to_unified(edge_objects, columns), warnings

    if algo == "FCI":
        graph, edge_objects = fci(data, alpha=alpha, show_progress=False)
        # edge_objects 是 Edge 列表，比 graph.matrix 更适合保留不确定端点语义
        return _edge_objects_to_unified(edge_objects or [], columns), warnings

    if algo == "GES":
        # 当前环境 numpy 2.x 下 causallearn 的 GES 可能报 TypeError；
        # 保留接口并返回明确错误，不伪造结果。
        try:
            res = ges(data, score_func=options.get("score_func", "local_score_BIC"))
            graph = res["G"]
            edge_objects = graph.get_graph_edges() if hasattr(graph, "get_graph_edges") else []
            return _edge_objects_to_unified(edge_objects, columns), warnings
        except Exception as exc:
            raise RuntimeError(
                "GES 在当前环境执行失败（常见于 causallearn 与 numpy 版本兼容问题）。"
                f" 原始错误：{exc}"
            ) from exc

    if algo in {"NOTEARS", "LINGAM", "PCMCI"}:
        raise ValueError(
            f"当前版本暂不支持执行 {algo}；目前仅支持 PC / FCI / GES 的真实运行。"
        )

    raise ValueError(
        f"不支持的算法：{algorithm}。当前仅支持 PC / FCI / GES。"
    )


def _run_internal(
    csv_path: str,
    algorithm: str,
    columns: Optional[list[str]],
    options: Optional[dict],
) -> dict:
    started = time.perf_counter()
    opts = dict(options or {})
    warnings: list[str] = []

    df, used_cols, _dropped_non_numeric, prep_warnings = _prepare_numeric_data(
        csv_path=csv_path,
        columns=columns,
    )
    warnings.extend(prep_warnings)

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    block_reason = _guard_data_shape(n_rows, n_cols)
    if block_reason:
        elapsed = time.perf_counter() - started
        warnings.append(block_reason)
        return {
            "algorithm": algorithm.upper(),
            "nodes": used_cols,
            "edges": [],
            "mermaid": _edges_to_mermaid(used_cols, []),
            "elapsed_s": round(elapsed, 4),
            "warnings": warnings,
            "options": opts,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "error": block_reason,
        }

    edges, algo_warnings = _execute_algorithm(
        data=df.to_numpy(dtype=float),
        columns=used_cols,
        algorithm=algorithm,
        options=opts,
    )
    warnings.extend(algo_warnings)

    elapsed = time.perf_counter() - started
    return {
        "algorithm": algorithm.upper(),
        "nodes": used_cols,
        "edges": edges,
        "mermaid": _edges_to_mermaid(used_cols, edges),
        "elapsed_s": round(elapsed, 4),
        "warnings": warnings,
        "options": opts,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "error": None,
    }


def run_causal_discovery(
    csv_path: str,
    algorithm: str,
    columns: list[str] | None = None,
    options: dict | None = None,
    timeout_s: int = 60,
) -> dict:
    """执行因果发现并统一返回结构化结果。"""
    if not Path(csv_path).exists():
        return {
            "algorithm": str(algorithm).upper(),
            "nodes": [],
            "edges": [],
            "mermaid": "graph LR\n  Empty[\"找不到 CSV 文件\"]",
            "elapsed_s": 0.0,
            "warnings": [f"CSV 文件不存在：{csv_path}"],
            "options": dict(options or {}),
            "error": "csv_not_found",
        }

    if timeout_s <= 0:
        timeout_s = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(
            _run_internal,
            csv_path,
            algorithm,
            columns,
            options,
        )
        try:
            return fut.result(timeout=float(timeout_s))
        except concurrent.futures.TimeoutError:
            return {
                "algorithm": str(algorithm).upper(),
                "nodes": [],
                "edges": [],
                "mermaid": "graph LR\n  Timeout[\"算法执行超时\"]",
                "elapsed_s": float(timeout_s),
                "warnings": [
                    f"算法执行超时（>{timeout_s}s）。建议减少变量数、先做特征筛选，或优先尝试 PC。"
                ],
                "options": dict(options or {}),
                "error": "timeout",
            }
        except Exception as exc:
            return {
                "algorithm": str(algorithm).upper(),
                "nodes": [],
                "edges": [],
                "mermaid": "graph LR\n  Error[\"算法执行失败\"]",
                "elapsed_s": 0.0,
                "warnings": [f"算法执行失败：{exc}"],
                "options": dict(options or {}),
                "error": str(exc),
            }


def _format_discovery_markdown(payload: dict) -> str:
    algo = payload.get("algorithm")
    n_rows = payload.get("n_rows", "?")
    n_cols = payload.get("n_cols", "?")
    edges = payload.get("edges") or []
    warns = payload.get("warnings") or []
    err = payload.get("error")
    elapsed = payload.get("elapsed_s")

    lines: list[str] = []
    lines.append(f"- 算法：{algo}")
    lines.append(f"- 数据规模：{n_rows} 行 × {n_cols} 列")
    lines.append(f"- 发现边数：{len(edges)}")
    if isinstance(elapsed, (int, float)):
        lines.append(f"- 耗时：{elapsed:.2f}s")
    if err:
        lines.append(f"- 执行状态：失败（{err}）")
    if warns:
        lines.append("- 关键警告：")
        for w in warns[:8]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


@tool
def run_uploaded_causal_discovery(
    algorithm: str,
    columns: list[str] | None = None,
    options: dict | None = None,
    timeout_s: int = 60,
) -> str:
    """对当前会话已上传数据执行因果发现（PC / FCI / GES）。

    该工具会自动读取当前会话上下文中的 thread_id（无需 LLM 传参），从
    ``app/uploads/<thread_id>/data.csv`` 读取数据，并把结果写入
    ``app/uploads/<thread_id>/dag.json``。

    返回 Markdown 摘要 + 严格 JSON 代码块，可直接用于 Mermaid 渲染和边解释。
    """
    thread_id = get_current_thread_id()
    if thread_id is None:
        return (
            "未能读取当前会话上下文。该工具应由聊天接口在会话中调用；"
            "若是直接测试，请在测试中通过 bind_thread_id 绑定会话 ID。"
        )
    thread_dir = _safe_thread_dir(thread_id)
    if thread_dir is None:
        return "thread_id 不合法，无法读取当前会话数据。"

    csv_path = thread_dir / "data.csv"
    profile_path = thread_dir / "profile.json"
    if not csv_path.exists():
        return "当前会话未找到 data.csv，请先上传 CSV。"
    if not profile_path.exists():
        return "当前会话未找到 profile.json，请先完成数据画像。"

    payload = run_causal_discovery(
        csv_path=str(csv_path),
        algorithm=algorithm,
        columns=columns,
        options=options,
        timeout_s=timeout_s,
    )

    dag_path = thread_dir / "dag.json"
    to_save = dict(payload)
    to_save["created_at"] = time.time()
    try:
        dag_path.write_text(
            json.dumps(to_save, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        payload.setdefault("warnings", []).append(f"dag.json 写入失败：{exc}")

    md = _format_discovery_markdown(payload)
    json_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{md}\n\n```json\n{json_block}\n```"

