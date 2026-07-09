"""因果效应估计工具（Sprint 5 / P1.1）。

两层 API：

1) 纯函数：
   - causal_effect_estimate(...)
     读取本地 CSV，基于 DoWhy 进行识别、估计与敏感性分析。

2) LangChain @tool 包装：
   - estimate_uploaded_causal_effect(...)
     自动读取当前会话 uploads 目录中的 data.csv，并落盘 effect.json。

设计约束：
- 不向 LLM 返回原始行，仅返回聚合指标与统计结果；
- thread_id 由 runtime_context 注入，不依赖 LLM 传参；
- 算法执行放到工作线程并支持 timeout，避免阻塞主流程。
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from common.runtime_context import get_current_thread_id
from tools.data_profile import _uploads_root

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SUPPORTED_METHODS = {
    "backdoor.linear_regression",
    "backdoor.propensity_score_matching",
}
_METHOD_CATALOG: list[dict[str, str]] = [
    {
        "method": "backdoor.linear_regression",
        "label": "后门准则 · 线性回归调整",
        "reason": "适合连续型 treatment/outcome，速度快、可解释性强，为默认推荐。",
    },
    {
        "method": "backdoor.propensity_score_matching",
        "label": "后门准则 · 倾向评分匹配",
        "reason": "适合二值/近似二值 treatment；连续 treatment 场景不适用。",
    },
]


def _normalize_effect_error_message(exc: Exception | str) -> str:
    msg = str(exc)
    if "d_separated" in msg and "networkx.algorithms" in msg:
        return (
            "DoWhy 与当前 networkx 版本存在兼容问题"
            "（缺少 d_separated API）。请升级/降级依赖后重试。"
        )
    if msg == "0" or "KeyError: 0" in msg:
        return (
            "DoWhy 回归估计与当前 pandas 版本存在兼容问题"
            "（regression_estimator 使用 params[0] 触发 KeyError）。"
            " 建议临时改用 pandas<3，或切换估计方法。"
        )
    return msg


def _patch_networkx_for_dowhy() -> None:
    """兼容 dowhy 在新 networkx 上对 d_separated 的旧引用。"""
    try:
        import networkx as nx  # type: ignore
    except Exception:
        return
    alg = getattr(nx, "algorithms", None)
    if alg is None:
        return
    if hasattr(alg, "d_separated"):
        return

    # networkx 新版通常提供 d_separation.is_d_separator
    ds = getattr(alg, "d_separation", None)
    is_d_sep = getattr(ds, "is_d_separator", None) if ds is not None else None
    if callable(is_d_sep):
        setattr(alg, "d_separated", is_d_sep)


def _safe_thread_dir(thread_id: str) -> Optional[Path]:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        return None
    return _uploads_root() / thread_id


def _safe_float(x: Any, n: int = 6) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except Exception:
        return None


def _build_dowhy_graph(
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> str:
    edges: list[str] = [f"{treatment} -> {outcome}"]
    for c in confounders:
        edges.append(f"{c} -> {treatment}")
        edges.append(f"{c} -> {outcome}")
    edge_part = "; ".join(edges)
    return f"digraph {{ {edge_part}; }}"


def _extract_ci_bounds(estimate: Any) -> tuple[Optional[float], Optional[float]]:
    try:
        ci = estimate.get_confidence_intervals()
        if ci is None:
            return None, None
        if hasattr(ci, "values"):
            vals = ci.values
            if len(vals) > 0 and len(vals[0]) >= 2:
                return _safe_float(vals[0][0]), _safe_float(vals[0][1])
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            return _safe_float(ci[0]), _safe_float(ci[1])
    except Exception:
        return None, None
    return None, None


def _extract_p_value(estimate: Any) -> Optional[float]:
    try:
        sig = estimate.test_stat_significance()
        if isinstance(sig, dict):
            if "p_value" in sig:
                return _safe_float(sig["p_value"])
            # 某些版本返回 {'p_value': {'...': ...}}
            for v in sig.values():
                if isinstance(v, dict) and "p_value" in v:
                    return _safe_float(v["p_value"])
                if isinstance(v, (list, tuple)) and v:
                    pv = _safe_float(v[-1])
                    if pv is not None:
                        return pv
        if isinstance(sig, (list, tuple)) and sig:
            return _safe_float(sig[-1])
    except Exception:
        return None
    return None


def _refute_summary(
    name: str,
    ref_obj: Any,
    point_estimate: Optional[float],
) -> dict[str, Any]:
    text = str(ref_obj).strip().replace("\n", " ")
    if len(text) > 240:
        text = text[:237] + "..."

    passed = True
    new_eff = _safe_float(getattr(ref_obj, "new_effect", None))
    if new_eff is not None and point_estimate is not None:
        tol = max(0.1, abs(point_estimate) * 0.5)
        passed = abs(new_eff - point_estimate) <= tol

    pv = _safe_float(getattr(ref_obj, "p_value", None))
    if pv is not None:
        passed = pv > 0.05

    return {
        "name": name,
        "passed": bool(passed),
        "summary": text or f"{name} completed.",
    }


def _run_effect_internal(
    csv_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str] | None,
    method: str,
    options: dict | None,
) -> dict:
    started = time.perf_counter()
    warnings: list[str] = []
    opts = dict(options or {})

    if method not in _SUPPORTED_METHODS:
        elapsed = time.perf_counter() - started
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": list(confounders or []),
            "method": method,
            "n": 0,
            "n_rows_used": 0,
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "refute_results": [],
            "warnings": warnings,
            "error": (
                f"当前仅支持方法：{', '.join(sorted(_SUPPORTED_METHODS))}。"
                " 你可以改用 backdoor.linear_regression。"
            ),
            "elapsed_s": round(elapsed, 4),
        }

    df = pd.read_csv(csv_path)
    total_rows = int(len(df))
    if treatment not in df.columns:
        raise ValueError(f"treatment 列不存在：{treatment}")
    if outcome not in df.columns:
        raise ValueError(f"outcome 列不存在：{outcome}")

    confs = [str(c) for c in (confounders or []) if str(c).strip()]
    # 去重并去除与 treatment/outcome 重复项
    uniq_confs: list[str] = []
    seen = set()
    for c in confs:
        if c in {treatment, outcome}:
            continue
        if c not in seen:
            seen.add(c)
            uniq_confs.append(c)

    missing_conf = [c for c in uniq_confs if c not in df.columns]
    if missing_conf:
        raise ValueError(f"confounders 中存在缺失列：{', '.join(missing_conf)}")

    cols = [treatment, outcome] + uniq_confs
    work = df[cols].copy()

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(work[c])]
    if non_numeric:
        raise ValueError(
            "DoWhy 估计当前仅支持数值列；以下列不是数值类型："
            + ", ".join(non_numeric)
        )

    before = len(work)
    work = work.dropna(axis=0)
    dropped = before - len(work)
    if dropped > 0:
        warnings.append(f"listwise deletion 删除了 {dropped} 行缺失值。")
    if len(work) < 20:
        raise ValueError("可用于估计的有效样本过少（<20）。")

    if not uniq_confs:
        warnings.append(
            "未提供 confounders；估计依赖于“无混杂”假设，结论风险较高。"
        )

    try:
        _patch_networkx_for_dowhy()
        from dowhy import CausalModel
    except Exception:
        elapsed = time.perf_counter() - started
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": uniq_confs,
            "method": method,
            "n": total_rows,
            "n_rows_used": int(len(work)),
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "refute_results": [],
            "warnings": warnings,
            "error": "缺少 DoWhy 依赖，请先安装：pip install dowhy",
            "elapsed_s": round(elapsed, 4),
        }

    graph = _build_dowhy_graph(
        treatment=treatment,
        outcome=outcome,
        confounders=uniq_confs,
    )
    model = CausalModel(
        data=work,
        treatment=treatment,
        outcome=outcome,
        graph=graph,
    )
    try:
        identified = model.identify_effect(proceed_when_unidentifiable=True)
    except Exception as exc:
        raise RuntimeError(_normalize_effect_error_message(exc)) from exc

    try:
        estimate = model.estimate_effect(
            identified,
            method_name=method,
            confidence_intervals=True,
            test_significance=True,
            method_params=opts.get("method_params"),
        )
    except Exception as exc:
        raise RuntimeError(_normalize_effect_error_message(exc)) from exc
    point = _safe_float(getattr(estimate, "value", None))
    ci_low, ci_high = _extract_ci_bounds(estimate)
    p_value = _extract_p_value(estimate)

    refute_results: list[dict[str, Any]] = []
    for ref_name in ("placebo_treatment_refuter", "random_common_cause"):
        try:
            ref = model.refute_estimate(
                identified,
                estimate,
                method_name=ref_name,
            )
            refute_results.append(_refute_summary(ref_name, ref, point))
        except Exception as exc:
            refute_results.append({
                "name": ref_name,
                "passed": False,
                "summary": f"{ref_name} 执行失败：{exc}",
            })

    elapsed = time.perf_counter() - started
    return {
        "treatment": treatment,
        "outcome": outcome,
        "confounders": uniq_confs,
        "method": method,
        "n": total_rows,
        "n_rows_used": int(len(work)),
        "point_estimate": point,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_value": p_value,
        "refute_results": refute_results,
        "warnings": warnings,
        "error": None,
        "elapsed_s": round(elapsed, 4),
    }


def causal_effect_estimate(
    csv_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
    method: str = "backdoor.linear_regression",
    options: dict | None = None,
    timeout_s: int = 60,
) -> dict:
    """在本地 CSV 上执行因果效应估计（DoWhy）。"""
    if not Path(csv_path).exists():
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": list(confounders or []),
            "method": method,
            "n": 0,
            "n_rows_used": 0,
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "refute_results": [],
            "warnings": [f"CSV 文件不存在：{csv_path}"],
            "error": "csv_not_found",
            "elapsed_s": 0.0,
        }

    timeout_s = max(1, int(timeout_s))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(
            _run_effect_internal,
            csv_path,
            treatment,
            outcome,
            confounders,
            method,
            options,
        )
        try:
            return fut.result(timeout=float(timeout_s))
        except concurrent.futures.TimeoutError:
            return {
                "treatment": treatment,
                "outcome": outcome,
                "confounders": list(confounders or []),
                "method": method,
                "n": 0,
                "n_rows_used": 0,
                "point_estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
                "refute_results": [],
                "warnings": [
                    f"因果效应估计超时（>{timeout_s}s）。建议减少变量数或改用线性回归后门法。"
                ],
                "error": "timeout",
                "elapsed_s": float(timeout_s),
            }
        except Exception as exc:
            return {
                "treatment": treatment,
                "outcome": outcome,
                "confounders": list(confounders or []),
                "method": method,
                "n": 0,
                "n_rows_used": 0,
                "point_estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
                "refute_results": [],
                "warnings": [f"因果效应估计失败：{_normalize_effect_error_message(exc)}"],
                "error": _normalize_effect_error_message(exc),
                "elapsed_s": 0.0,
            }


def _format_effect_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"- 估计目标：{payload.get('treatment')} -> {payload.get('outcome')}"
    )
    lines.append(f"- 方法：{payload.get('method')}")
    lines.append(
        f"- 样本：原始 {payload.get('n', 0)} 行，估计使用 {payload.get('n_rows_used', 0)} 行"
    )
    lines.append(f"- 点估计：{payload.get('point_estimate')}")
    lines.append(
        f"- 置信区间：[{payload.get('ci_lower')}, {payload.get('ci_upper')}]"
    )
    if payload.get("p_value") is not None:
        lines.append(f"- p 值：{payload.get('p_value')}")
    if payload.get("error"):
        lines.append(f"- 执行状态：失败（{payload.get('error')}）")
    refutes = payload.get("refute_results") or []
    if refutes:
        lines.append("- 敏感性分析：")
        for r in refutes[:4]:
            status = "通过" if r.get("passed") else "未通过"
            lines.append(f"  - {r.get('name')}：{status}；{r.get('summary')}")
    warns = payload.get("warnings") or []
    if warns:
        lines.append("- 关键警告：")
        for w in warns[:6]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


@tool
def estimate_uploaded_causal_effect(
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
    method: str = "backdoor.linear_regression",
    options: dict | None = None,
    timeout_s: int = 60,
) -> str:
    """对当前会话已上传数据执行因果效应估计（DoWhy）。

    适用于用户明确提出“估计 X 对 Y 的影响”场景。工具会自动读取当前会话中的
    data.csv，执行识别 + 估计 + refutation，并把结果写入 effect.json。

    返回格式：Markdown 摘要 + 末尾 JSON 代码块。调用时不要传 thread_id。
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

    payload = causal_effect_estimate(
        csv_path=str(csv_path),
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        method=method,
        options=options,
        timeout_s=timeout_s,
    )

    effect_path = thread_dir / "effect.json"
    to_save = dict(payload)
    to_save["created_at"] = time.time()
    try:
        effect_path.write_text(
            json.dumps(to_save, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        payload.setdefault("warnings", []).append(f"effect.json 写入失败：{exc}")

    md = _format_effect_markdown(payload)
    json_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{md}\n\n```json\n{json_block}\n```"


def prepare_effect_options(
    thread_dir: Path,
    treatment: str = "",
    outcome: str = "",
) -> dict:
    """读取已学习的 dag.json，生成因果效应估计的“图 + 方法”选择项。

    因果图必须来自此前因果发现的产物（dag.json）；若不存在则返回
    blocking=True，提示应先完成因果发现。
    """
    dag_path = thread_dir / "dag.json"
    if not dag_path.exists():
        return {
            "blocking": True,
            "blocking_reason": (
                "当前会话尚未生成因果图（未找到 dag.json）。"
                "因果效应估计必须基于已学习的因果图，请先运行因果发现。"
            ),
            "dag": None,
            "treatment": treatment or None,
            "outcome": outcome or None,
            "suggested_confounders": [],
            "methods": [],
            "warnings": [],
        }

    try:
        dag = json.loads(dag_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "blocking": True,
            "blocking_reason": f"dag.json 读取失败：{exc}。请重新运行因果发现。",
            "dag": None,
            "treatment": treatment or None,
            "outcome": outcome or None,
            "suggested_confounders": [],
            "methods": [],
            "warnings": [],
        }

    edges = [e for e in (dag.get("edges") or []) if isinstance(e, dict)]
    nodes = [str(n) for n in (dag.get("nodes") or [])]
    directed = [e for e in edges if e.get("type") == "directed"]
    undirected = [e for e in edges if e.get("type") != "directed"]

    if not edges:
        return {
            "blocking": True,
            "blocking_reason": (
                "已有的因果发现结果不包含任何边，无法用于效应估计。"
                "请调整参数或换算法重新运行因果发现。"
            ),
            "dag": {
                "algorithm": dag.get("algorithm"),
                "n_nodes": len(nodes),
                "n_edges": 0,
            },
            "treatment": treatment or None,
            "outcome": outcome or None,
            "suggested_confounders": [],
            "methods": [],
            "warnings": list(dag.get("warnings") or []),
        }

    warnings: list[str] = []
    if undirected:
        warnings.append(
            f"因果图包含 {len(undirected)} 条无向边，推导混杂集合时仅使用有向边，"
            "建议结合领域知识确认边方向。"
        )

    suggested: list[str] = []
    t = str(treatment or "").strip()
    o = str(outcome or "").strip()
    if t and t not in nodes:
        warnings.append(f"treatment「{t}」不在因果图节点中。")
    if o and o not in nodes:
        warnings.append(f"outcome「{o}」不在因果图节点中。")
    if t in nodes and o in nodes:
        parents_t = {str(e.get("from")) for e in directed if str(e.get("to")) == t}
        parents_o = {str(e.get("from")) for e in directed if str(e.get("to")) == o}
        suggested = sorted((parents_t & parents_o) - {t, o})
        if not suggested:
            warnings.append(
                "根据因果图未找到同时指向 treatment 与 outcome 的共同父节点，"
                "混杂集合建议为空；如有领域知识可手动补充。"
            )

    return {
        "blocking": False,
        "blocking_reason": None,
        "dag": {
            "algorithm": dag.get("algorithm"),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "edges": edges,
            "mermaid": dag.get("mermaid"),
        },
        "treatment": t or None,
        "outcome": o or None,
        "suggested_confounders": suggested,
        "methods": [dict(m) for m in _METHOD_CATALOG],
        "warnings": warnings,
    }


@tool
def prepare_uploaded_effect_options(
    treatment: str = "",
    outcome: str = "",
) -> str:
    """在执行因果效应估计前，读取当前会话已学习的因果图并生成可选方案。

    当用户表达“估计 X 对 Y 的因果效应”意图时，应先调用本工具（而不是直接估计）：
    - 若返回 blocking=True：说明当前会话没有因果发现结果（dag.json），
      必须先引导用户完成因果发现，再回到效应估计；
    - 若返回 blocking=False：返回已学习的因果图摘要、基于图推导的混杂集合建议、
      以及支持的估计方法清单，用于生成 effect-choice 卡片供用户点选。

    返回 Markdown 摘要 + 末尾 JSON 代码块。调用时不要传 thread_id。
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
    if not (thread_dir / "data.csv").exists():
        return "当前会话未找到 data.csv，请先上传 CSV。"

    payload = prepare_effect_options(thread_dir, treatment=treatment, outcome=outcome)

    lines: list[str] = []
    if payload["blocking"]:
        lines.append(f"- 状态：无法进入效应估计（{payload['blocking_reason']}）")
    else:
        dag = payload["dag"] or {}
        lines.append(
            f"- 因果图：来自 {dag.get('algorithm')} 算法，"
            f"{dag.get('n_nodes')} 个节点 / {dag.get('n_edges')} 条边"
        )
        if payload.get("treatment") and payload.get("outcome"):
            lines.append(
                f"- 估计目标：{payload['treatment']} -> {payload['outcome']}"
            )
        lines.append(
            "- 建议混杂集合："
            + (", ".join(payload["suggested_confounders"]) or "（空）")
        )
        lines.append(
            "- 可选方法：" + ", ".join(m["method"] for m in payload["methods"])
        )
    for w in payload.get("warnings") or []:
        lines.append(f"- 警告：{w}")

    json_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return "\n".join(lines) + f"\n\n```json\n{json_block}\n```"
