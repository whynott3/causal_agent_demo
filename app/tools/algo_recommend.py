"""因果发现算法推荐工具（Sprint 3 / P2.T3）。

两层 API：

1. 底层纯函数：
   - recommend_discovery_algorithm(profile, *, user_hints="", csv_path=None) -> dict
     纯启发式规则；不调 LLM、不依赖 causal-learn / dowhy 等重型库。

2. LangChain @tool 包装：
   - recommend_causal_discovery_algorithms(thread_id, user_hints="") -> str
     读 ``app/uploads/<thread_id>/profile.json``，可选读 data.csv 做轻量峰度检测，
     输出 Markdown 摘要 + 末尾附结构化 ```json``` 块，供 Agent 填入 algo-choice
     causal-card。

规则参照 ``因果助手改造.md`` Sprint 3 / P2.T3 表格。本工具只负责推荐；
算法真实执行由 ``run_causal_discovery`` / ``run_uploaded_causal_discovery`` 完成。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from common.runtime_context import get_current_thread_id
from tools.data_profile import _uploads_root  # 复用上传根目录与 thread 白名单语义


_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# 触发"用户怀疑隐藏混杂"的关键词（中英文）
_LATENT_KEYWORDS = (
    "隐藏混杂", "未观测", "未观察", "潜在混杂", "潜在变量",
    "latent", "unobserved", "hidden confound",
)
# 触发"时序数据"的列名 / 关键词
_TIME_NAME_RE = re.compile(r"(date|time|created|updated|timestamp|datetime)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _safe_thread_dir(thread_id: str) -> Optional[Path]:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        return None
    return _uploads_root() / thread_id


def _round(x: Any, n: int = 4) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except (TypeError, ValueError):
        return None


def _detect_non_gaussian(csv_path: Optional[str], numeric_names: list[str]) -> int:
    """统计 |超额峰度| > 1 的数值列数。csv_path 不可用时返回 0。"""
    if not csv_path or not numeric_names:
        return 0
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0
    hits = 0
    for name in numeric_names:
        if name not in df.columns:
            continue
        col = pd.to_numeric(df[name], errors="coerce").dropna()
        if len(col) < 30:
            continue
        try:
            k = float(col.kurtosis())  # pandas 默认是超额峰度（Fisher）
        except Exception:
            continue
        if abs(k) > 1.0:
            hits += 1
    return hits


def _build_data_signals(profile: dict, user_hints: str) -> dict:
    n_rows = int(profile.get("n_rows") or 0)
    n_cols = int(profile.get("n_cols") or 0)
    n_num = int(profile.get("n_numeric") or 0)
    n_cat = int(profile.get("n_categorical") or 0)
    missing = float(profile.get("missing_overall") or 0.0)
    cols = profile.get("columns") or []

    has_datetime = any(
        (c.get("dtype") == "datetime") or _TIME_NAME_RE.search(str(c.get("name") or ""))
        for c in cols
    )
    suspected_ts = has_datetime and n_rows >= 50
    mostly_numeric = n_num > 0 and n_num >= max(2, int(0.8 * n_cols))

    text = (user_hints or "").lower()
    latent = any(kw.lower() in text for kw in _LATENT_KEYWORDS)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": n_num,
        "n_categorical": n_cat,
        "missing_overall": _round(missing),
        "has_datetime": bool(has_datetime),
        "suspected_timeseries": bool(suspected_ts),
        "mostly_numeric": bool(mostly_numeric),
        "user_mentions_latent_confounding": bool(latent),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def recommend_discovery_algorithm(
    profile: dict,
    *,
    user_hints: str = "",
    csv_path: Optional[str] = None,
) -> dict:
    """按启发式规则推荐因果发现算法。

    Args:
        profile: ``profile_csv`` / ``profile.json`` 的输出 dict。
        user_hints: 用户本轮自然语言提示（用于检测「隐藏混杂」等关键词）。
        csv_path: 可选 CSV 路径；仅用于补充非高斯性（峰度）检测；
            不会把原始行返回给调用方。

    Returns:
        严格 JSON 兼容 dict，键见模块顶部 docstring。
    """
    signals = _build_data_signals(profile, user_hints)
    n_rows = signals["n_rows"]
    n_cols = signals["n_cols"]
    n_num = signals["n_numeric"]
    n_cat = signals["n_categorical"]
    missing = signals["missing_overall"] or 0.0

    cols = profile.get("columns") or []
    numeric_names = [c.get("name") for c in cols if c.get("dtype") == "numeric"]

    # ----- 阻断规则 -----
    if n_rows > 0 and n_rows < 100:
        return {
            "blocking": True,
            "blocking_reason": (
                f"样本量过小（n={n_rows}<100），因果发现结果极不稳定。"
                "建议先收集更多数据，或先用 DAG 工具基于领域知识手工建模。"
            ),
            "data_signals": signals,
            "recommendations": [],
            "not_recommended": [],
            "global_warnings": [],
        }
    if n_cols > 0 and n_rows > 0 and n_cols > n_rows:
        return {
            "blocking": True,
            "blocking_reason": (
                f"变量数大于样本量（cols={n_cols}>rows={n_rows}），"
                "不建议直接做因果发现；建议先做特征筛选或主成分分析降维。"
            ),
            "data_signals": signals,
            "recommendations": [],
            "not_recommended": [],
            "global_warnings": [],
        }
    if n_num < 2 and n_cat == 0:
        return {
            "blocking": True,
            "blocking_reason": (
                "数据集中有效变量过少（数值列<2 且无分类列），无法进行因果发现。"
            ),
            "data_signals": signals,
            "recommendations": [],
            "not_recommended": [],
            "global_warnings": [],
        }
    if n_num == 0 and n_cat > 0:
        return {
            "blocking": True,
            "blocking_reason": (
                "数据集中没有数值列，全部为分类/文本类型；当前推荐的因果发现算法"
                "需要至少 2 个数值变量。建议先对分类变量做合理的数值编码，"
                "或先基于领域知识用 DAG 工具手工建模。"
            ),
            "data_signals": signals,
            "recommendations": [],
            "not_recommended": [],
            "global_warnings": [],
        }

    # ----- 非阻断 global_warnings -----
    global_warnings: list[str] = []
    if missing > 0.2:
        global_warnings.append(
            f"整体缺失率较高（{missing:.1%}），建议先处理缺失再跑发现算法，"
            "否则边的方向和置信度都会受影响。"
        )
    if n_rows < 200:
        global_warnings.append(
            f"样本量偏少（n={n_rows}），所有算法的结果都需要保守解读。"
        )

    # ----- 推荐规则（条件可同时命中，按 priority 排序） -----
    recommendations: list[dict] = []
    not_recommended: list[dict] = []

    latent = signals["user_mentions_latent_confounding"]
    full_numeric = (n_cat == 0 and n_num >= 2)
    mid_scale_ok = (n_rows >= 200 and n_cols <= 30)

    # PC / GES：经典约束 / 得分 base
    if full_numeric and mid_scale_ok and not latent:
        recommendations.append({
            "algorithm": "PC",
            "priority": 1,
            "reason": (
                "数据规模适中、变量全为数值、未提示存在未观测混杂，"
                "PC 算法（基于条件独立性约束）是稳定的首选；"
                "结果会以 CPDAG 形式给出可识别的因果方向，无法定向的边以无向形式保留。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "PC 假设无未观测混杂；若该假设不成立，结果可能存在系统性偏差。",
            ],
        })
        recommendations.append({
            "algorithm": "GES",
            "priority": 2,
            "reason": (
                "得分函数（BIC）驱动的搜索，可与 PC 的结果交叉对比；"
                "在中小规模、全数值场景下与 PC 配对使用，能提高结论的稳健性。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "GES 同样假设线性 / 高斯关系与无隐藏混杂；与 PC 一同使用以互相印证。",
            ],
        })
    elif full_numeric and not latent:
        # 中等规模但变量数偏大或样本偏少，仍给 PC，但置 warnings
        recommendations.append({
            "algorithm": "PC",
            "priority": 1,
            "reason": (
                "数据全为数值变量，PC 仍是首选；但当前样本量 / 变量数边界条件不太理想，"
                "请结合结果谨慎解读。"
            ),
            "preconditions_ok": n_rows >= 200,
            "warnings": [
                "样本量或变量规模处于经验边界，建议同时跑 GES 互相印证。",
            ],
        })

    # FCI：可能有隐藏混杂
    if latent and full_numeric:
        recommendations.append({
            "algorithm": "FCI",
            "priority": 1,
            "reason": (
                "用户明确提示数据中可能存在未观测的混杂变量；"
                "FCI 算法能在存在隐藏变量的前提下输出 PAG（部分有向无环图），"
                "用双向边/圆头标记不可识别方向，比 PC/GES 更稳健。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "FCI 的输出可读性低于 PC：部分边方向不可识别，需要结合领域知识解读。",
            ],
        })

    # NOTEARS：高维或大样本连续优化
    if full_numeric and (n_cols > 30 or n_rows >= 1000):
        recommendations.append({
            "algorithm": "NOTEARS",
            "priority": 2 if not latent else 3,
            "reason": (
                "数据规模较大（高维或大样本），NOTEARS 把 DAG 学习转化为"
                "连续优化问题，可扩展到上百个变量；适合作为 PC/GES 的补充。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "NOTEARS 默认假设线性关系；非线性场景需要换成 NOTEARS-MLP 等扩展。",
                "NOTEARS 为连续优化结构学习，结果对阈值和正则系数较敏感，建议与 PC/FCI 交叉验证。",
            ],
        })

    # LiNGAM：非高斯性显著
    non_gauss_hits = _detect_non_gaussian(csv_path, numeric_names)
    if full_numeric and non_gauss_hits >= 2:
        recommendations.append({
            "algorithm": "LiNGAM",
            "priority": 2,
            "reason": (
                f"检测到 {non_gauss_hits} 个数值列具有明显非高斯分布（|超额峰度|>1），"
                "LiNGAM 可利用非高斯性识别完全有向的因果方向，是 PC/GES 之外的有力补充。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "LiNGAM 假设线性 + 无隐藏混杂 + 至少一个非高斯噪声；前提偏强。",
            ],
        })

    # PCMCI：疑似时序
    if signals["suspected_timeseries"]:
        recommendations.append({
            "algorithm": "PCMCI",
            "priority": 3,
            "reason": (
                "数据包含时间戳列且样本量足够，疑似时间序列数据；"
                "PCMCI 是时序因果发现的常用方法，能识别滞后期因果关系。"
            ),
            "preconditions_ok": True,
            "warnings": [
                "需要确认数据确实是按时间排序的面板/时序数据，而不是含有日期字段的横截面数据。",
                "PCMCI 当前仍未接入真实执行，建议先用领域知识校验后再选替代算法。",
            ],
        })

    # 排序：priority 升序，同 priority 保持插入顺序
    recommendations.sort(key=lambda r: (r["priority"], r["algorithm"]))

    # ----- not_recommended -----
    if latent:
        not_recommended.append({
            "algorithm": "PC",
            "reason": "用户提示存在未观测混杂；PC 在该前提下结果会有系统性偏差，已用 FCI 替代。",
        })
        not_recommended.append({
            "algorithm": "GES",
            "reason": "GES 同样假设无隐藏混杂，存在未观测变量时不建议使用。",
        })
    if n_cols > 30 and not any(r["algorithm"] == "PC" for r in recommendations):
        not_recommended.append({
            "algorithm": "PC",
            "reason": f"变量数偏大（cols={n_cols}>30），PC 的条件独立性检验数量会爆炸，建议优先试 NOTEARS。",
        })
    if not signals["suspected_timeseries"] and signals["has_datetime"]:
        not_recommended.append({
            "algorithm": "PCMCI",
            "reason": "虽含时间戳列，但样本量不足以构成可靠的时序分析，先按横截面数据处理。",
        })

    return {
        "blocking": False,
        "blocking_reason": None,
        "data_signals": signals,
        "recommendations": recommendations,
        "not_recommended": not_recommended,
        "global_warnings": global_warnings,
    }


# ---------------------------------------------------------------------------
# @tool 包装
# ---------------------------------------------------------------------------

def _format_recommendation_markdown(payload: dict) -> str:
    sig = payload.get("data_signals") or {}
    n_rows = sig.get("n_rows", 0)
    n_cols = sig.get("n_cols", 0)
    n_num = sig.get("n_numeric", 0)
    n_cat = sig.get("n_categorical", 0)
    missing = sig.get("missing_overall") or 0.0

    lines: list[str] = []
    lines.append(
        f"- 数据信号：{n_rows} 行 × {n_cols} 列，数值 {n_num} / 分类 {n_cat}，"
        f"整体缺失率 {missing:.1%}"
        + ("，疑似时序数据" if sig.get("suspected_timeseries") else "")
        + ("，用户提示存在未观测混杂" if sig.get("user_mentions_latent_confounding") else "")
    )

    if payload.get("blocking"):
        lines.append(f"- 阻断建议：{payload.get('blocking_reason')}")
    else:
        recs = payload.get("recommendations") or []
        if recs:
            lines.append("- 候选算法（按优先级）：")
            for r in recs[:5]:
                lines.append(
                    f"  - **{r.get('algorithm')}**（优先级 {r.get('priority')}）："
                    f"{r.get('reason')}"
                )
        nr = payload.get("not_recommended") or []
        if nr:
            lines.append("- 不推荐：")
            for r in nr[:3]:
                lines.append(f"  - {r.get('algorithm')}：{r.get('reason')}")
        gw = payload.get("global_warnings") or []
        if gw:
            lines.append("- 全局警告：")
            for w in gw:
                lines.append(f"  - {w}")
    return "\n".join(lines)


@tool
def recommend_causal_discovery_algorithms(user_hints: str = "") -> str:
    """根据当前会话已上传数据的画像（profile），结合用户补充说明（是否怀疑隐藏混杂、
    是否时序数据等），用启发式规则推荐因果发现算法
    （PC / GES / FCI / NOTEARS / LiNGAM / PCMCI）。

    返回 Markdown 摘要 + 末尾附一段 ```json``` 代码块，块内是结构化推荐结果
    （供 Agent 复制到 causal-card type=algo-choice 卡片）。
    若样本过小或变量过多，会返回阻断建议而非强行推荐；本版本仅推荐不执行算法。

    工具内部会自动读取当前会话 ID（由后端运行时上下文注入），**LLM 调用本工具
    时只需要传 ``user_hints``，不要也不应传 thread_id**。

    Args:
        user_hints: 用户自然语言提示，例如「我担心可能有未观测的混杂因素」。
    """
    thread_id = get_current_thread_id()
    if thread_id is None:
        return (
            "未能读取当前会话上下文。该工具应由聊天接口在会话中调用；"
            "若是直接测试，请在测试中通过 bind_thread_id 绑定会话 ID。"
        )

    thread_dir = _safe_thread_dir(thread_id)
    if thread_dir is None:
        return "thread_id 不合法，无法读取数据集画像。"

    profile_path = thread_dir / "profile.json"
    if not profile_path.exists():
        return (
            "未能从当前会话目录中找到数据画像文件；可能是用户尚未上传 CSV，"
            "或会话 ID 与上传时不一致。请提示用户先通过附件按钮上传 .csv，"
            "**禁止**在没有真实推荐结果的情况下自行编造算法对比或文字版推荐。"
        )

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"读取数据画像失败：{exc}"

    csv_path = thread_dir / "data.csv"
    payload = recommend_discovery_algorithm(
        profile,
        user_hints=user_hints,
        csv_path=str(csv_path) if csv_path.exists() else None,
    )

    md = _format_recommendation_markdown(payload)
    json_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{md}\n\n```json\n{json_block}\n```"
