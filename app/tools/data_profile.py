"""CSV 数据画像工具（Sprint 2 / P2.T2）。

两层 API：

1. 底层纯函数（便于单元测试）：
   - profile_csv(csv_path, encoding='utf-8') -> dict

2. LangChain @tool 包装（直接挂到 Agent）：
   - summarize_uploaded_dataset(thread_id) -> str
     读取当前会话已上传 CSV 的 profile.json，输出一段紧凑 Markdown 摘要；
     详细 profile 由前端 causal-card 卡片渲染，正文不要重复堆数值表。

实现原则：
- profile_csv 同步、纯函数，无副作用（不写文件）；
- 上传 endpoint 负责把 profile_csv 的返回结果 json.dump 到
  ``app/uploads/<thread_id>/profile.json``。
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


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_TOP_CORR_LIMIT = 15
_DATETIME_NAME_RE = re.compile(r"(date|time|created|updated|timestamp|datetime)", re.IGNORECASE)
_BOOL_VALUE_SET: set[Any] = {0, 1, 0.0, 1.0, True, False}
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _round_or_none(x: Any, n: int = 4) -> Optional[float]:
    """把 numpy / pandas 标量安全转换为 float 并保留 n 位小数；NaN/Inf 返回 None。"""
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, n)
    except (TypeError, ValueError):
        return None


def _is_categorical_dtype(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _classify_dtype(series: pd.Series, name: str) -> str:
    """把单列归类为 numeric / boolean / datetime / categorical / text。"""
    n = len(series)
    if n == 0:
        return "text"

    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        # 二值数值列：恰好 2 个唯一值且值域是 {0,1,True,False} 才算 boolean，
        # 仅 1 个唯一值的常量列保持为 numeric（由 warnings 标注常量）。
        if n_unique == 2:
            unique_vals = set(non_null.unique().tolist())
            if unique_vals.issubset(_BOOL_VALUE_SET):
                return "boolean"
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if _DATETIME_NAME_RE.search(str(name)) and not non_null.empty:
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", utc=False)
            fail_rate = float(parsed.isna().mean())
            if fail_rate < 0.05:
                return "datetime"
        except Exception:
            pass

    # 字符串 / 类别列：兼容 pandas 2.x 的 object 与 pandas 3.x 默认的 StringDtype。
    is_string_like = (
        _is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or series.dtype == object
    )
    if is_string_like:
        if n_unique > 0 and n_unique / n < 0.5 and n_unique <= 50:
            return "categorical"
        return "text"

    return "text"


def _column_stats(series: pd.Series, dtype_label: str) -> dict:
    non_null = series.dropna()
    if dtype_label == "numeric" and len(non_null) > 0:
        desc = pd.to_numeric(non_null, errors="coerce").describe()
        return {
            "mean": _round_or_none(desc.get("mean")),
            "std": _round_or_none(desc.get("std")),
            "p25": _round_or_none(desc.get("25%")),
            "p50": _round_or_none(desc.get("50%")),
            "p75": _round_or_none(desc.get("75%")),
            "min": _round_or_none(desc.get("min")),
            "max": _round_or_none(desc.get("max")),
        }
    if dtype_label in ("categorical", "boolean") and len(non_null) > 0:
        counts = non_null.value_counts(normalize=True).head(5)
        return {
            "top_values": [
                {"value": str(idx), "ratio": _round_or_none(ratio)}
                for idx, ratio in counts.items()
            ]
        }
    if dtype_label == "datetime":
        try:
            parsed = pd.to_datetime(series, errors="coerce", utc=False).dropna()
            if len(parsed) > 0:
                return {
                    "min_iso": parsed.min().isoformat(),
                    "max_iso": parsed.max().isoformat(),
                }
        except Exception:
            pass
        return {"min_iso": None, "max_iso": None}
    if dtype_label == "text" and len(non_null) > 0:
        lens = non_null.astype(str).str.len()
        return {
            "avg_len": _round_or_none(float(lens.mean()), 2),
            "max_len": int(lens.max()),
        }
    # 兜底
    if dtype_label == "numeric":
        return {"mean": None, "std": None, "p25": None, "p50": None, "p75": None, "min": None, "max": None}
    if dtype_label in ("categorical", "boolean"):
        return {"top_values": []}
    if dtype_label == "text":
        return {"avg_len": None, "max_len": None}
    return {}


def _top_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    if len(numeric_cols) < 2:
        return []
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    pearson = num_df.corr(method="pearson")
    spearman = num_df.corr(method="spearman")

    candidates: list[dict] = []
    for i, a in enumerate(numeric_cols):
        for j in range(i + 1, len(numeric_cols)):
            b = numeric_cols[j]
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if (p is None or pd.isna(p)) and (s is None or pd.isna(s)):
                continue
            candidates.append({
                "a": a,
                "b": b,
                "pearson": _round_or_none(p),
                "spearman": _round_or_none(s),
            })

    def _abs_pearson(item: dict) -> float:
        v = item.get("pearson")
        return abs(v) if isinstance(v, (int, float)) else 0.0

    candidates.sort(key=_abs_pearson, reverse=True)
    return candidates[:_TOP_CORR_LIMIT]


def _build_warnings(
    n_rows: int,
    n_cols: int,
    columns_info: list[dict],
) -> list[str]:
    warnings: list[str] = []
    for ci in columns_info:
        col = ci["name"]
        nu = ci["n_unique"]
        mr = ci["missing_rate"] or 0.0
        dtype_label = ci["dtype"]
        # 「疑似 ID 列」只在 text / categorical 列上触发，
        # 数值列与 datetime 列天然存在唯一值，不应误判。
        if (
            n_rows > 0
            and nu / n_rows > 0.95
            and dtype_label in ("text", "categorical")
        ):
            warnings.append(f"列 {col} 唯一值过多（{nu}/{n_rows}），疑似 ID 列")
        if mr > 0.4:
            warnings.append(f"列 {col} 缺失率过高（{mr:.1%}）")
        if nu <= 1:
            warnings.append(f"列 {col} 为常量列")
        if dtype_label == "datetime":
            warnings.append(f"列 {col} 疑似时间戳")
    if 0 < n_rows < 200:
        warnings.append(f"样本量较小（n={n_rows}），后续因果发现结果不稳定")
    if n_cols > n_rows > 0:
        warnings.append(f"变量数大于样本量（cols={n_cols}>rows={n_rows}），不建议直接做因果发现")
    return warnings


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def profile_csv(csv_path: str, encoding: str = "utf-8") -> dict:
    """对 CSV 做一份聚合画像，输出严格 JSON 兼容的 dict。

    Args:
        csv_path: 本地 CSV 文件路径。
        encoding: 文件编码（``utf-8`` / ``utf-8-sig`` / ``gbk`` 等）。

    Returns:
        见模块 docstring 顶部 JSON 契约。**所有 numpy 标量都已转换为原生
        Python 类型，可直接 ``json.dumps``**。
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    columns_info: list[dict] = []
    n_numeric = 0
    n_categorical = 0

    for col in df.columns:
        series = df[col]
        dtype_label = _classify_dtype(series, str(col))
        if dtype_label == "numeric":
            n_numeric += 1
        elif dtype_label == "categorical":
            n_categorical += 1

        n_unique = int(series.dropna().nunique())
        missing_rate = float(series.isna().mean()) if n_rows > 0 else 0.0
        stats = _column_stats(series, dtype_label)

        columns_info.append({
            "name": str(col),
            "dtype": dtype_label,
            "missing_rate": _round_or_none(missing_rate),
            "n_unique": n_unique,
            "stats": stats,
        })

    if n_rows > 0 and n_cols > 0:
        missing_overall = float(df.isna().to_numpy().mean())
    else:
        missing_overall = 0.0

    duplicates = int(df.duplicated().sum())

    numeric_cols = [ci["name"] for ci in columns_info if ci["dtype"] == "numeric"]
    top_corrs = _top_correlations(df, numeric_cols)

    warnings_list = _build_warnings(n_rows, n_cols, columns_info)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns_info,
        "missing_overall": _round_or_none(missing_overall),
        "duplicates": duplicates,
        "top_correlations": top_corrs,
        "warnings": warnings_list,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
    }


# ---------------------------------------------------------------------------
# @tool 包装
# ---------------------------------------------------------------------------

def _uploads_root() -> Path:
    """app/uploads 目录路径。

    优先 ``APP_UPLOADS_DIR`` 环境变量（便于测试）；
    否则回到工作目录下的 ``uploads``（项目运行入口是 ``cd app && python -m main``，
    cwd 就是 app/）。
    """
    import os
    override = os.getenv("APP_UPLOADS_DIR")
    if override:
        return Path(override)
    return Path("uploads")


def _safe_thread_dir(thread_id: str) -> Optional[Path]:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        return None
    return _uploads_root() / thread_id


@tool
def summarize_uploaded_dataset(thread_id: str) -> str:
    """读取当前会话已上传 CSV 的数据画像，输出一段紧凑 Markdown 摘要
    （行 / 列 / 数值列占比 / 缺失率 / 主要警告）。

    用途：用于「数据驱动模式 · 概览阶段」确认 profile 已就绪并给出自然
    语言概览；详细 profile 由前端通过卡片渲染，因此正文**不要**重复
    堆数值表。

    Args:
        thread_id: 当前会话 ID（与 /chat/stream 使用的 thread_id 一致）。

    Returns:
        Markdown 文本：5-8 行关键信号；若尚未上传或读取失败，返回提示。
    """
    thread_dir = _safe_thread_dir(thread_id)
    if thread_dir is None:
        return "thread_id 不合法，无法读取数据集画像。"

    profile_path = thread_dir / "profile.json"
    if not profile_path.exists():
        return "当前会话尚未上传 CSV 数据集；请提示用户先通过附件按钮上传 .csv。"

    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"读取数据画像失败：{exc}"

    n_rows = data.get("n_rows", 0)
    n_cols = data.get("n_cols", 0)
    n_num = data.get("n_numeric", 0)
    n_cat = data.get("n_categorical", 0)
    n_other = max(0, n_cols - n_num - n_cat)
    missing = data.get("missing_overall") or 0.0
    dups = data.get("duplicates", 0)

    lines: list[str] = []
    lines.append(f"- 规模：{n_rows} 行 × {n_cols} 列")
    lines.append(
        f"- 列类型分布：数值 {n_num} 列、分类 {n_cat} 列、其它 {n_other} 列"
    )
    lines.append(f"- 整体缺失率：{missing:.1%}")
    lines.append(f"- 完全重复行：{dups} 条")

    corrs = data.get("top_correlations") or []
    if corrs:
        top = corrs[0]
        lines.append(
            f"- 数值列相关性最强对：{top.get('a')} ↔ {top.get('b')}"
            f"（Pearson={top.get('pearson')}, Spearman={top.get('spearman')}）"
        )

    warnings_list = data.get("warnings") or []
    if warnings_list:
        lines.append(f"- 数据警告：共 {len(warnings_list)} 条，重点关注：")
        for w in warnings_list[:5]:
            lines.append(f"  - {w}")

    columns = data.get("columns") or []
    numeric_cols = [c for c in columns if c.get("dtype") == "numeric"]
    if numeric_cols:
        lines.append("- 数值列分布摘要（前 6 列）：")
        for c in numeric_cols[:6]:
            st = c.get("stats") or {}
            mean = st.get("mean")
            std = st.get("std")
            p50 = st.get("p50")
            lo = st.get("min")
            hi = st.get("max")
            parts = []
            if mean is not None and std is not None:
                parts.append(f"μ={mean}, σ={std}")
            if p50 is not None:
                parts.append(f"中位={p50}")
            if lo is not None and hi is not None:
                parts.append(f"范围=[{lo}, {hi}]")
            lines.append(f"  - {c.get('name')}: " + ("；".join(parts) if parts else "（无统计）"))

    cat_cols = [c for c in columns if c.get("dtype") in ("categorical", "boolean")]
    if cat_cols:
        lines.append("- 分类 / 二值列主要取值：")
        for c in cat_cols[:4]:
            tops = (c.get("stats") or {}).get("top_values") or []
            if not tops:
                continue
            top_str = ", ".join(
                f"{t.get('value')}({(t.get('ratio') or 0) * 100:.0f}%)"
                for t in tops[:3]
            )
            lines.append(f"  - {c.get('name')}: {top_str}")

    return "\n".join(lines)
