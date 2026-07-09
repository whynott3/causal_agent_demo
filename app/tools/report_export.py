"""因果分析报告导出工具（Sprint 6 / P5）。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

from common.runtime_context import get_current_thread_id
from tools.data_profile import _uploads_root

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _safe_thread_dir(thread_id: str) -> Optional[Path]:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        return None
    return _uploads_root() / thread_id


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"读取 {path.name} 失败：{exc}"


def _fmt_num(x: Any) -> str:
    if x is None:
        return "未提供"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def _extract_top_corr(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "未提供"
    corrs = profile.get("top_correlations") or []
    if not corrs:
        return "未提供"
    top = corrs[0]
    return (
        f"{top.get('a')} 与 {top.get('b')}（Pearson={_fmt_num(top.get('pearson'))}，"
        f"Spearman={_fmt_num(top.get('spearman'))}）"
    )


def _build_markdown(
    thread_id: str,
    profile: dict[str, Any] | None,
    dag: dict[str, Any] | None,
    effect: dict[str, Any] | None,
    recent_question: str,
    include_refs: bool,
    warnings: list[str],
) -> str:
    n_rows = profile.get("n_rows") if profile else None
    n_cols = profile.get("n_cols") if profile else None
    n_num = profile.get("n_numeric") if profile else None
    n_cat = profile.get("n_categorical") if profile else None

    dag_algo = dag.get("algorithm") if dag else None
    dag_edges = len(dag.get("edges") or []) if dag else None
    mermaid = dag.get("mermaid") if dag else None

    eff_t = effect.get("treatment") if effect else None
    eff_o = effect.get("outcome") if effect else None
    eff_method = effect.get("method") if effect else None
    eff_point = effect.get("point_estimate") if effect else None
    eff_ci_low = effect.get("ci_lower") if effect else None
    eff_ci_high = effect.get("ci_upper") if effect else None
    eff_refute = effect.get("refute_results") if effect else []

    section_problem = recent_question.strip() if recent_question else "未提供（建议在导出时传入最近一轮研究问题）。"
    section_data = (
        f"- 数据规模：{n_rows if n_rows is not None else '未提供'} 行 × {n_cols if n_cols is not None else '未提供'} 列\n"
        f"- 列类型：数值 {n_num if n_num is not None else '未提供'}，分类 {n_cat if n_cat is not None else '未提供'}\n"
        f"- 代表性相关信号：{_extract_top_corr(profile)}"
        if profile
        else "- 未运行该步骤（未找到 profile.json）。"
    )
    section_roles = (
        f"- Treatment：{eff_t or '未提供'}\n"
        f"- Outcome：{eff_o or '未提供'}\n"
        f"- Confounders：{', '.join(effect.get('confounders') or []) if effect else '未提供'}"
    )
    section_dag = (
        f"- 结构学习算法：{dag_algo or '未提供'}\n"
        f"- 发现边数：{dag_edges if dag_edges is not None else '未提供'}\n"
        f"- 说明：该图为观测数据下的候选结构，不等同于已证明的真实因果图。"
        if dag
        else "- 未运行该步骤（未找到 dag.json）。"
    )
    section_ident = (
        "- 建议基于候选 DAG 与领域知识识别后门路径，并验证调整集合是否满足可识别性。\n"
        "- 若 DAG 未确认，当前识别策略仅供探索，不应直接用于政策结论。"
    )
    section_effect = (
        f"- 方法：{eff_method or '未提供'}\n"
        f"- 点估计：{_fmt_num(eff_point)}\n"
        f"- 置信区间：[{_fmt_num(eff_ci_low)}, {_fmt_num(eff_ci_high)}]\n"
        + (
            "- 敏感性分析：\n"
            + "\n".join(
                f"  - {r.get('name')}: {'通过' if r.get('passed') else '未通过'}；{r.get('summary')}"
                for r in eff_refute[:6]
            )
            if eff_refute
            else "- 敏感性分析：未提供"
        )
        if effect
        else "- 未运行该步骤（未找到 effect.json）。"
    )
    section_risk = (
        "- 结论依赖于因果图假设、可识别性假设与数据质量假设。\n"
        "- 观测数据不能自动证明因果方向；需结合实验、先验知识或稳健性检验。\n"
        "- 若存在隐藏混杂、样本偏差或测量误差，估计可能产生偏差。"
    )
    if effect and eff_point is not None:
        direction = "正向" if float(eff_point) >= 0 else "负向"
        section_conclusion = (
            f"- 当前估计显示 {eff_t or '处理变量'} 对 {eff_o or '结果变量'} 为{direction}影响（估计值 {_fmt_num(eff_point)}）。\n"
            "- 建议下一步：补充稳健性检验、对比替代模型、复核变量定义与时间顺序。"
        )
    else:
        section_conclusion = "- 当前无法给出定量因果结论（缺少效应估计结果）。建议先完成 effect 估计步骤。"

    refs = "- 未检索到可归档的检索来源。" if include_refs else "- 已省略。"

    lines = [
        f"# 因果分析报告（{thread_id}）",
        "",
        "## 1. 问题定义",
        section_problem,
        "",
        "## 2. 数据概览",
        section_data,
        "",
        "## 3. 变量角色",
        section_roles,
        "",
        "## 4. 候选因果图与结构发现结果",
        section_dag,
        "",
    ]
    if mermaid:
        lines.extend(
            [
                "```mermaid",
                mermaid,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. 识别策略与调整集合",
            section_ident,
            "",
            "## 6. 因果效应估计结果",
            section_effect,
            "",
            "## 7. 关键假设与风险",
            section_risk,
            "",
            "## 8. 结论与下一步",
            section_conclusion,
            "",
            "## 9. 参考资料",
            refs,
        ]
    )
    if warnings:
        lines.extend(
            [
                "",
                "## 附录：构建警告",
                *[f"- {w}" for w in warnings],
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_causal_report(
    thread_id: str,
    *,
    include_refs: bool = True,
    recent_question: str = "",
) -> dict[str, Any]:
    """基于会话目录工件构建完整 Markdown 报告。"""
    warnings: list[str] = []
    thread_dir = _safe_thread_dir(thread_id)
    if thread_dir is None:
        return {
            "title": "因果分析报告",
            "thread_id": thread_id,
            "generated_at": time.time(),
            "markdown": "",
            "sections": [],
            "warnings": ["thread_id 不合法。"],
            "error": "invalid_thread_id",
        }

    profile, err = _read_json(thread_dir / "profile.json")
    if err:
        warnings.append(err)
    if profile is None:
        warnings.append("未找到 profile.json，数据概览将以“未运行”呈现。")

    dag, err = _read_json(thread_dir / "dag.json")
    if err:
        warnings.append(err)
    if dag is None:
        warnings.append("未找到 dag.json，结构发现章节将以“未运行”呈现。")

    effect, err = _read_json(thread_dir / "effect.json")
    if err:
        warnings.append(err)
    if effect is None:
        warnings.append("未找到 effect.json，效应估计章节将以“未运行”呈现。")

    markdown = _build_markdown(
        thread_id=thread_id,
        profile=profile,
        dag=dag,
        effect=effect,
        recent_question=recent_question,
        include_refs=include_refs,
        warnings=warnings,
    )
    sections = [
        "问题定义",
        "数据概览",
        "变量角色",
        "候选因果图与结构发现结果",
        "识别策略与调整集合",
        "因果效应估计结果",
        "关键假设与风险",
        "结论与下一步",
        "参考资料",
    ]
    return {
        "title": "因果分析报告",
        "thread_id": thread_id,
        "generated_at": time.time(),
        "markdown": markdown,
        "sections": sections,
        "warnings": warnings,
        "error": None,
    }


@tool
def build_uploaded_causal_report(
    include_refs: bool = True,
    recent_question: str = "",
) -> str:
    """为当前会话生成可复制的完整因果分析报告。

    使用当前会话目录下已存在的 profile/dag/effect 工件进行汇总。若某步骤未运行，
    报告会保留对应章节并标注“未运行该步骤”，不会编造数字结论。
    """
    thread_id = get_current_thread_id()
    if thread_id is None:
        return (
            "未能读取当前会话上下文。该工具应由聊天接口在会话中调用；"
            "若是直接测试，请在测试中通过 bind_thread_id 绑定会话 ID。"
        )
    payload = build_causal_report(
        thread_id=thread_id,
        include_refs=bool(include_refs),
        recent_question=recent_question or "",
    )
    if payload.get("error"):
        return f"报告生成失败：{payload.get('error')}"

    summary = [
        "- 报告已生成",
        f"- 覆盖章节：{len(payload.get('sections') or [])}",
        f"- 警告数：{len(payload.get('warnings') or [])}",
    ]
    md = payload.get("markdown", "")
    return "\n".join(summary) + f"\n\n```markdown\n{md}\n```"

