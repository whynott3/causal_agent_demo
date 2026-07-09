"""轻量评测运行器（Sprint 5 / P4.1）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import yaml


def load_cases(cases_path: str) -> list[dict[str, Any]]:
    p = Path(cases_path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cases" in raw:
        raw = raw["cases"]
    if not isinstance(raw, list):
        raise ValueError("cases 文件格式错误：应为 list 或 {cases: list}")
    out: list[dict[str, Any]] = []
    for i, case in enumerate(raw, 1):
        if not isinstance(case, dict):
            continue
        c = dict(case)
        c.setdefault("id", f"case-{i:03d}")
        c.setdefault("category", "misc")
        c.setdefault("messages", [])
        c.setdefault("must_include", [])
        c.setdefault("must_not_include", [])
        c.setdefault("required_sections", [])
        c.setdefault("tags", [])
        out.append(c)
    return out


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def keyword_coverage(answer: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    text = _normalize_text(answer)
    hit = 0
    for kw in must_include:
        if _normalize_text(str(kw)) in text:
            hit += 1
    return hit / max(1, len(must_include))


def forbidden_violations(answer: str, must_not_include: list[str]) -> list[str]:
    text = _normalize_text(answer)
    bad: list[str] = []
    for kw in must_not_include:
        s = _normalize_text(str(kw))
        if s and s in text:
            bad.append(str(kw))
    return bad


def section_coverage(answer: str, required_sections: list[str]) -> float:
    if not required_sections:
        return 1.0
    text = _normalize_text(answer)
    hit = 0
    for sec in required_sections:
        if _normalize_text(str(sec)) in text:
            hit += 1
    return hit / max(1, len(required_sections))


def score_case(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    kw_score = keyword_coverage(answer, case.get("must_include") or [])
    sec_score = section_coverage(answer, case.get("required_sections") or [])
    violations = forbidden_violations(answer, case.get("must_not_include") or [])
    pass_flag = kw_score >= 0.6 and sec_score >= 0.6 and not violations
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "pass": pass_flag,
        "keyword_coverage": round(kw_score, 4),
        "section_coverage": round(sec_score, 4),
        "forbidden_violations": violations,
    }


async def _agent_responder(messages: list[str], disable_web: bool = True) -> str:
    from agents.causal_agent import causal_chat

    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    answer = ""
    queued = list(messages or [])
    if disable_web and queued:
        queued[0] = (
            "（评测模式：优先基于本地知识作答，除非问题明确要求联网检索）\n"
            + queued[0]
        )
    for msg in queued:
        chunks: list[str] = []
        async for chunk in causal_chat(msg, "", thread_id):
            chunks.append(str(chunk))
        answer = "".join(chunks)
    return answer


def build_default_responder(disable_web: bool = True) -> Callable[[list[str]], str]:
    def _call(messages: list[str]) -> str:
        return asyncio.run(_agent_responder(messages, disable_web=disable_web))

    return _call


def run_evaluation(
    cases: list[dict[str, Any]],
    responder: Callable[[list[str]], str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        answer = responder(case.get("messages") or [])
        row = score_case(answer, case)
        row["answer_preview"] = (answer or "")[:200]
        rows.append(row)
    total = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round((passed / total) if total else 0.0, 4),
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"- Total: {report.get('total', 0)}")
    lines.append(f"- Passed: {report.get('passed', 0)}")
    lines.append(f"- Pass rate: {report.get('pass_rate', 0):.2%}")
    lines.append("")
    lines.append("## Cases")
    for row in report.get("rows", []):
        lines.append(
            f"- `{row['id']}` ({row['category']}): "
            f"{'PASS' if row['pass'] else 'FAIL'} | "
            f"kw={row['keyword_coverage']:.2f}, sec={row['section_coverage']:.2f}, "
            f"violations={len(row['forbidden_violations'])}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal assistant eval cases.")
    parser.add_argument("--cases", default="app/eval/cases.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--report-file", default="")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    disable_web = os.getenv("EVAL_DISABLE_WEB", "1") != "0"
    responder = build_default_responder(disable_web=disable_web)
    report = run_evaluation(cases, responder)

    if args.output == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        text = render_markdown(report)

    if args.report_file:
        rp = Path(args.report_file)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

