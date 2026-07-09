"""报告导出接口（Sprint 6 / P5）。"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response

from tools.report_export import build_causal_report

router = APIRouter()


@router.get("/report/{thread_id}")
async def get_causal_report(
    thread_id: str,
    format: Literal["markdown", "json"] = Query("markdown"),
    download: bool = Query(False),
    include_refs: bool = Query(True),
) -> Response:
    """导出指定会话的因果分析报告。"""
    payload = build_causal_report(thread_id=thread_id, include_refs=include_refs)
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=payload["error"])

    if format == "json":
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="causal-report-{thread_id}.json"'
        return Response(
            content=body,
            media_type="application/json; charset=utf-8",
            headers=headers,
        )

    md = payload.get("markdown", "")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="causal-report-{thread_id}.md"'
    return PlainTextResponse(content=md, headers=headers)

