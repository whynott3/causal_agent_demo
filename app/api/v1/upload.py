"""CSV 上传接口（Sprint 2 / P2.T1 + P2.T8 子集）。

提供两个路由：
- ``POST /api/v1/upload/csv``：multipart 表单，入参 ``thread_id`` + ``file``，
  服务器把文件落到 ``<uploads_root>/<thread_id>/data.csv``，
  并同步把 :func:`tools.data_profile.profile_csv` 的结果写入
  ``<uploads_root>/<thread_id>/profile.json``。
- ``GET /api/v1/upload/{thread_id}/status``：前端切换 / 刷新会话时用来恢复
  "已上传"状态。

设计约束：
- ``thread_id`` 走白名单 ``^[A-Za-z0-9_-]{1,128}$``，严防路径穿越；
- 流式分块写盘，单文件 ≤ 100 MB；
- 编码兜底 ``utf-8`` → ``utf-8-sig`` → ``gbk``；
- 实际的耗时操作 (profile_csv) 用 :func:`asyncio.to_thread` 投递到线程池，
  避免阻塞 ASGI 事件循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from tools.data_profile import profile_csv

logger = logging.getLogger("causal_agent")

router = APIRouter()


# ---------------------------------------------------------------------------
# 常量与小工具
# ---------------------------------------------------------------------------

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_BYTES = 100 * 1024 * 1024
_CHUNK_SIZE = 1 << 20  # 1 MB
_ALLOWED_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk")


def uploads_root() -> Path:
    """上传根目录。

    优先读环境变量 ``APP_UPLOADS_DIR``（测试用），否则使用 ``uploads``
    （项目运行时 cwd 为 ``app/``，因此实际路径是 ``app/uploads``）。
    """
    override = os.getenv("APP_UPLOADS_DIR")
    return Path(override) if override else Path("uploads")


def _validate_thread_id(thread_id: str) -> str:
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        raise HTTPException(
            status_code=400,
            detail="thread_id 非法，仅允许字母 / 数字 / 下划线 / 短横线，长度 1-128。",
        )
    if any(ch in thread_id for ch in ("..", "/", "\\", "\x00")):
        raise HTTPException(status_code=400, detail="thread_id 包含非法字符。")
    return thread_id


def _safe_thread_dir(thread_id: str) -> Path:
    """构造受白名单约束的会话上传目录路径。"""
    _validate_thread_id(thread_id)
    base = uploads_root().resolve()
    target = (base / thread_id).resolve()
    if not str(target).startswith(str(base)):
        # 防御性检查：拒绝任何能逃逸到 base 之外的拼接结果
        raise HTTPException(status_code=400, detail="thread_id 路径越界。")
    return target


def _detect_encoding(path: Path) -> str | None:
    """依次尝试 utf-8 / utf-8-sig / gbk，返回第一个能解码全文的编码。"""
    raw = path.read_bytes()
    for enc in _ALLOWED_ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return None


def _to_jsonable(value: Any) -> Any:
    """把 numpy / pandas 标量转为可 json 化的原生类型，NaN/NaT → None。"""
    if value is None:
        return None
    try:
        import math

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except Exception:
        pass
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


# ---------------------------------------------------------------------------
# POST /upload/csv
# ---------------------------------------------------------------------------

@router.post("/upload/csv")
async def upload_csv(
    thread_id: str = Form(..., description="当前会话 ID"),
    file: UploadFile = File(..., description="CSV 文件"),
) -> dict[str, Any]:
    """上传当前会话所属的 CSV 数据集并自动生成数据画像。"""
    _validate_thread_id(thread_id)

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="仅支持 .csv 文件。")

    target_dir = _safe_thread_dir(thread_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "data.csv"
    tmp_path = target_path.with_suffix(".csv.part")

    total_bytes = 0
    try:
        with tmp_path.open("wb") as fp:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_BYTES:
                    fp.close()
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 100 MB 上限，已收到 {total_bytes} 字节。",
                    )
                fp.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error(f"写入 CSV 失败 {tmp_path}: {exc}")
        raise HTTPException(status_code=500, detail=f"写入失败：{exc}") from exc

    encoding = _detect_encoding(tmp_path)
    if encoding is None:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="无法以 utf-8 / utf-8-sig / gbk 解码该 CSV，请确认文件编码。",
        )
    logger.info(f"[upload] thread_id={thread_id!r} 使用 {encoding} 解码 CSV")

    tmp_path.replace(target_path)

    try:
        profile = await asyncio.to_thread(profile_csv, str(target_path), encoding)
    except Exception as exc:
        logger.error(f"profile_csv 失败 {target_path}: {exc}")
        raise HTTPException(status_code=500, detail=f"数据画像生成失败：{exc}") from exc

    profile_path = target_dir / "profile.json"
    try:
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.error(f"写入 profile.json 失败 {profile_path}: {exc}")
        raise HTTPException(status_code=500, detail=f"画像落盘失败：{exc}") from exc

    try:
        head_df = pd.read_csv(target_path, encoding=encoding, nrows=5)
        head_records = [
            {str(k): _to_jsonable(v) for k, v in row.items()}
            for row in head_df.to_dict(orient="records")
        ]
    except Exception as exc:
        logger.warning(f"head 预览读取失败 {target_path}: {exc}")
        head_records = []

    return {
        "path": str(target_path).replace("\\", "/"),
        "rows": int(profile["n_rows"]),
        "cols": int(profile["n_cols"]),
        "head": head_records,
        "encoding": encoding,
        "filename": filename,
        "profile_path": str(profile_path).replace("\\", "/"),
        "profile_summary": {
            "n_rows": int(profile["n_rows"]),
            "n_cols": int(profile["n_cols"]),
            "n_numeric": int(profile["n_numeric"]),
            "n_categorical": int(profile["n_categorical"]),
            "missing_overall": profile["missing_overall"],
            "warnings_count": len(profile.get("warnings") or []),
        },
    }


# ---------------------------------------------------------------------------
# GET /upload/{thread_id}/status
# ---------------------------------------------------------------------------

@router.get("/upload/{thread_id}/status")
async def upload_status(thread_id: str) -> dict[str, Any]:
    """查询当前会话是否已有上传文件，给前端会话切换 / 刷新恢复用。"""
    try:
        target_dir = _safe_thread_dir(thread_id)
    except HTTPException:
        return {
            "uploaded": False,
            "rows": None,
            "cols": None,
            "filename": None,
            "uploaded_at": None,
        }

    csv_path = target_dir / "data.csv"
    profile_path = target_dir / "profile.json"
    if not csv_path.exists():
        return {
            "uploaded": False,
            "rows": None,
            "cols": None,
            "filename": None,
            "uploaded_at": None,
        }

    rows: int | None = None
    cols: int | None = None
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            rows = int(data.get("n_rows")) if data.get("n_rows") is not None else None
            cols = int(data.get("n_cols")) if data.get("n_cols") is not None else None
        except Exception as exc:
            logger.warning(f"读取 profile.json 失败 {profile_path}: {exc}")

    try:
        uploaded_at = csv_path.stat().st_mtime
    except OSError:
        uploaded_at = None

    return {
        "uploaded": True,
        "rows": rows,
        "cols": cols,
        "filename": "data.csv",
        "uploaded_at": uploaded_at,
    }


# ---------------------------------------------------------------------------
# GET /upload/{thread_id}/profile
# ---------------------------------------------------------------------------

@router.get("/upload/{thread_id}/profile")
async def get_profile(thread_id: str) -> dict[str, Any]:
    """返回当前会话的完整数据画像（profile.json 内容）。

    供前端在上传成功后直接拉取并渲染卡片，不依赖 LLM 复述 JSON。
    """
    target_dir = _safe_thread_dir(thread_id)
    profile_path = target_dir / "profile.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="当前会话尚未生成数据画像，请先上传 CSV。")
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"读取 profile.json 失败 {profile_path}: {exc}")
        raise HTTPException(status_code=500, detail=f"读取数据画像失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 内部辅助：被 agents.causal_agent.clear_messages 复用
# ---------------------------------------------------------------------------

def cleanup_thread_uploads(thread_id: str) -> bool:
    """同步清理 ``<uploads_root>/<thread_id>/`` 整个目录。

    被 :func:`agents.causal_agent.clear_messages` 调用；失败时仅 warning。
    返回是否真正删除过目录（不存在时返回 False）。
    """
    if not isinstance(thread_id, str) or not _THREAD_ID_RE.match(thread_id):
        logger.warning(f"cleanup_thread_uploads: thread_id 不合法，跳过 {thread_id!r}")
        return False
    try:
        target_dir = uploads_root().resolve() / thread_id
        if not target_dir.exists():
            return False
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.info(f"已清理上传目录 {target_dir}")
        return True
    except Exception as exc:
        logger.warning(f"cleanup_thread_uploads 失败 {thread_id!r}: {exc}")
        return False
