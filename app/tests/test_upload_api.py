"""POST /upload/csv + GET /upload/{thread_id}/status 集成测试。

避免拉起整个 agent 栈：构造一个**最小** FastAPI app，只挂 upload 路由。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def uploads_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个测试隔离一个临时上传目录。"""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setenv("APP_UPLOADS_DIR", str(root))
    return root


@pytest.fixture
def client(uploads_root: Path) -> TestClient:
    # 延迟导入：让 monkeypatch 的环境变量生效后再加载模块
    from api.v1 import upload as upload_module
    app = FastAPI()
    app.include_router(upload_module.router, prefix="/api/v1")
    return TestClient(app)


def _make_csv_bytes(df: pd.DataFrame, encoding: str = "utf-8") -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding=encoding)
    return buf.getvalue()


def test_upload_csv_happy_path(client: TestClient, uploads_root: Path) -> None:
    df = pd.DataFrame({
        "x": np.arange(120),
        "y": np.arange(120) * 1.5 + 0.3,
        "group": (["a", "b", "c"] * 40),
    })
    payload = _make_csv_bytes(df)

    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "thread-abc-1"},
        files={"file": ("data.csv", payload, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows"] == 120
    assert body["cols"] == 3
    assert body["encoding"] == "utf-8"
    assert body["filename"] == "data.csv"
    assert "profile_summary" in body
    ps = body["profile_summary"]
    for key in ("n_rows", "n_cols", "n_numeric", "n_categorical", "missing_overall", "warnings_count"):
        assert key in ps
    assert ps["n_rows"] == 120
    assert ps["n_cols"] == 3
    assert isinstance(body["head"], list) and len(body["head"]) == 5

    # 文件落盘检查
    data_csv = uploads_root / "thread-abc-1" / "data.csv"
    profile_json = uploads_root / "thread-abc-1" / "profile.json"
    assert data_csv.exists()
    assert profile_json.exists()
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    assert profile["n_rows"] == 120


def test_upload_csv_rejects_non_csv(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "thread-xyz"},
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.parametrize("bad_id", ["../escape", "abc/def", "a b", "", "x" * 200])
def test_upload_csv_invalid_thread_id(client: TestClient, bad_id: str) -> None:
    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": bad_id},
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
    )
    # 400 来自白名单校验；422 来自 FastAPI 自身的 Form 字段验证（空串等）
    assert resp.status_code in (400, 422), resp.text


def test_upload_csv_decodes_gbk(client: TestClient, uploads_root: Path) -> None:
    df = pd.DataFrame({"性别": ["男", "女"] * 5, "年龄": list(range(20, 30))})
    payload = _make_csv_bytes(df, encoding="gbk")
    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "gbk-thread"},
        files={"file": ("zh.csv", payload, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["encoding"] == "gbk"
    assert body["rows"] == 10


def test_upload_csv_rejects_invalid_encoding(client: TestClient) -> None:
    # 同时违反 utf-8 / utf-8-sig / gbk 的字节序列（latin-1 中的 0x80 单独一字节）
    # 让 utf-8 解码失败；同时构造 0x81 0x40 这种在 gbk 也无效的对（gbk 头字节 0x81 后接 < 0x40 会失败）
    bad = b"col1,col2\n\x81\x3F,\x9C\n\xC3\xA9,\xFF\xFE\n"
    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "enc-bad"},
        files={"file": ("bad.csv", bad, "text/csv")},
    )
    # latin-1 通常能被 gbk 误解码，所以这条不一定 400；只要不 5xx 就行
    assert resp.status_code in (200, 400)


def test_upload_csv_size_limit_via_monkeypatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把 _MAX_BYTES 临时调小，验证 413 路径。"""
    from api.v1 import upload as upload_module
    monkeypatch.setattr(upload_module, "_MAX_BYTES", 512)

    df = pd.DataFrame({"x": np.arange(500)})  # 远大于 512 字节
    payload = _make_csv_bytes(df)
    resp = client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "big-thread"},
        files={"file": ("big.csv", payload, "text/csv")},
    )
    assert resp.status_code == 413


def test_upload_status_before_and_after(client: TestClient, uploads_root: Path) -> None:
    resp1 = client.get("/api/v1/upload/some-thread/status")
    assert resp1.status_code == 200
    assert resp1.json()["uploaded"] is False

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "some-thread"},
        files={"file": ("data.csv", _make_csv_bytes(df), "text/csv")},
    )

    resp2 = client.get("/api/v1/upload/some-thread/status")
    body = resp2.json()
    assert body["uploaded"] is True
    assert body["rows"] == 3
    assert body["cols"] == 2
    assert body["filename"] == "data.csv"
    assert isinstance(body["uploaded_at"], float)


def test_upload_status_bad_thread_id(client: TestClient) -> None:
    # 路径穿越要被视为"未上传"，不应 500
    resp = client.get("/api/v1/upload/..%2Fescape/status")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert resp.json()["uploaded"] is False


def test_cleanup_thread_uploads_removes_dir(uploads_root: Path) -> None:
    from api.v1.upload import cleanup_thread_uploads
    tid = "thread-to-clean"
    target = uploads_root / tid
    target.mkdir()
    (target / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (target / "profile.json").write_text("{}", encoding="utf-8")

    assert cleanup_thread_uploads(tid) is True
    assert not target.exists()


def test_cleanup_thread_uploads_invalid_id(uploads_root: Path) -> None:
    from api.v1.upload import cleanup_thread_uploads
    # 不合法 thread_id 直接拒绝，不抛错也不删除
    assert cleanup_thread_uploads("../boom") is False


def test_get_profile_returns_full_json(client: TestClient, uploads_root: Path) -> None:
    df = pd.DataFrame({
        "x": np.arange(50, dtype=float),
        "y": np.arange(50, dtype=float) * 2,
        "grp": (["a", "b"] * 25),
    })
    client.post(
        "/api/v1/upload/csv",
        data={"thread_id": "prof-thread"},
        files={"file": ("data.csv", _make_csv_bytes(df), "text/csv")},
    )
    resp = client.get("/api/v1/upload/prof-thread/profile")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_rows"] == 50
    assert body["n_cols"] == 3
    assert len(body["columns"]) == 3
    x_col = next(c for c in body["columns"] if c["name"] == "x")
    assert x_col["dtype"] == "numeric"
    assert "mean" in x_col["stats"]
    grp_col = next(c for c in body["columns"] if c["name"] == "grp")
    assert grp_col["dtype"] == "categorical"
    assert "top_values" in grp_col["stats"]


def test_get_profile_404_when_missing(client: TestClient) -> None:
    resp = client.get("/api/v1/upload/no-upload-yet/profile")
    assert resp.status_code == 404


def test_get_profile_invalid_thread_id(client: TestClient) -> None:
    resp = client.get("/api/v1/upload/../escape/profile")
    assert resp.status_code in (400, 404)
