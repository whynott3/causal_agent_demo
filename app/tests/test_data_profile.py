"""data_profile.profile_csv 单元测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.data_profile import profile_csv, summarize_uploaded_dataset


@pytest.fixture
def mixed_csv(tmp_path: Path) -> Path:
    """构造一个 8 列混合类型 CSV：numeric / categorical / boolean / datetime / text + 缺失 + 常量 + ID。"""
    rng = np.random.default_rng(seed=0)
    n = 250
    df = pd.DataFrame({
        "edu": rng.choice([12, 16, 18, 22], size=n),
        "income": rng.normal(5000, 1500, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "is_member": rng.choice([0, 1], size=n),
        "user_id": [f"u{i:05d}" for i in range(n)],
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "sparse_col": [np.nan] * 150 + list(rng.normal(0, 1, size=100)),
        "const_col": [1] * n,
    })
    # 注入若干 NaN 到 income
    df.loc[df.sample(15, random_state=1).index, "income"] = np.nan
    p = tmp_path / "demo.csv"
    df.to_csv(p, index=False)
    return p


def _col(profile: dict, name: str) -> dict:
    for c in profile["columns"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


def test_profile_basic_shape(mixed_csv: Path) -> None:
    profile = profile_csv(str(mixed_csv))
    assert profile["n_rows"] == 250
    assert profile["n_cols"] == 8
    assert profile["n_numeric"] >= 3  # edu, income, sparse_col, const_col
    assert profile["n_categorical"] >= 1  # gender
    assert isinstance(profile["columns"], list)
    assert isinstance(profile["warnings"], list)
    assert isinstance(profile["top_correlations"], list)
    # 整张 profile 必须可 json 序列化（前端 / API 都依赖这一点）
    json.dumps(profile, ensure_ascii=False)


def test_profile_dtype_classification(mixed_csv: Path) -> None:
    profile = profile_csv(str(mixed_csv))
    assert _col(profile, "edu")["dtype"] == "numeric"
    assert _col(profile, "income")["dtype"] == "numeric"
    assert _col(profile, "gender")["dtype"] == "categorical"
    assert _col(profile, "is_member")["dtype"] == "boolean"
    assert _col(profile, "signup_date")["dtype"] == "datetime"
    assert _col(profile, "user_id")["dtype"] == "text"
    assert _col(profile, "sparse_col")["dtype"] == "numeric"
    # 常量列单值，n_unique==1，按规则保持 numeric（不应被误归 boolean）
    assert _col(profile, "const_col")["dtype"] == "numeric"


def test_profile_numeric_stats_and_missing(mixed_csv: Path) -> None:
    income = _col(profile_csv(str(mixed_csv)), "income")
    assert income["dtype"] == "numeric"
    assert income["missing_rate"] is not None and income["missing_rate"] > 0
    stats = income["stats"]
    for k in ("mean", "std", "p25", "p50", "p75", "min", "max"):
        assert k in stats
        assert stats[k] is None or isinstance(stats[k], (int, float))


def test_profile_top_correlations_sorted_and_capped(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=42)
    n = 400
    # 20 个数值列 → 190 对组合，远大于 15
    data = {f"v{i}": rng.normal(0, 1, size=n) for i in range(20)}
    df = pd.DataFrame(data)
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)
    profile = profile_csv(str(p))

    corrs = profile["top_correlations"]
    assert 0 < len(corrs) <= 15  # 上限 15
    # 按 |Pearson| 降序
    abs_p = [abs(c["pearson"] or 0) for c in corrs]
    assert abs_p == sorted(abs_p, reverse=True)


def test_profile_warnings_heuristics(mixed_csv: Path) -> None:
    profile = profile_csv(str(mixed_csv))
    warns = " | ".join(profile["warnings"])
    assert "user_id" in warns and "疑似 ID 列" in warns
    assert "sparse_col" in warns and "缺失率过高" in warns
    assert "const_col" in warns and "常量列" in warns
    assert "signup_date" in warns and "疑似时间戳" in warns


def test_profile_small_sample_warning(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.arange(50), "y": np.arange(50) * 2.0})
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    profile = profile_csv(str(p))
    assert any("样本量较小" in w for w in profile["warnings"])


def test_profile_wide_dataset_warning(tmp_path: Path) -> None:
    df = pd.DataFrame({f"c{i}": np.arange(5).astype(float) for i in range(50)})
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    profile = profile_csv(str(p))
    assert any("变量数大于样本量" in w for w in profile["warnings"])


def test_profile_handles_gbk_encoding(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "性别": ["男", "女", "男", "女", "男"] * 4,
        "年龄": list(range(20, 40)),
        "收入": [3000.0, 4500.0, 5200.5, 6100.2, 7800.8] * 4,
    })
    p = tmp_path / "gbk.csv"
    df.to_csv(p, index=False, encoding="gbk")
    profile = profile_csv(str(p), encoding="gbk")
    assert profile["n_rows"] == 20
    assert any(c["name"] == "性别" for c in profile["columns"])
    assert _col(profile, "性别")["dtype"] in ("categorical", "boolean")


def test_summarize_uploaded_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mixed_csv: Path) -> None:
    profile = profile_csv(str(mixed_csv))
    tid = "test-thread-001"
    uploads = tmp_path / "uploads"
    (uploads / tid).mkdir(parents=True)
    (uploads / tid / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv("APP_UPLOADS_DIR", str(uploads))

    md = summarize_uploaded_dataset.invoke({"thread_id": tid})
    assert "250 行 × 8 列" in md
    assert "整体缺失率" in md
    assert "数据警告" in md
    assert "数值列分布摘要" in md
    assert "edu" in md or "income" in md


def test_summarize_uploaded_dataset_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    md = summarize_uploaded_dataset.invoke({"thread_id": "not-exist"})
    assert "尚未上传" in md


def test_summarize_uploaded_dataset_bad_thread_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    md = summarize_uploaded_dataset.invoke({"thread_id": "../escape"})
    assert "thread_id" in md and "不合法" in md
