"""recommend_discovery_algorithm + recommend_causal_discovery_algorithms 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.runtime_context import bind_thread_id
from tools.algo_recommend import (
    recommend_causal_discovery_algorithms,
    recommend_discovery_algorithm,
)


# ---------------------------------------------------------------------------
# 构造 profile 的小工具
# ---------------------------------------------------------------------------

def _make_profile(
    n_rows: int,
    n_cols: int,
    n_numeric: int,
    n_categorical: int = 0,
    missing_overall: float = 0.0,
    extra_columns: list[dict] | None = None,
) -> dict:
    """构造一个最小可用的 profile dict（仅供启发式规则消费）。"""
    cols: list[dict] = []
    for i in range(n_numeric):
        cols.append({"name": f"num_{i}", "dtype": "numeric"})
    for i in range(n_categorical):
        cols.append({"name": f"cat_{i}", "dtype": "categorical"})
    if extra_columns:
        cols.extend(extra_columns)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols if n_cols else len(cols),
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "missing_overall": missing_overall,
        "columns": cols,
        "warnings": [],
        "top_correlations": [],
        "duplicates": 0,
    }


def _algorithms(rec: dict) -> list[str]:
    return [r["algorithm"] for r in rec.get("recommendations") or []]


def _priority(rec: dict, algo: str) -> int | None:
    for r in rec.get("recommendations") or []:
        if r["algorithm"] == algo:
            return r["priority"]
    return None


# ---------------------------------------------------------------------------
# 1) 理想场景：全数值 n=500, p=10 → PC priority=1, GES 存在
# ---------------------------------------------------------------------------

def test_ideal_full_numeric_pc_first() -> None:
    profile = _make_profile(n_rows=500, n_cols=10, n_numeric=10)
    out = recommend_discovery_algorithm(profile)
    assert out["blocking"] is False
    algos = _algorithms(out)
    assert "PC" in algos
    assert "GES" in algos
    assert _priority(out, "PC") == 1
    # JSON 可序列化
    json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2) latent 提示：user_hints 含「隐藏混杂」→ FCI priority=1，PC/GES 移到 not_recommended
# ---------------------------------------------------------------------------

def test_latent_hints_prefer_fci() -> None:
    profile = _make_profile(n_rows=500, n_cols=10, n_numeric=10)
    out = recommend_discovery_algorithm(profile, user_hints="我担心数据中可能存在未观测的隐藏混杂变量")
    assert _priority(out, "FCI") == 1
    # latent 时 PC/GES 不在 recommendations
    assert "PC" not in _algorithms(out)
    assert "GES" not in _algorithms(out)
    not_rec_algos = [r["algorithm"] for r in out.get("not_recommended") or []]
    assert "PC" in not_rec_algos
    assert "GES" in not_rec_algos


def test_latent_hints_english_keyword() -> None:
    profile = _make_profile(n_rows=500, n_cols=10, n_numeric=10)
    out = recommend_discovery_algorithm(profile, user_hints="There may be unobserved confounders.")
    assert _priority(out, "FCI") == 1


# ---------------------------------------------------------------------------
# 3) 高维：n_cols=35 → NOTEARS 出现
# ---------------------------------------------------------------------------

def test_high_dim_recommends_notears() -> None:
    profile = _make_profile(n_rows=600, n_cols=35, n_numeric=35)
    out = recommend_discovery_algorithm(profile)
    assert "NOTEARS" in _algorithms(out)


# ---------------------------------------------------------------------------
# 4) 大样本：n_rows=2000 → NOTEARS 出现
# ---------------------------------------------------------------------------

def test_large_sample_recommends_notears() -> None:
    profile = _make_profile(n_rows=2000, n_cols=15, n_numeric=15)
    out = recommend_discovery_algorithm(profile)
    assert "NOTEARS" in _algorithms(out)


# ---------------------------------------------------------------------------
# 5) blocking：n_rows=50 → blocking=True, recommendations 空
# ---------------------------------------------------------------------------

def test_blocking_small_sample() -> None:
    profile = _make_profile(n_rows=50, n_cols=5, n_numeric=5)
    out = recommend_discovery_algorithm(profile)
    assert out["blocking"] is True
    assert "样本量过小" in (out["blocking_reason"] or "")
    assert out["recommendations"] == []


# ---------------------------------------------------------------------------
# 6) blocking：cols > rows
# ---------------------------------------------------------------------------

def test_blocking_cols_greater_than_rows() -> None:
    # 注意：n_rows 要 ≥ 100 才不会先命中"样本量过小"阻断分支
    profile = _make_profile(n_rows=150, n_cols=200, n_numeric=200)
    out = recommend_discovery_algorithm(profile)
    assert out["blocking"] is True
    assert "变量数大于样本量" in (out["blocking_reason"] or "")


# ---------------------------------------------------------------------------
# 7) 高缺失：missing_overall=0.25 → global_warnings 含缺失提示
# ---------------------------------------------------------------------------

def test_high_missing_global_warning() -> None:
    profile = _make_profile(n_rows=500, n_cols=10, n_numeric=10, missing_overall=0.25)
    out = recommend_discovery_algorithm(profile)
    assert out["blocking"] is False
    assert any("缺失率" in w for w in out["global_warnings"])


# ---------------------------------------------------------------------------
# 8) datetime 列 → PCMCI 候选 + suspected_timeseries
# ---------------------------------------------------------------------------

def test_datetime_suggests_pcmci() -> None:
    extra = [{"name": "signup_date", "dtype": "datetime"}]
    profile = _make_profile(n_rows=500, n_cols=11, n_numeric=10, extra_columns=extra)
    out = recommend_discovery_algorithm(profile)
    assert out["data_signals"]["suspected_timeseries"] is True
    assert "PCMCI" in _algorithms(out)


def test_datetime_but_small_sample_no_pcmci() -> None:
    extra = [{"name": "ts", "dtype": "datetime"}]
    profile = _make_profile(n_rows=120, n_cols=6, n_numeric=5, extra_columns=extra)
    out = recommend_discovery_algorithm(profile, user_hints="")
    # 样本量 < 200 时 suspected_timeseries 仍 True（>=50），PCMCI 仍可出现，
    # 但应通过 global_warnings 标记"样本偏少"
    assert out["data_signals"]["suspected_timeseries"] is True
    assert any("样本量偏少" in w for w in out["global_warnings"])


# ---------------------------------------------------------------------------
# 9) 全分类无数值 → 阻断
# ---------------------------------------------------------------------------

def test_all_categorical_blocking() -> None:
    profile = _make_profile(n_rows=500, n_cols=5, n_numeric=0, n_categorical=5)
    out = recommend_discovery_algorithm(profile)
    assert out["blocking"] is True
    assert "数值" in (out["blocking_reason"] or "")


# ---------------------------------------------------------------------------
# 10) JSON 可序列化（关键字段非 numpy 类型）
# ---------------------------------------------------------------------------

def test_payload_is_json_serializable() -> None:
    profile = _make_profile(n_rows=500, n_cols=10, n_numeric=10)
    out = recommend_discovery_algorithm(profile, user_hints="可能存在隐藏混杂")
    s = json.dumps(out, ensure_ascii=False)
    assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------------------
# 11) @tool 集成：基于 thread_id 读 profile.json + data.csv
# ---------------------------------------------------------------------------

def test_tool_integration_full_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({f"v{i}": rng.normal(0, 1, size=n) for i in range(8)})

    uploads = tmp_path / "uploads"
    tid = "algo-test-1"
    (uploads / tid).mkdir(parents=True)
    df.to_csv(uploads / tid / "data.csv", index=False)

    from tools.data_profile import profile_csv  # 复用真实 profile 生成
    profile = profile_csv(str(uploads / tid / "data.csv"))
    (uploads / tid / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv("APP_UPLOADS_DIR", str(uploads))

    # Sprint 3 P0：工具不再接收 thread_id 参数；通过 bind_thread_id 注入
    with bind_thread_id(tid):
        md = recommend_causal_discovery_algorithms.invoke({"user_hints": ""})
    assert "候选算法" in md
    # 末尾应附严格合法 JSON 块
    assert "```json" in md
    json_part = md.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(json_part)
    assert "recommendations" in payload
    assert payload["blocking"] is False
    assert any(r["algorithm"] == "PC" for r in payload["recommendations"])


# ---------------------------------------------------------------------------
# 12) @tool 友好错误：未上传 / 不合法 thread_id 不抛栈
# ---------------------------------------------------------------------------

def test_tool_missing_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    with bind_thread_id("no-such-thread"):
        md = recommend_causal_discovery_algorithms.invoke({"user_hints": ""})
    assert "未能从当前会话目录中找到数据画像文件" in md
    # 兜底文案要明确禁止 LLM 编造推荐
    assert "禁止" in md and "编造" in md


def test_tool_bad_thread_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    with bind_thread_id("../escape"):
        md = recommend_causal_discovery_algorithms.invoke({"user_hints": ""})
    assert "thread_id" in md and "不合法" in md


def test_tool_without_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未绑定 thread_id 时返回友好提示。"""
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    md = recommend_causal_discovery_algorithms.invoke({"user_hints": ""})
    assert "未能读取当前会话上下文" in md


# ---------------------------------------------------------------------------
# 13) 非高斯检测：构造 chi-square 分布的数值列 → LiNGAM 出现
# ---------------------------------------------------------------------------

def test_lingam_when_non_gaussian(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({
        "a": rng.chisquare(df=2, size=n),         # 高峰度
        "b": rng.exponential(scale=1.0, size=n),  # 高峰度
        "c": rng.normal(0, 1, size=n),            # 高斯
        "d": rng.normal(0, 1, size=n),            # 高斯
        "e": rng.lognormal(0, 1, size=n),         # 高峰度
    })

    uploads = tmp_path / "uploads"
    tid = "lingam-test"
    (uploads / tid).mkdir(parents=True)
    df.to_csv(uploads / tid / "data.csv", index=False)

    from tools.data_profile import profile_csv
    profile = profile_csv(str(uploads / tid / "data.csv"))
    (uploads / tid / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv("APP_UPLOADS_DIR", str(uploads))

    with bind_thread_id(tid):
        md = recommend_causal_discovery_algorithms.invoke({"user_hints": ""})
    json_part = md.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(json_part)
    algos = [r["algorithm"] for r in payload["recommendations"]]
    assert "LiNGAM" in algos
