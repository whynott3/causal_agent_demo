"""causal_discovery 工具测试（Sprint 4）。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.runtime_context import bind_thread_id
from tools.causal_discovery import (
    _merge_edges,
    run_causal_discovery,
    run_uploaded_causal_discovery,
)
from tools.data_profile import profile_csv


def _mk_simple_numeric_csv(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, size=n)
    x2 = 0.8 * x1 + rng.normal(0, 0.2, size=n)
    x3 = 0.7 * x2 + rng.normal(0, 0.2, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    p = path / "data.csv"
    df.to_csv(p, index=False)
    return p


def test_run_causal_discovery_basic_pc(tmp_path: Path) -> None:
    csv_path = _mk_simple_numeric_csv(tmp_path, n=500)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert out["algorithm"] == "PC"
    assert isinstance(out["nodes"], list) and len(out["nodes"]) == 3
    assert isinstance(out["edges"], list)
    assert isinstance(out["mermaid"], str) and "graph LR" in out["mermaid"]
    assert isinstance(out["elapsed_s"], float)


def test_small_sample_blocking(tmp_path: Path) -> None:
    csv_path = _mk_simple_numeric_csv(tmp_path, n=80)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert out["error"] is not None
    assert any("样本量过小" in w for w in out["warnings"])


def test_cols_greater_than_rows_blocking(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # 让 n_rows >= 100，避免先命中“小样本”阻断分支
    df = pd.DataFrame({f"v{i}": rng.normal(size=150) for i in range(200)})
    csv_path = tmp_path / "wide.csv"
    df.to_csv(csv_path, index=False)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert out["error"] is not None
    assert any("变量数大于样本量" in w for w in out["warnings"])


def test_non_numeric_columns_are_dropped_with_warning(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "cat": ["a" if i % 2 else "b" for i in range(n)],
    })
    csv_path = tmp_path / "mix.csv"
    df.to_csv(csv_path, index=False)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert out["error"] is None
    assert "cat" not in out["nodes"]
    assert any("非数值列" in w for w in out["warnings"])


def test_missing_values_generate_warning(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    df.loc[df.sample(20, random_state=1).index, "x2"] = np.nan
    csv_path = tmp_path / "missing.csv"
    df.to_csv(csv_path, index=False)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert out["error"] is None
    assert any("缺失值" in w for w in out["warnings"])


def test_notears_not_supported(tmp_path: Path) -> None:
    csv_path = _mk_simple_numeric_csv(tmp_path, n=300)
    out = run_causal_discovery(str(csv_path), algorithm="NOTEARS")
    assert out["error"] is not None
    assert "暂不支持执行 NOTEARS" in str(out["error"])


def test_tool_reads_uploaded_files_and_writes_dag_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads = tmp_path / "uploads"
    tid = "causal-test-1"
    thread_dir = uploads / tid
    thread_dir.mkdir(parents=True)

    csv_path = _mk_simple_numeric_csv(thread_dir, n=400)
    prof = profile_csv(str(csv_path))
    (thread_dir / "profile.json").write_text(
        json.dumps(prof, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_UPLOADS_DIR", str(uploads))

    with bind_thread_id(tid):
        md = run_uploaded_causal_discovery.invoke({"algorithm": "PC"})

    assert "算法：PC" in md
    assert "```json" in md
    dag_path = thread_dir / "dag.json"
    assert dag_path.exists()
    dag = json.loads(dag_path.read_text(encoding="utf-8"))
    assert dag["algorithm"] == "PC"
    assert "created_at" in dag


def test_tool_without_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_UPLOADS_DIR", str(tmp_path / "uploads"))
    out = run_uploaded_causal_discovery.invoke({"algorithm": "PC"})
    assert "未能读取当前会话上下文" in out


def test_mermaid_contains_graph_lr(tmp_path: Path) -> None:
    csv_path = _mk_simple_numeric_csv(tmp_path, n=350)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    assert "graph LR" in out["mermaid"]


def test_edges_json_serializable(tmp_path: Path) -> None:
    csv_path = _mk_simple_numeric_csv(tmp_path, n=350)
    out = run_causal_discovery(str(csv_path), algorithm="PC")
    s = json.dumps(out["edges"], ensure_ascii=False)
    assert isinstance(s, str)


def test_timeout_handling_via_monkeypatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.causal_discovery as cd

    csv_path = _mk_simple_numeric_csv(tmp_path, n=300)

    def slow(*args, **kwargs):
        time.sleep(0.2)
        return {
            "algorithm": "PC",
            "nodes": ["x1", "x2"],
            "edges": [],
            "mermaid": "graph LR",
            "elapsed_s": 0.2,
            "warnings": [],
            "options": {},
            "error": None,
        }

    monkeypatch.setattr(cd, "_run_internal", slow)
    out = cd.run_causal_discovery(str(csv_path), algorithm="PC", timeout_s=0.05)
    assert out["error"] == "timeout"
    assert any("超时" in w for w in out["warnings"])


def test_ges_returns_warning_not_fake_success(tmp_path: Path) -> None:
    # 当前环境下 GES 可能因依赖兼容问题失败；要求返回明确 error/warning，而非伪造成功图。
    csv_path = _mk_simple_numeric_csv(tmp_path, n=400)
    out = run_causal_discovery(str(csv_path), algorithm="GES")
    if out["error"] is None:
        # 若环境正好可用，至少结构完整
        assert out["algorithm"] == "GES"
        assert isinstance(out["edges"], list)
    else:
        assert "GES" in str(out["error"])
        assert any("失败" in w or "兼容" in w for w in out["warnings"])


def test_merge_edges_prefers_undirected_on_conflict() -> None:
    edges = [
        {"from": "Weather", "to": "Footfall", "type": "undirected", "confidence": None},
        {"from": "Weather", "to": "Footfall", "type": "directed", "confidence": None},
        {"from": "Footfall", "to": "Weather", "type": "directed", "confidence": None},
    ]
    merged = _merge_edges(edges)
    wf = [
        e for e in merged
        if {e.get("from"), e.get("to")} == {"Weather", "Footfall"}
    ]
    assert len(wf) == 1
    assert wf[0]["type"] == "undirected"

