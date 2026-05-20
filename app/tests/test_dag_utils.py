"""DAG 工具集的最小用例（Sprint 1 / P1.3）。

覆盖：parse_dag_text、find_frontdoor_paths、find_backdoor_paths、
suggest_adjustment_set（含 mediators 字段）、to_mermaid。
"""

from __future__ import annotations

import pytest

from tools.dag_utils import (
    find_backdoor_paths,
    find_frontdoor_paths,
    parse_dag_text,
    suggest_adjustment_set,
    to_mermaid,
)


# ---------------------------------------------------------------------------
# 用例 1：parse_dag_text - 多种边写法、行分隔、环检测
# ---------------------------------------------------------------------------

def test_parse_dag_text_mixed_separators_and_arrows():
    text = """
    # 这是注释
    教育水平 -> 收入
    家庭背景 -> 教育水平；家庭背景→收入，家庭背景 --> 收入
    """
    info = parse_dag_text(text)

    assert info["nodes"] == ["教育水平", "收入", "家庭背景"]
    assert ("教育水平", "收入") in info["edges"]
    assert ("家庭背景", "教育水平") in info["edges"]
    assert ("家庭背景", "收入") in info["edges"]
    assert info["has_cycle"] is False
    assert info["cycles"] == []


def test_parse_dag_text_detects_cycle():
    info = parse_dag_text("A -> B; B -> C; C -> A")
    assert info["has_cycle"] is True
    assert info["cycles"]
    # 第一个环至少包含 3 个节点
    assert len(info["cycles"][0]) == 3


# ---------------------------------------------------------------------------
# 用例 2：find_frontdoor_paths - 因果路径 / 有向简单路径
# ---------------------------------------------------------------------------

def test_find_frontdoor_paths_smoking_to_cancer():
    nodes = ["吸烟", "焦油", "肺癌", "基因"]
    edges = [("吸烟", "焦油"), ("焦油", "肺癌"), ("基因", "吸烟"), ("基因", "肺癌")]
    paths = find_frontdoor_paths(nodes, edges, "吸烟", "肺癌")
    assert paths == [["吸烟", "焦油", "肺癌"]]


def test_find_frontdoor_paths_no_path_returns_empty():
    paths = find_frontdoor_paths(["A", "B", "C"], [("A", "C")], "B", "C")
    assert paths == []


def test_find_frontdoor_paths_missing_node_raises():
    with pytest.raises(ValueError):
        find_frontdoor_paths(["A", "B"], [("A", "B")], "X", "B")


# ---------------------------------------------------------------------------
# 用例 3：find_backdoor_paths - 后门路径定义校验
# ---------------------------------------------------------------------------

def test_find_backdoor_paths_classic_confounder():
    # T <- C -> Y：后门路径恰好 1 条
    nodes = ["T", "Y", "C"]
    edges = [("T", "Y"), ("C", "T"), ("C", "Y")]
    paths = find_backdoor_paths(nodes, edges, "T", "Y")
    assert paths == [["T", "C", "Y"]]


def test_find_backdoor_paths_excludes_pure_causal_path():
    # T -> M -> Y：只有因果路径，没有后门
    nodes = ["T", "M", "Y"]
    edges = [("T", "M"), ("M", "Y")]
    paths = find_backdoor_paths(nodes, edges, "T", "Y")
    assert paths == []


# ---------------------------------------------------------------------------
# 用例 4：suggest_adjustment_set - 含 mediators 字段
# ---------------------------------------------------------------------------

def test_suggest_adjustment_set_with_mediator_and_confounder():
    nodes = ["吸烟", "焦油", "肺癌", "基因"]
    edges = [("吸烟", "焦油"), ("焦油", "肺癌"), ("基因", "吸烟"), ("基因", "肺癌")]
    result = suggest_adjustment_set(nodes, edges, "吸烟", "肺癌")

    assert result["adjustment_set"] == ["基因"]
    assert result["is_sufficient"] is True
    assert result["blocked_backdoor_paths"] == 1
    assert result["remaining_backdoor_paths"] == []
    assert result["mediators"] == ["焦油"]
    assert "焦油" in result["forbidden_nodes"]


def test_suggest_adjustment_set_pure_confounder_no_mediator():
    nodes = ["教育水平", "收入", "家庭背景"]
    edges = [("教育水平", "收入"), ("家庭背景", "教育水平"), ("家庭背景", "收入")]
    result = suggest_adjustment_set(nodes, edges, "教育水平", "收入")

    assert result["adjustment_set"] == ["家庭背景"]
    assert result["is_sufficient"] is True
    assert result["mediators"] == []
    assert result["forbidden_nodes"] == []


# ---------------------------------------------------------------------------
# 用例 5：to_mermaid - 高亮 + 方向 + 中文节点
# ---------------------------------------------------------------------------

def test_to_mermaid_basic_lr():
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]
    out = to_mermaid(nodes, edges, direction="LR")
    assert out.startswith("graph LR")
    assert 'n0["A"]' in out
    assert "n0 --> n1" in out
    assert "n1 --> n2" in out


def test_to_mermaid_highlight_assigns_classes():
    nodes = ["吸烟", "焦油", "肺癌", "基因"]
    edges = [("吸烟", "焦油"), ("焦油", "肺癌"), ("基因", "吸烟"), ("基因", "肺癌")]
    out = to_mermaid(
        nodes, edges,
        direction="LR",
        highlight={
            "treatment": "吸烟",
            "outcome": "肺癌",
            "adjustment": ["基因"],
            "mediators": ["焦油"],
        },
    )
    assert "classDef cTreat" in out
    assert "classDef cOut" in out
    assert "classDef cAdj" in out
    assert "classDef cMed" in out
    # 高亮颜色断言（与系统提示词补丁里的色板一致）
    assert "#fde68a" in out  # treatment
    assert "#bbf7d0" in out  # outcome
    assert "#bae6fd" in out  # adjustment
    assert "#e9d5ff" in out  # mediator
    # 各节点 id 应分别拿到对应 class
    assert "class n0 cTreat" in out
    assert "class n2 cOut" in out
    assert "class n3 cAdj" in out
    assert "class n1 cMed" in out


def test_to_mermaid_invalid_direction_falls_back_to_lr():
    out = to_mermaid(["A"], [], direction="ZZ")
    assert out.startswith("graph LR")
