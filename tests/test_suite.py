"""用例集、回放与报告：缺轨迹要记未判定，通过率只在已判定范围内算。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceeval import report, suite
from traceeval.trajectory import Status


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_cases_parses_fields(tmp_path: Path):
    path = _write(
        tmp_path / "suite.jsonl",
        [{"id": "c1", "question": "q", "expected_tools": ["a"], "match": "exact", "tags": ["x"]}],
    )
    cases = suite.load_cases(path)

    assert cases[0].case_id == "c1"
    assert cases[0].match == "exact"
    assert cases[0].tags == ["x"]


def test_load_cases_requires_id(tmp_path: Path):
    path = _write(tmp_path / "suite.jsonl", [{"expected_tools": ["a"]}])
    with pytest.raises(ValueError, match="缺少 id"):
        suite.load_cases(path)


def test_load_traces_supports_mixed_formats(tmp_path: Path):
    path = _write(
        tmp_path / "traces.jsonl",
        [
            {"id": "c1", "format": "calls", "payload": [{"name": "a", "result_ok": True}]},
            {
                "id": "c2",
                "format": "events",
                "payload": [
                    {"type": "tool_call", "id": "e1", "name": "b"},
                    {"type": "tool_result", "id": "e1", "ok": True},
                ],
            },
        ],
    )
    traces = suite.load_traces(path)

    assert traces["c1"].names() == ["a"]
    assert traces["c2"].names() == ["b"]


def test_missing_trace_is_undetermined_not_failure(tmp_path: Path):
    """轨迹没录上说明没测，不能算成模型失败。"""
    cases = [suite.Case(case_id="c1", expected_tools=["a"])]
    results = suite.run(cases, {})

    assert results[0].status is Status.UNDETERMINED
    assert "缺该用例的轨迹记录" in results[0].reasons[0]


def test_per_case_match_overrides_default(tmp_path: Path):
    from traceeval.trajectory import Call, Trajectory

    cases = [suite.Case(case_id="c1", expected_tools=["a", "b"], match="exact")]
    traces = {"c1": Trajectory(calls=[Call("b", result_ok=True), Call("a", result_ok=True)])}
    results = suite.run(cases, traces, match="any_order")

    assert results[0].status is Status.FAIL  # exact 要求顺序


def test_summary_excludes_undetermined_from_pass_rate():
    from traceeval.trajectory import Call, Trajectory

    cases = [
        suite.Case(case_id="c1", expected_tools=["a"]),
        suite.Case(case_id="c2", expected_tools=["b"]),
        suite.Case(case_id="c3", expected_tools=["c"]),
    ]
    traces = {
        "c1": Trajectory(calls=[Call("a", result_ok=True)]),
        "c2": Trajectory(calls=[Call("zzz", result_ok=True)]),
    }
    summary = report.summarize(suite.run(cases, traces))

    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["undetermined"] == 1
    assert summary["pass_rate"] == 50.0
    assert summary["undetermined_rate"] == pytest.approx(33.3, abs=0.1)


def test_markdown_report_contains_summary_and_reasons():
    from traceeval.trajectory import Call, Trajectory

    cases = [suite.Case(case_id="c1", expected_tools=["a"])]
    traces = {"c1": Trajectory(calls=[Call("a", result_ok=None)])}
    text = report.to_markdown(suite.run(cases, traces))

    assert "轨迹评测报告" in text
    assert "假成功" in text
    assert "ghost_call" in text
