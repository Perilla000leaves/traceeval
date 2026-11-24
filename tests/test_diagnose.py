"""归因诊断：稳定失败 / 偶发失败、改动建议、改动前后对比、过度调用检测。"""
from __future__ import annotations

from traceeval.diagnose import (
    BLOCKED,
    FLAKY,
    STABLE_FAIL,
    STABLE_PASS,
    compare,
    diagnose,
    summary_by_code,
    to_markdown,
)
from traceeval.metrics import CaseResult, score
from traceeval.trajectory import Call, Status, Trajectory


def _result(case_id: str, status: Status, codes=()) -> CaseResult:
    return CaseResult(case_id=case_id, status=status, codes=list(codes))


def test_all_pass_is_stable_pass():
    items = [_result("c1", Status.PASS) for _ in range(3)]
    assert diagnose(items)[0].verdict == STABLE_PASS


def test_all_fail_is_stable_fail():
    items = [_result("c1", Status.FAIL, ["missing_tool"]) for _ in range(3)]
    diagnosis = diagnose(items)[0]
    assert diagnosis.verdict == STABLE_FAIL
    assert "工具描述" in diagnosis.suggestion


def test_mixed_is_flaky():
    items = [_result("c1", Status.PASS), _result("c1", Status.FAIL, ["ghost_call"])]
    diagnosis = diagnose(items)[0]
    assert diagnosis.verdict == FLAKY
    assert diagnosis.pass_rate == 50.0


def test_all_undetermined_is_blocked():
    items = [_result("c1", Status.UNDETERMINED) for _ in range(2)]
    assert diagnose(items)[0].verdict == BLOCKED


def test_stable_fail_sorts_before_pass():
    items = [_result("a", Status.PASS), _result("b", Status.FAIL, ["args_mismatch"])]
    assert [item.case_id for item in diagnose(items)] == ["b", "a"]


def test_suggestion_follows_most_common_code():
    items = [
        _result("c1", Status.FAIL, ["duplicate_call"]),
        _result("c1", Status.FAIL, ["duplicate_call"]),
        _result("c1", Status.FAIL, ["args_mismatch"]),
    ]
    assert "重复调用" in diagnose(items)[0].suggestion


def test_summary_by_code_counts_failures():
    items = [
        _result("a", Status.FAIL, ["ghost_call"]),
        _result("b", Status.FAIL, ["ghost_call", "missing_tool"]),
    ]
    assert summary_by_code(items) == {"ghost_call": 2, "missing_tool": 1}


def test_compare_reports_fixed_regressed_remaining():
    before = diagnose([_result("a", Status.FAIL, ["missing_tool"]), _result("b", Status.PASS)])
    after = diagnose([_result("a", Status.PASS), _result("b", Status.FAIL, ["args_mismatch"])])
    diff = compare(before, after)

    assert diff["fixed"] == ["a"]
    assert diff["regressed"] == ["b"]
    assert diff["remaining"] == ["b"]


def test_markdown_lists_only_problem_cases():
    items = [_result("good", Status.PASS), _result("bad", Status.FAIL, ["missing_tool"])]
    text = to_markdown(diagnose(items))

    assert "bad" in text
    assert "| good |" not in text


# ---------------------------------------------------------------- 过度调用


def test_expect_no_calls_passes_when_model_does_not_call_tools():
    result = score("c1", [], Trajectory(calls=[]), expect_no_calls=True)
    assert result.status is Status.PASS


def test_expect_no_calls_fails_when_model_calls_any_tool():
    """过度调用：用户只是打个招呼，模型却去查工单。"""
    result = score("c1", [], Trajectory(calls=[Call("lookup_ticket", result_ok=True)]), expect_no_calls=True)
    assert result.status is Status.FAIL
    assert "over_call" in result.codes
    assert "不该调用工具" in result.reasons[0]


def test_without_expect_no_calls_empty_expectation_is_still_undetermined():
    result = score("c1", [], Trajectory(calls=[]))
    assert result.status is Status.UNDETERMINED


def test_forbid_tools_passes_when_only_other_tools_called():
    """允许核实（lookup），但禁止发通知：这种精细断言比"一个都不许调"实用。"""
    result = score(
        "c1",
        [],
        Trajectory(calls=[Call("lookup_ticket", result_ok=True)]),
        forbid_tools=["notify_customer"],
    )
    assert result.status is Status.PASS


def test_forbid_tools_fails_when_forbidden_tool_called():
    result = score(
        "c1",
        [],
        Trajectory(calls=[Call("notify_customer", result_ok=True)]),
        forbid_tools=["notify_customer"],
    )
    assert result.status is Status.FAIL
    assert "forbidden_tool" in result.codes
