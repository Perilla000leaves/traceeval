"""打分：假成功优先、未判定单列、失败原因可归因。"""
from __future__ import annotations

from traceeval.metrics import (
    REASON_ARGS,
    REASON_DUPLICATE,
    REASON_GHOST,
    REASON_MISSING,
    score,
)
from traceeval.trajectory import Call, Status, Trajectory


def _trajectory(*calls: Call) -> Trajectory:
    return Trajectory(calls=list(calls))


def test_all_tools_hit_is_pass():
    result = score("c1", ["a", "b"], _trajectory(Call("a", result_ok=True), Call("b", result_ok=True)))
    assert result.status is Status.PASS
    assert result.reasons == []


def test_missing_tool_fails_with_reason():
    result = score("c1", ["a", "b"], _trajectory(Call("a", result_ok=True)))
    assert result.status is Status.FAIL
    assert REASON_MISSING in result.codes
    assert "b" in " ".join(result.reasons)


def test_ghost_call_is_failure_even_if_tool_names_match():
    """假成功：工具名对上了，但那次调用没有返回 —— 只看名字会误判成通过。"""
    result = score("c1", ["a"], _trajectory(Call("a", result_ok=None)))
    assert result.status is Status.FAIL
    assert REASON_GHOST in result.codes
    assert "假成功" in result.reasons[0]


def test_require_results_false_ignores_ghosts():
    """只有请求侧日志时，允许关掉假成功检测。"""
    result = score("c1", ["a"], _trajectory(Call("a", result_ok=None)), require_results=False)
    assert result.status is Status.PASS


def test_missing_arg_key_fails():
    result = score(
        "c1",
        [{"name": "list", "args": {"date": "2026-11-18", "limit": 10}}],
        _trajectory(Call("list", args={"date": "2026-11-18"}, result_ok=True)),
    )
    assert result.status is Status.FAIL
    assert REASON_ARGS in result.codes
    assert "limit" in " ".join(result.reasons)


def test_full_args_mode_checks_values():
    result = score(
        "c1",
        [{"name": "list", "args": {"date": "2026-11-18"}}],
        _trajectory(Call("list", args={"date": "2026-01-01"}, result_ok=True)),
        args_mode="full",
    )
    assert result.status is Status.FAIL
    assert REASON_ARGS in result.codes


def test_duplicate_calls_fail_by_default():
    result = score("c1", ["a"], _trajectory(Call("a", result_ok=True), Call("a", result_ok=True)))
    assert result.status is Status.FAIL
    assert REASON_DUPLICATE in result.codes


def test_duplicate_calls_can_be_tolerated():
    result = score(
        "c1",
        ["a"],
        _trajectory(Call("a", result_ok=True), Call("a", result_ok=True)),
        fail_on_duplicates=False,
    )
    assert result.status is Status.PASS


def test_no_expected_tools_is_undetermined_not_failure():
    """没写期望 = 没测。算成失败会拉低通过率，算成通过会虚高。"""
    result = score("c1", [], _trajectory(Call("a", result_ok=True)))
    assert result.status is Status.UNDETERMINED
    assert "未提供期望工具" in result.reasons[0]


def test_run_error_fails_fast():
    result = score("c1", ["a"], _trajectory(), error="timeout")
    assert result.status is Status.FAIL
    assert "运行报错" in result.reasons[0]


def test_metrics_carry_details_for_report():
    result = score(
        "c1",
        ["a", "b"],
        _trajectory(Call("a", result_ok=None), Call("b", result_ok=True), Call("b", result_ok=True)),
    )
    assert result.metrics["ghosts"] == ["a"]
    assert result.metrics["duplicates"] == ["b"]
    assert "b" in result.metrics["missing"] or result.metrics["matched"] is True
