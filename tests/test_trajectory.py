"""轨迹模型与匹配语义：归一化、三种匹配、重复与假成功识别。"""
from __future__ import annotations

import pytest

from traceeval.trajectory import (
    ANY_VALUE,
    Call,
    ExpectedCall,
    Trajectory,
    as_expected,
    duplicates,
    match_names,
    normalize_pair,
)


def test_normalize_keys_mode_ignores_value_drift():
    actual = [Call(name="list_tickets", args={"date": "2026-11-18", "limit": 10})]
    expected = [ExpectedCall(name="list_tickets", args={"date": "2026-01-01"})]

    act, exp = normalize_pair(actual, expected, args_mode="keys")

    assert act[0]["args"] == {"date": ANY_VALUE}
    assert exp[0]["args"] == {"date": ANY_VALUE}
    assert act[0]["args"] == exp[0]["args"]


def test_normalize_full_mode_compares_values():
    actual = [Call(name="t", args={"date": "2026-11-18"})]
    expected = [ExpectedCall(name="t", args={"date": "2026-01-01"})]

    act, exp = normalize_pair(actual, expected, args_mode="full")

    assert act[0]["args"] != exp[0]["args"]


def test_normalize_without_expected_args_skips_comparison():
    act, exp = normalize_pair([Call(name="t", args={"x": 1})], [ExpectedCall(name="t")])
    assert act[0]["args"] is None and exp[0]["args"] is None


def test_match_exact_requires_order_and_length():
    assert match_names(["a", "b"], ["a", "b"], "exact")
    assert not match_names(["b", "a"], ["a", "b"], "exact")
    assert not match_names(["a", "b", "c"], ["a", "b"], "exact")


def test_match_in_order_allows_extra_calls():
    assert match_names(["x", "a", "y", "b"], ["a", "b"], "in_order")
    assert not match_names(["b", "a"], ["a", "b"], "in_order")


def test_match_any_order_ignores_order():
    assert match_names(["b", "x", "a"], ["a", "b"], "any_order")
    assert not match_names(["a"], ["a", "b"], "any_order")


def test_unknown_match_mode_falls_back_to_any_order():
    assert match_names(["b", "a"], ["a", "b"], "nonsense")


def test_duplicates_keeps_first_seen_order():
    assert duplicates(["a", "b", "a", "b", "a"]) == ["a", "b"]
    assert duplicates(["a", "b"]) == []


def test_as_expected_accepts_strings_and_dicts():
    expected = as_expected(["a", {"name": "b", "args": '{"k": 1}'}, ""])
    assert [item.name for item in expected] == ["a", "b"]
    assert expected[1].args == {"k": 1}


def test_ghosts_are_calls_without_result():
    trajectory = Trajectory(
        calls=[
            Call(name="ok_tool", result_ok=True),
            Call(name="never_returned", result_ok=None),
        ]
    )
    assert [call.name for call in trajectory.ghosts()] == ["never_returned"]


@pytest.mark.parametrize("mode", ["exact", "in_order", "any_order"])
def test_empty_expected_never_matches(mode):
    assert match_names(["a"], [], mode) is False
