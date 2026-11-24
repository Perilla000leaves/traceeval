"""轨迹模型与匹配：把「一次 agent 运行」变成可比较的调用序列。

评测 tool-calling agent 时，最容易被忽略的不是模型调错工具，而是**判定器自己出错**：

- 期望里只写工具名、实际调用带了 `date="2026-11-10"` 这类会漂的参数，严格相等直接判 0；
- 模型"声称"调用了工具，但轨迹里根本没有对应的返回（假成功），只看工具名会判过；
- 没给期望的用例被算成通过，通过率凭空变高。

这里只定义数据结构和比较规则，不掺任何模型 SDK。
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: 归一化时把参数值替换成这个占位符，比较退化成「key 集合比较」
ANY_VALUE = "<any>"


class Status(StrEnum):
    """一条用例的判定结果。

    `UNDETERMINED` 必须与 `FAIL` 分开：它是「没测」，不是「没通过」。
    把两者混在一起，通过率就会变成一个无法解释的数字。
    """

    PASS = "pass"
    FAIL = "fail"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Call:
    """一次工具调用。

    `result_ok` 为 `None` 表示**没有拿到返回**——这是假成功的判定依据：
    模型发出了 tool_call，但消息流里没有对应的 tool result。
    """

    name: str
    args: dict[str, Any] | None = None
    result_ok: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ExpectedCall:
    """期望的一次调用。`args` 为 None 表示不关心参数。"""

    name: str
    args: dict[str, Any] | None = None


@dataclass
class Trajectory:
    """一次运行里的完整调用序列，外加上下文里能拿到的元信息。"""

    calls: list[Call] = field(default_factory=list)
    latency_ms: float | None = None
    cost_usd: float | None = None

    def names(self) -> list[str]:
        return [call.name for call in self.calls]

    def ghosts(self) -> list[Call]:
        """发出去了但没有返回的调用（假成功的直接证据）。"""
        return [call for call in self.calls if call.result_ok is None]


def _parse_args(raw: Any) -> dict[str, Any] | None:
    """事件里的参数常是 JSON 字符串；解析不出 dict 就当「没写参数」。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def as_expected(items: Iterable[Any]) -> list[ExpectedCall]:
    """把测试集里写的工具名或 dict 统一成 ExpectedCall。"""
    expected: list[ExpectedCall] = []
    for item in items:
        if isinstance(item, ExpectedCall):
            expected.append(item)
        elif isinstance(item, str):
            expected.append(ExpectedCall(name=item))
        elif isinstance(item, dict):
            expected.append(
                ExpectedCall(name=str(item.get("name") or ""), args=_parse_args(item.get("args")))
            )
    return [item for item in expected if item.name]


def normalize_pair(
    actual: Sequence[Call],
    expected: Sequence[ExpectedCall],
    args_mode: str = "keys",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """成对归一化实际与期望的调用，让 `==` 恢复意义。

    规则（两侧同一套）：

    - 期望侧没写参数 → 不关心，实际侧也置空；
    - `args_mode="full"` → 原样比较（参数必须完全一致）；
    - 默认 `keys` 模式 → 只比较期望提到的 key，值统一替换成占位符。

    期望没提到的实际调用不参与参数比较（匹配阶段只看 name 是否出现）。
    """
    act = [{"name": call.name, "args": call.args} for call in actual]
    exp = [{"name": want.name, "args": want.args} for want in expected]
    paired = [False] * len(act)

    for want in exp:
        for index, have in enumerate(act):
            if paired[index] or have["name"] != want["name"]:
                continue
            paired[index] = True
            if want["args"] is None:
                have["args"] = None
            elif args_mode == "full":
                pass
            else:
                wanted = set(want["args"])
                given = set(have["args"] or {})
                want["args"] = {key: ANY_VALUE for key in sorted(wanted)}
                have["args"] = {key: ANY_VALUE for key in sorted(wanted & given)}
            break

    for index, have in enumerate(act):
        if not paired[index]:
            have["args"] = None
    return act, exp


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    iterator = iter(haystack)
    return all(item in iterator for item in needle)


def match_names(actual: Sequence[str], expected: Sequence[str], mode: str = "any_order") -> bool:
    """按三种语义判断工具名是否匹配。

    - `exact`：数量与顺序完全一致；
    - `in_order`：期望是实际调用链路的**子序列**（允许多调别的工具）；
    - `any_order`：期望的工具都出现过即可。
    """
    mode = str(mode or "any_order").strip().lower().replace("-", "_")
    actual_names = [name for name in actual if name]
    expected_names = [name for name in expected if name]
    if not expected_names:
        return False
    if mode == "exact":
        return actual_names == expected_names
    if mode == "in_order":
        return _is_subsequence(expected_names, actual_names)
    return not (Counter(expected_names) - Counter(actual_names))


def duplicates(names: Iterable[str]) -> list[str]:
    """被重复调用的工具名（保序）。长任务里同一个工具反复调是典型翻车形态。"""
    seen: set[str] = set()
    repeated: list[str] = []
    for name in names:
        if not name:
            continue
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    return repeated


__all__ = [
    "ANY_VALUE",
    "Call",
    "ExpectedCall",
    "Status",
    "Trajectory",
    "as_expected",
    "duplicates",
    "match_names",
    "normalize_pair",
]
