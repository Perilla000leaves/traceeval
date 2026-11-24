"""归因诊断：把「哪几条没过」变成「该改什么」。

评测器只回答"过没过"是不够的——要提升通过率，需要三件事：

1. **区分稳定失败与偶发失败**：同一用例跑多轮，全挂是系统性问题，时好时坏是采样噪声
   （或者期望本身不稳），两者的处理方式完全不同；
2. **把失败码映射到改动点**：缺工具 → 看工具描述；参数不符 → 看 schema 与示例；
   过度调用 → 看工具适用边界；
3. **对比改动前后**：哪些修好了、哪些是新引入的回归。

这个模块只做这三件事，不做任何判定——判定仍由 `score()` 负责。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .metrics import (
    REASON_ARGS,
    REASON_DUPLICATE,
    REASON_ERROR,
    REASON_FORBIDDEN,
    REASON_GHOST,
    REASON_MISSING,
    REASON_ORDER,
    REASON_OVER_CALL,
    CaseResult,
)
from .trajectory import Status

#: 失败码 → 该去看哪里。写的是"改动点"，不是"再试一次"。
SUGGESTIONS: dict[str, str] = {
    REASON_GHOST: "工具返回空或异常没有结构化：改成 {ok: false, error}，否则模型分不清「查不到」和「调用失败」",
    REASON_MISSING: "该调的工具没调：多半是工具描述没写清触发条件或前置条件",
    REASON_ORDER: "顺序或数量不符：先确认期望是否过严（先核实再动手是合理行为），否则在提示里写明依赖顺序",
    REASON_ARGS: "参数不满足：检查参数 schema、默认值、以及提示里的格式示例",
    REASON_DUPLICATE: "同一工具被重复调用：提示里加「已拿到结果不要重复调用」，或让结果具备幂等语义",
    REASON_OVER_CALL: "不该调用却调用了：把工具的适用边界写进描述，或在该场景明确要求直接回答",
    REASON_FORBIDDEN: "调用了被禁止的工具（如危险写操作）：在提示里明确该场景必须先确认、不得直接执行",
    REASON_ERROR: "运行报错：先看轨迹是否缺失，再看工具异常是否被静默吞掉",
}

#: 判定类型
STABLE_PASS = "stable_pass"
STABLE_FAIL = "stable_fail"
FLAKY = "flaky"
BLOCKED = "blocked"


@dataclass
class CaseDiagnosis:
    """一条用例跨多轮的诊断结果。"""

    case_id: str
    runs: int
    passed: int
    failed: int
    undetermined: int
    verdict: str
    codes: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    suggestion: str = ""

    @property
    def pass_rate(self) -> float:
        judged = self.passed + self.failed
        return round(100.0 * self.passed / judged, 1) if judged else 0.0


def _verdict(passed: int, failed: int, undetermined: int) -> str:
    if passed + failed == 0:
        return BLOCKED
    if failed == 0:
        return STABLE_PASS
    if passed == 0:
        return STABLE_FAIL
    return FLAKY


def diagnose(results: Sequence[CaseResult]) -> list[CaseDiagnosis]:
    """按用例聚合多轮结果，给出判定类型与改动建议。

    `results` 里同一 `case_id` 出现多次即视为多轮运行。
    """
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.case_id].append(result)

    diagnoses: list[CaseDiagnosis] = []
    for case_id, items in grouped.items():
        passed = sum(1 for item in items if item.status is Status.PASS)
        failed = sum(1 for item in items if item.status is Status.FAIL)
        undetermined = len(items) - passed - failed

        codes: list[str] = []
        examples: list[str] = []
        for item in items:
            for code in item.codes:
                if code not in codes:
                    codes.append(code)
            if item.status is Status.FAIL:
                examples.extend(item.reasons[:1])

        verdict = _verdict(passed, failed, undetermined)
        suggestion = ""
        if verdict in (STABLE_FAIL, FLAKY) and codes:
            top = Counter(code for item in items for code in item.codes).most_common(1)[0][0]
            suggestion = SUGGESTIONS.get(top, "")

        diagnoses.append(
            CaseDiagnosis(
                case_id=case_id,
                runs=len(items),
                passed=passed,
                failed=failed,
                undetermined=undetermined,
                verdict=verdict,
                codes=codes,
                examples=examples[:3],
                suggestion=suggestion,
            )
        )

    order = {STABLE_FAIL: 0, FLAKY: 1, BLOCKED: 2, STABLE_PASS: 3}
    diagnoses.sort(key=lambda item: (order.get(item.verdict, 9), item.case_id))
    return diagnoses


def summary_by_code(results: Sequence[CaseResult]) -> dict[str, int]:
    """全局失败码分布：先修数量最多的那一类，收益最大。"""
    return dict(Counter(code for item in results for code in item.codes).most_common())


def compare(before: Sequence[CaseDiagnosis], after: Sequence[CaseDiagnosis]) -> dict[str, list[str]]:
    """对比两次诊断：修好了什么、回归了什么、还剩什么。"""
    old = {item.case_id: item.verdict for item in before}
    new = {item.case_id: item.verdict for item in after}
    good = {STABLE_PASS}

    fixed = [cid for cid, verdict in old.items() if verdict not in good and new.get(cid) in good]
    regressed = [cid for cid, verdict in new.items() if verdict not in good and old.get(cid) in good]
    remaining = [cid for cid, verdict in new.items() if verdict not in good]
    return {"fixed": sorted(fixed), "regressed": sorted(regressed), "remaining": sorted(remaining)}


def to_markdown(diagnoses: Sequence[CaseDiagnosis], *, title: str = "归因诊断") -> str:
    """诊断报告：稳定失败排前面，每条给原因与改动点。"""
    counts = Counter(item.verdict for item in diagnoses)
    lines = [
        f"# {title}",
        "",
        "| 判定 | 数量 |",
        "|---|---|",
        f"| 稳定失败 | {counts.get(STABLE_FAIL, 0)} |",
        f"| 时好时坏 | {counts.get(FLAKY, 0)} |",
        f"| 无法判定 | {counts.get(BLOCKED, 0)} |",
        f"| 稳定通过 | {counts.get(STABLE_PASS, 0)} |",
        "",
    ]

    problem = [item for item in diagnoses if item.verdict in (STABLE_FAIL, FLAKY)]
    if problem:
        lines += ["## 需要改的用例", "", "| 用例 | 通过/轮次 | 判定 | 原因码 | 建议 |", "|---|---|---|---|---|"]
        for item in problem:
            lines.append(
                f"| {item.case_id} | {item.passed}/{item.runs} | {item.verdict} | "
                f"{', '.join(item.codes) or '-'} | {item.suggestion or '-'} |"
            )
        lines.append("")

    return "\n".join(lines)


def to_json(diagnoses: Sequence[CaseDiagnosis]) -> str:
    """给下游工具消费的 JSON。"""
    return json.dumps([item.__dict__ for item in diagnoses], ensure_ascii=False, indent=2, default=int)


__all__ = [
    "BLOCKED",
    "FLAKY",
    "STABLE_FAIL",
    "STABLE_PASS",
    "SUGGESTIONS",
    "CaseDiagnosis",
    "compare",
    "diagnose",
    "summary_by_code",
    "to_json",
    "to_markdown",
]
