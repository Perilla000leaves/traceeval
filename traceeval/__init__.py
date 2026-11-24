"""traceeval —— 评测 tool-calling agent 的调用轨迹。

跟"只看最终答案"的评测不同，这里评的是**调用序列本身**：调了哪些工具、参数对不对、
有没有拿到返回、有没有重复调、顺序是否符合预期。

三个主张：

1. **假成功要单独判**。模型发出了 tool_call 但没拿到返回，日志里看起来一切正常，
   后面的推理却建立在空结果上。只比工具名的评测会把这类判成通过。
2. **未判定不等于失败**。没写期望工具的用例说明"没测"，单列成 `UNDETERMINED`，
   报告里同时给通过率和未判定率。
3. **轨迹可回放**。用例集与轨迹集分开，别人拿你的 traces 就能复现结果，不需要 API key。
"""

from .adapters import ADAPTERS, parse
from .integrations import TraceCallbackHandler, trajectory_from_otel_spans
from .metrics import CaseResult, score, score_many
from .report import summarize, to_dicts, to_markdown
from .suite import Case, load_cases, load_traces, run
from .testing import TrajectoryAssertionError, assert_tools
from .trajectory import (
    ANY_VALUE,
    Call,
    ExpectedCall,
    Status,
    Trajectory,
    as_expected,
    duplicates,
    match_names,
    normalize_pair,
)

__version__ = "0.1.0"

__all__ = [
    "ADAPTERS",
    "ANY_VALUE",
    "Call",
    "Case",
    "CaseResult",
    "ExpectedCall",
    "Status",
    "TraceCallbackHandler",
    "Trajectory",
    "TrajectoryAssertionError",
    "__version__",
    "as_expected",
    "assert_tools",
    "duplicates",
    "load_cases",
    "load_traces",
    "match_names",
    "normalize_pair",
    "parse",
    "run",
    "score",
    "score_many",
    "summarize",
    "to_dicts",
    "to_markdown",
    "trajectory_from_otel_spans",
]
