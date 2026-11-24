# traceeval

评测 **tool-calling agent 的调用轨迹**，而不只是最终答案。

现有的 LLM 评测工具大多在比「回复像不像标准答案」。但对 agent 来说，真正决定成败的是
**它调了哪些工具、参数对不对、有没有拿到返回**。这个项目只做这件事：把一次运行变成可比较、
可归因、可复现的调用序列。

```bash
pip install -e .
python -m traceeval run --suite suites/support_agent.jsonl --traces suites/support_agent.baseline.jsonl
```

**运行时零依赖**：`pip install traceeval` 不会给你装 LangChain / ADK / OpenAI SDK。
接入层用鸭子类型实现，框架是你自己的，评测器不掺和。

内置一份 7 条的示例集与录好的轨迹，**不需要任何 API key** 就能跑出完整报告。

## 三个主张

### 1. 假成功必须单独判

模型发出了 `tool_call`，但消息流里没有对应的 `tool_result`。日志看得起来一切正常，
后续推理却建立在一个空结果上——这是 agent 最隐蔽的失败形态，而**只比工具名的评测会把它判成通过**。

```
假成功：1 个调用没有返回（rebuild_category）
```

判定依据是「发出与返回按 id 配对」，不是工具名出现了就算过。

### 2. 未判定不等于失败

没写期望工具的用例说明**没测**。把它算成失败会拉低通过率，算成通过会虚高，
两种做法都让数字失去解释力。所以结果是三态：

```
| 用例 | 通过 | 失败 | 未判定 | 通过率（已判定） | 未判定率 | 假成功调用 |
| 7    | 1    | 4    | 2      | 20.0%            | 28.6%    | 1          |
```

`通过率` 只在已判定的用例里算，`未判定率` 单列——它本身就是评测质量的指标。
轨迹没录上、判定器缺依赖、用例没写期望，都属于未判定。

### 3. 轨迹可回放，评测可复现

用例集（`suites/*.jsonl`）和轨迹集（`*.baseline.jsonl`）是分开的：前者写「期望调什么」，
后者是某次运行录下来的真实序列。**拿别人的轨迹集重新打分不需要模型、不需要联网**，
这是能做公开基线比较的前提。

## 接入你的框架

评测器有没有用，取决于能不能接进你现在这套代码。**判定内核零依赖，接入层每个框架一个薄适配器**：

| 你的栈 | 怎么接 | 用哪个 |
|---|---|---|
| LangChain | 回调处理器（鸭子类型，不强制安装） | `TraceCallbackHandler` |
| LangGraph | **同一个回调处理器**（LangGraph 复用 LangChain 回调协议） | `TraceCallbackHandler` |
| Google ADK | `before_tool_callback` / `after_tool_callback` / `on_tool_error_callback` | `AdkTraceRecorder` |
| OpenAI SDK | `assistant.tool_calls` + `role="tool"` | `format: "openai"` |
| Anthropic | `tool_use` / `tool_result` 内容块 | `format: "anthropic"` |
| OpenTelemetry（Phoenix / Langfuse / LangSmith 都能导出） | GenAI 语义约定 | `trajectory_from_otel_spans` |
| MCP / 自建 runtime | `tool_call` / `tool_result` 事件流 | `format: "events"` |
| pytest 单测 | 断言助手 | `assert_tools(...)` |

### LangChain / LangGraph

```python
from traceeval import TraceCallbackHandler, score

handler = TraceCallbackHandler()
agent.invoke({"question": "查一下这个工单"}, config={"callbacks": [handler]})

result = score("case-1", ["lookup_ticket", "lookup_customer"], handler.trajectory(), match="in_order")
print(result.status, result.reasons)
```

**为什么不继承 `BaseCallbackHandler`**：LangChain 只按方法名回调，继承只会带来看不见的版本耦合。
鸭子类型的好处是——装上 traceeval 不改变你的依赖树，LangChain 大版本升级也不会让它挂。

LangGraph 复用同一套回调协议（会额外传 `parent_run_id` / `tags` / `metadata`），
所以 `TraceCallbackHandler` 一行都不用改。

> **一个真实的坑**：LangChain 的 `handle_event` 会直接读 `handler.ignore_agent` /
> `handler.raise_error`（不是 `getattr(..., False)`）。鸭子类型缺这两个属性会抛
> `AttributeError`，而外层 `try/except` 只打一条 warning 就跳过——表现是**装上评测器后轨迹永远为空**。
> 这两个属性是协议的一部分，`TraceCallbackHandler` 里显式声明了。这个 bug 就是被兼容测试抓出来的。

### Google ADK

```python
from traceeval import AdkTraceRecorder, score

recorder = AdkTraceRecorder()
agent = LlmAgent(
    ...,
    before_tool_callback=recorder.before_tool_callback,
    after_tool_callback=recorder.after_tool_callback,
    on_tool_error_callback=recorder.on_tool_error_callback,
)
await runner.run_async(...)

print(score("case-1", ["lookup_ticket"], recorder.trajectory()).status)
```

两个细节：回调**必须返回 `None`**（ADK 里返回 dict 会短路并当工具结果用）；
ADK 的回调签名没有每次调用的唯一 id，录制器按工具名 FIFO 配对——单个 agent 内工具顺序执行时成立，
并发场景请改用 `trajectory_from_adk_events` 读事件流（事件里带 id）。

### 三个框架一次对照

`examples/three_frameworks.py` 用同一份期望评三条来源不同的轨迹：

```
框架                    结果            原因
LangChain             pass          -
LangGraph             pass          -
ADK（第二个工具中断）    fail          假成功：1 个调用没有返回（lookup_customer）
```

前两行说明「换框架不影响判定」，第三行说明「中断的工具调用会被抓到」——这正是本项目的意义：
**评测口径不随框架变，但框架层的问题跑不掉。**

### OpenTelemetry

```python
from traceeval import trajectory_from_otel_spans

trajectory = trajectory_from_otel_spans(spans)   # 读 gen_ai.tool.name / call.id / call.arguments
```

接 OTel 等于接所有能导出 OTel 的追踪后端，不用为每个厂商写一遍。

### 在单测里断言

```python
from traceeval.testing import assert_tools

def test_agent_calls_lookup_before_customer():
    result = run_agent("查一下这个工单的客户")
    assert_tools(result.trajectory(), ["lookup_ticket", "lookup_customer"], match="in_order")
```

断言失败时直接报出「期望 / 实际 / 原因」，**假成功也会被抓到**——这是手写 `assert tools == [...]` 做不到的。

## 能测什么

| 维度 | 怎么判 |
|---|---|
| 工具命中 | 三种匹配语义：`exact`（顺序与数量一致）/ `in_order`（子序列）/ `any_order`（无序） |
| 参数正确性 | `keys` 模式只比期望提到的参数名（容忍日期、ID 这类值漂移）；`full` 模式连值一起比 |
| 假成功 | 发出调用但没有对应返回 |
| 重复调用 | 同一工具在一条轨迹里被调多次（长任务典型翻车） |
| 未判定 | 没写期望、缺轨迹、判定器不可用 |

## 输入格式

三种主流形态都能直接读，也可以用自己的事件流：

```jsonc
// openai：assistant.tool_calls + role="tool"
{"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "lookup_ticket", "arguments": "{}"}}]}
{"role": "tool", "tool_call_id": "t1", "content": "..."}

// anthropic：tool_use / tool_result 内容块
{"type": "tool_use", "id": "u1", "name": "lookup_ticket", "input": {}}
{"type": "tool_result", "tool_use_id": "u1", "content": "..."}

// events：MCP、自建 runtime、OTel span 都可以映射成这个形态
{"type": "tool_call", "id": "e1", "name": "lookup_ticket", "args": {}}
{"type": "tool_result", "id": "e1", "ok": true}
```

## 当门禁用

默认只输出报告、不因失败返回非零（它是分析工具，不是 CI 门禁）。需要门禁时：

```bash
python -m traceeval run --suite s.jsonl --traces t.jsonl --strict        # 有失败即退出 1
python -m traceeval run --suite s.jsonl --traces t.jsonl --fail-under 90 # 通过率低于 90% 退出 1
```

## 它不做什么

诚实地划边界：

- **不跑模型**。输入是录好的轨迹，评测器不负责生成。想测自己的模型，按上面的格式录一份 traces 即可。
- **不判语义正确性**。参数「对不对」只按你声明的期望比，不判断回答内容是否合理——那是另一类评测。
- **不绑框架**但有接入层。核心判定不 import 任何框架（装上 traceeval 不改变你的依赖树），
  LangChain / OpenAI / Anthropic / OTel / MCP 各有薄适配器，见上文「接入你的框架」。
- **不含延迟与成本归因**。报告会带上轨迹里记录的 `latency_ms` / `cost_usd` 的 P50/P95 与总和，
  但不会去推算「为什么慢」。

## 为什么不用现成方案

- 平台型（LangSmith 等）绑定服务，离线跑不了，也不适合做公开基线；
- 只看最终答案的评测（多数 benchmark）测不出工具序列的问题；
- 框架自带的评测器通常只在自己的运行时里可用，且**不区分未判定**。

如果你只是想知道「回复对不对」，这个项目帮不上忙；如果你在排查「agent 为什么行为不对」，
它要的就是这份轨迹。

## 依赖

**核心零依赖**（`dependencies = []`），**接入模块真依赖框架**——这是有意的分层：

| 模块 | 依赖 | 为什么 |
|---|---|---|
| `trajectory` / `metrics` / `report` / `suite` / `cli` | 无 | 判定逻辑与框架无关；测 OpenAI SDK 的人不该被迫装 LangChain |
| `integrations.langchain` | `langchain-core`（`pip install traceeval[langchain]`） | **真继承 `BaseCallbackHandler`**，不是照着文档猜协议 |
| `integrations.adk` | `google-adk`（`pip install traceeval[adk]`） | 用真实 `BaseTool` 做类型判断 |
| 测试 | `pytest`（`[test]`） | — |

装了就真继承、没装就退化成同名接口（`LANGCHAIN_AVAILABLE` / `ADK_AVAILABLE` 会告诉你当前状态），
所以核心测试在没有框架的环境里照样能跑——CI 的 `test` job 就是这种环境，
`compat` job 则装上真实框架跑同一批接入测试。

```python
from traceeval.integrations import LANGCHAIN_AVAILABLE
# True = 当前用的是真实 BaseCallbackHandler 子类
```

## 开发

```bash
python -m pytest -q          # 64 条用例
ruff check .
```

Roadmap 见 [docs/roadmap.md](docs/roadmap.md)。
