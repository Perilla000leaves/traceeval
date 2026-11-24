# traceeval

评测 **tool-calling agent 的调用轨迹**，而不只是最终答案。

现有的 LLM 评测工具大多在比「回复像不像标准答案」。但对 agent 来说，真正决定成败的是
**它调了哪些工具、参数对不对、有没有拿到返回**。这个项目只做这件事：把一次运行变成可比较、
可归因、可复现的调用序列。

```bash
pip install -e ".[test]"
python -m traceeval run --suite suites/support_agent.jsonl --traces suites/support_agent.baseline.jsonl
```

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
- **不绑框架**。没有 LangChain / ADK / OpenAI 依赖；适配只发生在读入轨迹那一步。
- **不含延迟与成本归因**。报告会带上轨迹里记录的 `latency_ms` / `cost_usd` 的 P50/P95 与总和，
  但不会去推算「为什么慢」。

## 为什么不用现成方案

- 平台型（LangSmith 等）绑定服务，离线跑不了，也不适合做公开基线；
- 只看最终答案的评测（多数 benchmark）测不出工具序列的问题；
- 框架自带的评测器通常只在自己的运行时里可用，且**不区分未判定**。

如果你只是想知道「回复对不对」，这个项目帮不上忙；如果你在排查「agent 为什么行为不对」，
它要的就是这份轨迹。

## 开发

```bash
python -m pytest -q          # 40+ 条用例
ruff check .
```

Roadmap 见 [docs/roadmap.md](docs/roadmap.md)。
