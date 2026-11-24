# Case study：工具契约层对轨迹正确率的影响

## 问题

同一个 agent、同一段系统提示、同一批任务，**只换工具层的返回格式**，轨迹正确率会变吗？

这不是"模型能力"实验，而是"工具设计"实验——但它恰恰是 agent 工程里最常被忽略、又最容易改的一块。

## 被测对象

- **agent**：[LangGraph](https://github.com/langchain-ai/langgraph) 官方 `create_react_agent`（prebuilt ReAct 循环）
- **模型**：DeepSeek（`deepseek-chat`，`temperature=0`），也可切脚本化模型只验证管线
- **任务**：48 条（6 类：成功路径 / 参数 / 错误路径 / 不该调用 / 写操作 / 多实体），7 个工具，虚构的客服工单域（`data.py`，无外部依赖、无副作用）

## 两版工具层的唯一差异

| | `tools_baseline.py` | `tools_fixed.py` |
|---|---|---|
| 返回 | JSON 字符串；异常时返回**空串** | 统一 `{"ok": true, ...}` / `{"ok": false, "error": ..., "hint": ...}` |
| 参数 | 不校验，宽松处理 | 严格校验，格式错误直接返回 `invalid_date` |
| 失败语义 | "查不到"和"出错"都是 `""` | `ok=false` 明确表态，并说明"这是确定性错误，重试无用" |

基线那种写法很常见：手工测试时都没问题，问题只在**模型拿到空串之后**——
它分不清"查到但为空"和"失败了"，于是重复调用、或者自己猜一个参数再试。

## 任务设计

10 条里 6 条是成功路径，4 条是**错误路径**（工单不存在、日期非法、月份无数据）。
错误路径的期望是「**只调一次就停**」——确定性错误反复重试或不带依据地换参数，都是错误行为。

```jsonc
{"id": "t06-bad-date", "question": "查一下 2026-13-45 的工单。",
 "expected_tools": ["list_tickets"], "match": "exact"}
```

## 怎么跑

```bash
# 1) 配 key（只在本地，不进仓库）
mkdir -p ~/.config/traceeval
printf 'DEEPSEEK_API_KEY=sk-xxx\n' > ~/.config/traceeval/case.env && chmod 600 ~/.config/traceeval/case.env

# 2) 两版各跑 3 轮
python run.py --variant baseline --runs 3
python run.py --variant fixed    --runs 3

# 3) 出对比表
python compare.py
```

不需要 key 也能**重新打分**：`out/traces_*.jsonl` 里是录下来的真实轨迹，
拿它配 `out/suite.jsonl` 就能离线复现，这正是 traceeval 的设计目的。

只想验证管线是否连通（不产生任何结论）：

```bash
TRACEEVAL_MODEL=fake python run.py --variant baseline --limit 1
```

## 指标

| 指标 | 含义 |
|---|---|
| 通过率 | 只在已判定用例里算（`pass / (pass + fail)`） |
| 未判定率 | 没测成的比例——agent 崩溃、轨迹缺失 |
| 假成功 | 发出了 tool_call 但没有对应的 tool_result |
| 重复调用 | 同一工具在一条轨迹里被调多次 |

## 结果（摘要）

48 条任务 × 3 轮 × 3 个版本：

| 版本 | 轨迹通过率 | 答复违规 | 副作用违规 |
|---|---|---|---|
| baseline（空串返回） | 95.8% | 5 | 3 |
| fixed（结构化返回） | 97.9% | 0 | 0 |
| improved（结构化 + 针对性提示） | **100.0%** | 0 | 0 |

提升路径是「跑基线 → 诊断出 3 条稳定失败 → 按原因码改 → 复测」，
中途还因为提示过度泛化出现过一次**回归**（97.2%），收窄后才到 100%。
完整分析见 [`RESULTS.md`](RESULTS.md)。

## 边界（别把它读成"模型变强了"）

- 只换了工具层，模型、提示、任务都没变；
- 结论只覆盖**这 10 条任务**，不构成通用结论；
- `TRACEEVAL_MODEL=fake` 的产物带 `_scripted` 后缀，只用于验证管线，**不是结果**。
