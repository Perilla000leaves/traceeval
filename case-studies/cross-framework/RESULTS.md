# 结果：同模型、同工具、同任务，只换框架

- **模型**：`deepseek-chat`，`temperature=0`
- **任务**：与 `langgraph-react` 同一份 48 条（6 类）
- **工具**：同一批 Python 函数，各框架各自包装（LangChain `tool()` / ADK `FunctionTool`）
- **提示**：同一段系统提示（improved 版）
- **时间**：2026-11-24

## 总览

| 框架 | 轨迹通过率 | 通过/总数 | 答复违规 | 副作用违规 | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|
| LangGraph `create_react_agent` | **100.0%** | 48/48 | 0 | 0 | 1677 | 2628 |
| LangChain 1.x `create_agent` | **100.0%** | 48/48 | 0 | 0 | 1640 | 3073 |
| Google ADK `LlmAgent` | **100.0%** | 48/48 | 0 | 0 | 1706 | 3400 |

**结论一：在这个任务集和这个模型上，框架不影响轨迹正确率。** 三家的调用序列、参数、答复、副作用全部一致；
差异只在延迟（ADK 的 P95 略高，多一层事件流分发）。

## 但第一次跑出来的数字是 39.6%

第一次跑，ADK 只有 **19/48（39.6%）**，失败码是 `duplicate_call` ×27、`args_mismatch` ×18、`order_mismatch` ×18。
看上去像"ADK 明显更差"。**实际是我的适配层写错了。**

### 定位过程

1. 轨迹里第一条调用是 `lookup_ticket()`，**参数为空**；
2. 工具报 "mandatory input parameters are not present: ticket_id"，模型于是重试一次（所以有 duplicate）；
3. 追问：模型为什么不知道该传参数？打印发给模型的 tools 载荷 —— `parameters: {"type":"object","properties":{}}`，**properties 是空的**；
4. 根因：ADK 2.15 把参数 schema 放在 `FunctionDeclaration.parameters_json_schema`，
   而我的转换函数还在读老字段 `parameters`（这里恒为 `None`）→ 每个工具都变成"无参数工具"。

修掉这一处（并把 schema 改成走 `model_dump()` 取，避免 lazy 字段取不到）之后，ADK 立刻回到 48/48。

### 这条比 100% 更值得写

> **跨框架评测里，适配层是最大的混淆变量。**
> 一个字段名写错就能让某个框架"看起来差 60 个百分点"，而失败现象（重复调用、参数不符）
> 会被误读成"这个框架的 agent 质量不行"。

这和另一处踩坑是同一个模式：LangChain 的 `handle_event` 会读 `handler.ignore_agent`，
鸭子类型缺属性时异常被吞、轨迹永远为空。**两处都是"适配层静默失效"，都只有跑真框架才能发现。**

## 过程中还有两个坑

| 坑 | 现象 | 处理 |
|---|---|---|
| event loop 复用 | 每条任务 `asyncio.run()` → `RuntimeError: Event loop is closed` | 整个 run 复用一个 loop |
| litellm 依赖链 | ADK 官方接 OpenAI 兼容模型要装 litellm，装到一半缺 `h2`/`boto3`/`importlib_metadata` | 放弃，自己写 80 行 `BaseLlm` 适配 |

第二个坑本身是个结论：**ADK 接非 Google 模型需要自己写模型适配层**，官方只内置 Gemini + litellm；
自写的 `DeepSeekLlm` 只做三件事——contents → messages、function_declarations → OpenAI tools、
tool_calls → `LlmResponse`。

## 复现

```bash
# 需要 DEEPSEEK_API_KEY（~/.config/traceeval/case.env）
python run_framework.py --framework lg
python run_framework.py --framework lc
python run_framework.py --framework adk
python compare_frameworks.py
```

## 边界（别把这个 100% 当结论）

- **每条任务只跑 1 轮**（n=1），无法区分稳定与偶发；三家的 100% 是"单次全过"；
- 只有一个模型；任务集当初是围绕 LangGraph 设计的，对 ADK 并不算"原生"的任务；
- 延迟差异受网络抖动影响，**不足以下"某框架更快"的结论**，只能说明量级相同；
- ADK 侧用的是自写 `BaseLlm`，和官方 `LiteLlm` 的行为不完全等价。

## 下一步值得做的

1. **每框架跑 3 轮**（n=3），看是否有偶发差异；
2. 换一个更弱的模型，看框架差异是否在弱模型上放大；
3. 加**多 Agent（sub-agent 转交）**任务——单 Agent ReAct 是三家最像的场景，差异可能藏在编排层。
