# Case study：跨框架对比（LangGraph / LangChain / ADK）

同一批任务、同一模型、同一套工具函数、同一段系统提示，只换 agent 框架。

```
python run_framework.py --framework lg    # LangGraph create_react_agent
python run_framework.py --framework lc    # LangChain 1.x create_agent
python run_framework.py --framework adk   # Google ADK LlmAgent
python compare_frameworks.py
```

需要 `DEEPSEEK_API_KEY`（写在 `~/.config/traceeval/case.env`）。

## 文件

| 文件 | 作用 |
|---|---|
| `tools_common.py` | 7 个工具函数（普通函数，各框架各自包装）+ 系统提示 + 任务加载 |
| `adk_model.py` | ADK 的 `BaseLlm` 适配（DeepSeek 走 OpenAI 兼容接口，约 80 行，无额外依赖） |
| `frameworks.py` | 三个框架的构造与单次运行（LC/LG 走 LangChain 回调，ADK 走 tool 回调） |
| `checks.py` | 答复与副作用核验（与 langgraph-react 同口径） |
| `run_framework.py` | 跑单个框架的全量任务并落盘 |
| `compare_frameworks.py` | 出对比表 |

## 结果

见 [`RESULTS.md`](RESULTS.md)。一句话：**三家都是 100%**，但第一次跑 ADK 只有 39.6%——
原因是我的适配层漏传工具 schema，**不是框架差异**。
