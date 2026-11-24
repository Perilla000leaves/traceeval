"""对比三个框架的结果：`python compare_frameworks.py`。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
NAMES = {"lg": "LangGraph", "lc": "LangChain", "adk": "Google ADK"}


def main() -> None:
    summaries = {}
    for key in NAMES:
        path = OUT / f"summary_{key}.json"
        if path.is_file():
            summaries[key] = json.loads(path.read_text(encoding="utf-8"))
    if len(summaries) < 2:
        raise SystemExit(f"至少需要两个框架的结果，当前：{list(summaries)}")

    rows = []
    for key, data in summaries.items():
        rows.append(
            [
                NAMES[key],
                f"{data['pass_rate']}%",
                f"{data['passed']}/{data['total']}",
                str(data.get("answer_violations", 0)),
                str(data.get("side_effect_violations", 0)),
                f"{data.get('latency_p50_ms') or '-'}",
                f"{data.get('latency_p95_ms') or '-'}",
            ]
        )
    headers = ["框架", "轨迹通过率", "通过/总数", "答复违规", "副作用违规", "P50(ms)", "P95(ms)"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()

    details = {}
    for key in summaries:
        details[key] = {r["id"]: r["status"] for r in json.loads((OUT / f"details_{key}.json").read_text(encoding="utf-8"))}
    ids = sorted(next(iter(details.values())))
    print("| 用例 | " + " | ".join(NAMES[k] for k in details) + " |")
    print("|" + "---|" * (len(details) + 1))
    for case_id in ids:
        values = [details[key].get(case_id, "-") for key in details]
        if len(set(values)) > 1:
            print(f"| {case_id} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
