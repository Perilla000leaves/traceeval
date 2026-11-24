"""对比 baseline / fixed 两次运行，生成 markdown 对比表。

读 `out/summary_*.json` 与 `out/details_*.json`，不需要 key、不需要重跑。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load(tag: str) -> tuple[dict, list[dict]]:
    summary = json.loads((OUT / f"summary_{tag}.json").read_text(encoding="utf-8"))
    details = json.loads((OUT / f"details_{tag}.json").read_text(encoding="utf-8"))
    return summary, details


def table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    tags = [tag.name.removeprefix("summary_").removesuffix(".json") for tag in sorted(OUT.glob("summary_*.json"))]
    if len(tags) < 2:
        raise SystemExit(f"至少需要两份 summary，当前：{tags}")

    summaries = {tag: load(tag) for tag in tags}
    overview_rows = []
    for tag, (summary, _details) in summaries.items():
        overview_rows.append(
            [
                tag,
                f"{summary['pass_rate']}%" if summary["pass_rate"] is not None else "n/a",
                f"{summary['undetermined_rate']}%",
                str(summary["passed"]),
                str(summary["failed"]),
                str(summary["undetermined"]),
                str(summary["ghost_calls"]),
            ]
        )

    print("## 总览\n")
    print(table(overview_rows, ["版本", "通过率", "未判定率", "通过", "失败", "未判定", "假成功"]))
    print()

    # 逐用例对比
    by_id: dict[str, dict[str, str]] = {}
    for tag, (_summary, details) in summaries.items():
        for row in details:
            by_id.setdefault(row["id"], {})[tag] = row["status"]

    case_rows = []
    for case_id, per_tag in sorted(by_id.items()):
        case_rows.append([case_id] + [per_tag.get(tag, "-") for tag in tags])
    print("## 逐用例\n")
    print(table(case_rows, ["用例"] + tags))


if __name__ == "__main__":
    main()
