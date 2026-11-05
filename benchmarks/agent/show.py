"""Render a ``run.py`` session as Markdown: prompts, tool calls, results, replies.

    python benchmarks/agent/show.py /tmp/agent-runs/vague-1 > vague-1.md
    python benchmarks/agent/show.py /tmp/agent-runs/vague-1 --brief   # one line a call
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))


def _call_name(name: str) -> str:
    return name.replace("mcp__abbot__", "abbot.")


def render(out: Path, limit: int, brief: bool) -> str:
    record = json.loads((out / "run.json").read_text())
    lines = [f"# {out.name}", "",
             f"Scenario `{record['scenario']}`, abbot `{record['abbot']}`"
             + (f", model `{record['model']}`" if record.get("model") else "")
             + (", with the abbot skill" if record.get("skill") else ""), ""]
    for n, turn in enumerate(record["turns"], start=1):
        label = "Prompt" if n == 1 else "Reply"
        lines += [f"## Turn {n}", "", f"**{label}:** {turn['prompt']}", ""]
        for raw in (out / f"turn-{n}.jsonl").read_text().splitlines():
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "assistant":
                for c in msg["message"]["content"]:
                    if c["type"] == "text":
                        lines += [c["text"], ""]
                    elif c["type"] == "tool_use":
                        args = json.dumps(c["input"])
                        if brief:
                            lines += [f"- `{_call_name(c['name'])}` {args[:200]}"]
                        else:
                            lines += [f"**Call `{_call_name(c['name'])}`**", "```json",
                                      json.dumps(c["input"], indent=1), "```", ""]
            elif kind == "user":
                for c in msg["message"]["content"]:
                    if not (isinstance(c, dict) and c.get("type") == "tool_result"):
                        continue
                    text = _result_text(c["content"])
                    error = "error " if c.get("is_error") else ""
                    if brief:
                        lines += [f"  - → {error}{text[:200]!r}"]
                    else:
                        if len(text) > limit:
                            text = text[:limit] + " …"
                        lines += ["*Result (error):*" if error else "*Result:*", "```",
                                  text, "```", ""]
            elif kind == "result":
                footer = (f"*Turn {n}: {msg.get('subtype')}, "
                          f"{msg.get('num_turns')} turns, "
                          f"${msg.get('total_cost_usd', 0):.2f}, "
                          f"{msg.get('duration_ms', 0) / 1000:.0f} s*")
                lines += ["", footer, ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=1500,
                        help="truncate tool results to this many characters")
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args(argv)
    print(render(args.out, args.limit, args.brief))


if __name__ == "__main__":
    main()
