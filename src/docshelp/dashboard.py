"""A monitoring dashboard as one static HTML file: `uv run poe dashboard` (lesson 9.9).

LangSmith gives every project a prebuilt dashboard (traces, latency, errors, tokens, cost, feedback) and lets you
build custom ones: charts of any metric, filtered and grouped by metadata such as plan or A/B variant. This is
the local twin: it reads the trace, feedback and alert logs that Part 9 writes and renders the same views into
`data/reports/dashboard.html`. No server, no JavaScript, no CDN: open the file in any browser.
"""

from __future__ import annotations

import html
from pathlib import Path

from docshelp.config import DATA_DIR
from docshelp.console import ensure_utf8
from docshelp.monitoring import (
    ALERTS_LOG,
    FEEDBACK_LOG,
    TRACE_LOG,
    feedback_by_trace,
    load_pricing,
    metrics,
    question_of,
    summarize,
    trace_cost,
)
from docshelp.observability import load_rows

DASHBOARD_PATH = DATA_DIR / "reports" / "dashboard.html"

STYLE = """
:root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --card: #f5f5f7; --bar: #3b6fd8; --bad: #c6392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161618; --fg: #f5f5f7; --muted: #a1a1a6; --card: #232326; --bar: #6d9cf5; --bad: #f2776a; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 16px; }
h1 { font-size: 24px; } h2 { font-size: 18px; margin-top: 32px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { background: var(--card); border-radius: 10px; padding: 12px; }
.tile .v { font-size: 22px; font-weight: 600; } .tile .k { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; width: 100%; } td, th { text-align: left; padding: 6px 8px;
        border-bottom: 1px solid var(--card); font-size: 14px; } th { color: var(--muted); font-weight: 500; }
.bad { color: var(--bad); } svg text { fill: var(--muted); font-size: 11px; }
.scroll { overflow-x: auto; }
"""


def tile(label: str, value: str, bad: bool = False) -> str:
    return f'<div class="tile"><div class="v{" bad" if bad else ""}">{html.escape(value)}</div><div class="k">{label}</div></div>'


def bars(values: dict[str, float], unit: str = "", width: int = 640) -> str:
    """A horizontal bar chart as inline SVG."""
    if not values:
        return "<p>No data yet.</p>"
    top = max(values.values()) or 1
    rows = []
    for i, (label, value) in enumerate(values.items()):
        length = int((width - 220) * value / top)
        y = i * 26
        rows.append(
            f'<text x="0" y="{y + 15}">{html.escape(label[:28])}</text>'
            f'<rect x="200" y="{y + 4}" width="{max(length, 1)}" height="16" rx="3" fill="var(--bar)"/>'
            f'<text x="{206 + length}" y="{y + 15}">{value:,.2f}{unit}</text>'
        )
    return f'<div class="scroll"><svg width="{width}" height="{len(values) * 26}">{"".join(rows)}</svg></div>'


def table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def histogram(latencies: list[float], buckets: int = 8) -> dict[str, float]:
    if not latencies:
        return {}
    top = max(latencies) or 1
    counts = [0] * buckets
    for value in latencies:
        counts[min(int(value / top * buckets), buckets - 1)] += 1
    step = top / buckets
    return {f"{i * step:.1f}-{(i + 1) * step:.1f} s": float(c) for i, c in enumerate(counts)}


def render(summaries: list[dict], feedback: list[dict], alerts: list[dict], pricing: dict) -> str:
    overall = metrics(summaries, feedback, pricing)
    scores = feedback_by_trace(feedback)
    sections = [
        f"<h1>DocsHelp monitoring</h1><p>{overall['traces']} traces from the local trace log.</p>",
        '<div class="tiles">'
        + tile("traces", f"{overall['traces']}")
        + tile("error rate", f"{overall['error_rate']:.0%}", overall["error_rate"] > 0.05)
        + tile("p50 latency", f"{overall['p50_latency']:.2f} s")
        + tile("p95 latency", f"{overall['p95_latency']:.2f} s")
        + tile("tokens", f"{overall['tokens']:,}")
        + tile("cost", f"${overall['cost']:.4f}")
        + tile("negative sentiment", f"{overall['negative_rate']:.0%}", overall["negative_rate"] > 0.2)
        + tile("injection attempts", f"{overall['flagged']}", overall["flagged"] > 0)
        + "</div>",
        "<h2>Latency distribution</h2>" + bars(histogram([s["latency"] for s in summaries]), " traces"),
    ]
    for group in ("variant", "plan"):
        values = sorted({s["metadata"].get(group, "-") for s in summaries})
        rows = []
        for value in values:
            members = [s for s in summaries if s["metadata"].get(group, "-") == value]
            m = metrics(members, feedback, pricing)
            rows.append(
                [
                    value,
                    m["traces"],
                    f"{m['error_rate']:.0%}",
                    f"{m['p95_latency']:.2f} s",
                    m["tokens"],
                    f"${m['cost']:.4f}",
                    f"{m['negative_rate']:.0%}",
                ]
            )
        sections.append(
            f"<h2>By {group}</h2>" + table([group, "traces", "errors", "p95", "tokens", "cost", "negative"], rows)
        )
    keys = sorted({k for s in scores.values() for k in s})
    averages = {k: sum(s[k] for s in scores.values() if k in s) / sum(k in s for s in scores.values()) for k in keys}
    sections.append("<h2>Online evaluator scores (average)</h2>" + bars(averages))
    slowest = sorted(summaries, key=lambda s: -s["latency"])[:5]
    sections.append(
        "<h2>Slowest traces</h2>"
        + table(
            ["question", "latency", "tokens", "cost", "status"],
            [
                [
                    question_of(s)[:70],
                    f"{s['latency']:.2f} s",
                    s["total_tokens"],
                    f"${trace_cost(s, pricing):.5f}",
                    s["status"],
                ]
                for s in slowest
            ],
        )
    )
    sections.append(
        "<h2>Alerts</h2>"
        + (
            table(
                ["alert", "metric", "value", "threshold", "window"],
                [[a["alert"], a["metric"], a["value"], a["threshold"], f"{a['window_minutes']} min"] for a in alerts],
            )
            if alerts
            else "<p>No alerts fired.</p>"
        )
    )
    body = "\n".join(sections)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>DocsHelp monitoring</title>'
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )


def write_dashboard(
    path: Path = DASHBOARD_PATH,
    *,
    trace_log: Path = TRACE_LOG,
    feedback_log: Path = FEEDBACK_LOG,
    alerts_log: Path = ALERTS_LOG,
) -> Path:
    page = render(summarize(load_rows(trace_log)), load_rows(feedback_log), load_rows(alerts_log), load_pricing())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    ensure_utf8()
    path = write_dashboard()
    print(f"Wrote {path} ({len(load_rows(TRACE_LOG))} runs). Open it in a browser.")
    if not load_rows(TRACE_LOG):
        print("The trace log is empty: run the Part 9 lessons first, e.g. `uv run poe lesson 9.1`.")
