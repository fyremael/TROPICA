from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "unified_trace_summary.csv"
SVG_PATH = ARTIFACT_DIR / "unified_trace_visuals.svg"


COLORS = {
    "dyck": "#0072c3",
    "json_schema": "#198038",
    "workflow": "#8a3ffc",
    "grid": "#f1c21b",
    "tokenizer": "#ff832b",
    "control_delta": "#1192e8",
    "contract": "#da1e28",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    return [
        {
            "Family": row["Family"],
            "Cases": float(row["Cases"]),
            "Failures": float(row["Failures"]),
            "TraceEvents": float(row["TraceEvents"]),
            "NegativeControls": float(row["NegativeControls"]),
            "DurationMs": float(row["DurationMs"]),
            "Notes": row["Notes"],
        }
        for row in raw
    ]


def fmt(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def bar_panel(rows, metric: str, x: int, y: int, width: int, title: str) -> str:
    max_value = max(float(row[metric]) for row in rows) or 1.0
    label_w = 150
    bar_w = width - label_w - 72
    row_h = 42
    parts = [f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>']
    for idx, row in enumerate(rows):
        family = str(row["Family"])
        value = float(row[metric])
        yy = y + 32 + idx * row_h
        fill_w = max(2, int((value / max_value) * bar_w)) if value else 0
        parts.extend(
            [
                f'<text x="{x}" y="{yy + 16}" class="label">{escape(family)}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="22" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{fill_w}" height="22" rx="3" fill="{COLORS.get(family, "#6f6f6f")}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 16}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def status_grid(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Fail-closed contract status</text>']
    for idx, row in enumerate(rows):
        family = str(row["Family"])
        failures = int(float(row["Failures"]))
        xx = x + (idx % 2) * 360
        yy = y + 34 + (idx // 2) * 68
        status = "PASS" if failures == 0 else f"{failures} FAIL"
        color = "#198038" if failures == 0 else "#da1e28"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="330" height="52" rx="6" class="card" />',
                f'<circle cx="{xx + 24}" cy="{yy + 26}" r="8" fill="{color}" />',
                f'<text x="{xx + 44}" y="{yy + 22}" class="badge-title">{escape(family)}</text>',
                f'<text x="{xx + 44}" y="{yy + 40}" class="badge-value">{status} - {escape(str(row["Notes"])[:34])}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(row["Cases"]) for row in rows)
    total_failures = sum(float(row["Failures"]) for row in rows)
    trace_events = sum(float(row["TraceEvents"]) for row in rows)
    negative_controls = sum(float(row["NegativeControls"]) for row in rows)
    families = len({str(row["Family"]) for row in rows})
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="840" viewBox="0 0 1280 840" role="img" aria-labelledby="title desc">
  <title id="title">Unified support trace summary</title>
  <desc id="desc">Unified planner, guard, policy, tokenizer, and ControlDelta trace evidence.</desc>
  <style>
    .bg {{ fill: #f6f7fb; }}
    .title {{ font: 700 31px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .metric {{ font: 700 28px Arial, sans-serif; fill: #161616; }}
    .metric-label {{ font: 13px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .label {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .value {{ font: 700 13px Arial, sans-serif; fill: #161616; }}
    .track {{ fill: #dde3ee; }}
    .card {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .badge-title {{ font: 700 13px Arial, sans-serif; fill: #262626; }}
    .badge-value {{ font: 700 12px Arial, sans-serif; fill: #198038; }}
  </style>
  <rect class="bg" width="1280" height="840" />
  <text x="54" y="58" class="title">Unified Contract Trace Surface</text>
  <text x="54" y="88" class="subtitle">Every selected action is checked against planner, guard, policy, and final support evidence</text>
  <rect x="54" y="122" width="210" height="78" rx="6" class="card" />
  <text x="78" y="156" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="182" class="metric-label">cases</text>
  <rect x="290" y="122" width="210" height="78" rx="6" class="card" />
  <text x="314" y="156" class="metric">{fmt(total_failures)}</text>
  <text x="314" y="182" class="metric-label">failures</text>
  <rect x="526" y="122" width="210" height="78" rx="6" class="card" />
  <text x="550" y="156" class="metric">{fmt(trace_events)}</text>
  <text x="550" y="182" class="metric-label">trace events</text>
  <rect x="762" y="122" width="210" height="78" rx="6" class="card" />
  <text x="786" y="156" class="metric">{fmt(negative_controls)}</text>
  <text x="786" y="182" class="metric-label">negative controls</text>
  <rect x="998" y="122" width="210" height="78" rx="6" class="card" />
  <text x="1022" y="156" class="metric">{families}</text>
  <text x="1022" y="182" class="metric-label">trace families</text>
  {bar_panel(rows, "TraceEvents", 54, 260, 560, "Trace events by family")}
  {bar_panel(rows, "Cases", 674, 260, 520, "Case coverage by family")}
  {status_grid(rows, 54, 590)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
