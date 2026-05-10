from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "model_integration_summary.csv"
SVG_PATH = ARTIFACT_DIR / "model_integration_visuals.svg"

COLORS = {
    "scripted": "#0072c3",
    "hostile": "#da1e28",
    "callable": "#198038",
    "negative-controls": "#8a3ffc",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for row in raw:
        rows.append(
            {
                "Provider": row["Provider"],
                "Adapter": row["Adapter"],
                "Suite": row["Suite"],
                "Cases": float(row["Cases"]),
                "Failures": float(row["Failures"]),
                "DurationMs": float(row["DurationMs"]),
                "Outputs": float(row["Outputs"]),
                "TraceSteps": float(row["TraceSteps"]),
                "Notes": row["Notes"],
            }
        )
    return rows


def fmt(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def bar_panel(rows, metric: str, x: int, y: int, width: int, title: str) -> str:
    max_value = max(float(row[metric]) for row in rows) or 1.0
    label_w = 230
    bar_w = width - label_w - 80
    row_h = 46
    parts = [f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>']
    for idx, row in enumerate(rows):
        provider = str(row["Provider"])
        value = float(row[metric])
        yy = y + 34 + idx * row_h
        fill_w = int((value / max_value) * bar_w)
        label = f'{provider}: {row["Suite"]}'
        parts.extend(
            [
                f'<text x="{x}" y="{yy + 16}" class="label">{escape(label[:34])}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="24" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{fill_w}" height="24" rx="3" fill="{COLORS.get(provider, "#6f6f6f")}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 17}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def status_cards(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Provider fail-closed status</text>']
    for idx, row in enumerate(rows):
        provider = str(row["Provider"])
        failures = int(float(row["Failures"]))
        xx = x + (idx % 2) * 420
        yy = y + 34 + (idx // 2) * 74
        status = "PASS" if failures == 0 else f"{failures} FAIL"
        color = "#198038" if failures == 0 else "#da1e28"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="380" height="56" rx="6" class="card" />',
                f'<circle cx="{xx + 26}" cy="{yy + 28}" r="9" fill="{color}" />',
                f'<text x="{xx + 48}" y="{yy + 23}" class="badge-title">{escape(provider)}</text>',
                f'<text x="{xx + 48}" y="{yy + 42}" class="badge-value">{escape(str(row["Suite"]))}: {status}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(row["Cases"]) for row in rows)
    total_failures = sum(float(row["Failures"]) for row in rows)
    trace_steps = sum(float(row["TraceSteps"]) for row in rows)
    max_outputs = max(float(row["Outputs"]) for row in rows)
    providers = sorted({str(row["Provider"]) for row in rows})
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="820" viewBox="0 0 1280 820" role="img" aria-labelledby="title desc">
  <title id="title">Offline model integration summary</title>
  <desc id="desc">Structured output decoder providers, traces, and fail-closed evidence.</desc>
  <style>
    .bg {{ fill: #f6f7fb; }}
    .title {{ font: 700 31px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .metric {{ font: 700 27px Arial, sans-serif; fill: #161616; }}
    .metric-label {{ font: 13px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .label {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .value {{ font: 700 13px Arial, sans-serif; fill: #161616; }}
    .track {{ fill: #dde3ee; }}
    .card {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .badge-title {{ font: 700 13px Arial, sans-serif; fill: #262626; }}
    .badge-value {{ font: 700 12px Arial, sans-serif; fill: #198038; }}
  </style>
  <rect class="bg" width="1280" height="820" />
  <text x="54" y="58" class="title">Offline Model Integration SDK</text>
  <text x="54" y="88" class="subtitle">Model-facing providers drive real tokenizer masks while illegal logits fail closed</text>
  <rect x="54" y="122" width="210" height="78" rx="6" class="card" />
  <text x="78" y="156" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="182" class="metric-label">cases</text>
  <rect x="290" y="122" width="210" height="78" rx="6" class="card" />
  <text x="314" y="156" class="metric">{fmt(total_failures)}</text>
  <text x="314" y="182" class="metric-label">failures</text>
  <rect x="526" y="122" width="210" height="78" rx="6" class="card" />
  <text x="550" y="156" class="metric">{fmt(trace_steps)}</text>
  <text x="550" y="182" class="metric-label">trace events</text>
  <rect x="762" y="122" width="210" height="78" rx="6" class="card" />
  <text x="786" y="156" class="metric">{fmt(max_outputs)}</text>
  <text x="786" y="182" class="metric-label">compiled outputs</text>
  <rect x="998" y="122" width="220" height="78" rx="6" class="card" />
  <text x="1022" y="156" class="metric">{escape(", ".join(providers))}</text>
  <text x="1022" y="182" class="metric-label">provider surfaces</text>
  {bar_panel(rows, "Cases", 54, 260, 570, "Case volume by provider")}
  {bar_panel(rows, "TraceSteps", 674, 260, 530, "Trace events generated")}
  {status_cards(rows, 54, 570)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
