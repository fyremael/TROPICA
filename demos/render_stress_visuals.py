from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "stress_summary.csv"
SVG_PATH = ARTIFACT_DIR / "stress_visuals.svg"

COLORS = {
    "Dyck adversarial decode": "#0072c3",
    "Empty support contract": "#198038",
    "JSON schema subset": "#8a3ffc",
    "Tokenizer automata": "#ff832b",
    "Tool workflow graph": "#6f6f6f",
    "Grid LTL planner": "#1192e8",
    "ControlDelta numerics": "#d12771",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    typed = []
    for row in rows:
        typed.append(
            {
                "Domain": row["Domain"],
                "Cases": float(row["Cases"]),
                "Failures": float(row["Failures"]),
                "DurationMs": float(row["DurationMs"]),
                "Notes": row["Notes"],
            }
        )
    return typed


def fmt(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def bar_rows(rows, metric: str, x: int, y: int, width: int, title: str) -> str:
    max_value = max(float(r[metric]) for r in rows) or 1.0
    label_w = 188
    bar_w = width - label_w - 76
    row_h = 34
    parts = [f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>']
    for idx, row in enumerate(rows):
        domain = str(row["Domain"])
        value = float(row[metric])
        yy = y + 32 + idx * row_h
        fill_w = int((value / max_value) * bar_w)
        parts.extend(
            [
                f'<text x="{x}" y="{yy + 15}" class="label">{escape(domain)}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="20" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{fill_w}" height="20" rx="3" fill="{COLORS[domain]}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 15}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def failure_strip(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Failure surface</text>']
    for idx, row in enumerate(rows):
        domain = str(row["Domain"])
        failures = int(float(row["Failures"]))
        xx = x + (idx % 4) * 248
        yy = y + 30 + (idx // 4) * 70
        status = "PASS" if failures == 0 else f"{failures} FAIL"
        color = "#198038" if failures == 0 else "#da1e28"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="224" height="48" rx="5" fill="#ffffff" stroke="#d0d0d0" />',
                f'<circle cx="{xx + 24}" cy="{yy + 24}" r="8" fill="{color}" />',
                f'<text x="{xx + 42}" y="{yy + 20}" class="badge-title">{escape(domain)}</text>',
                f'<text x="{xx + 42}" y="{yy + 37}" class="badge-value">{status}</text>',
            ]
        )
    return "\n".join(parts)


def notes(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Interpretation hooks</text>']
    for idx, row in enumerate(rows[:5]):
        yy = y + 30 + idx * 24
        parts.append(f'<text x="{x}" y="{yy}" class="note">{escape(str(row["Domain"]))}: {escape(str(row["Notes"]))}</text>')
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(r["Cases"]) for r in rows)
    total_failures = sum(float(r["Failures"]) for r in rows)
    total_ms = sum(float(r["DurationMs"]) for r in rows)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="900" viewBox="0 0 1180 900" role="img" aria-labelledby="title desc">
  <title id="title">Stress harness summary</title>
  <desc id="desc">Stress harness dashboard showing zero failures, case volume, and runtime by domain.</desc>
  <style>
    .bg {{ fill: #f7f8f3; }}
    .title {{ font: 700 30px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .metric {{ font: 700 26px Arial, sans-serif; fill: #161616; }}
    .metric-label {{ font: 13px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .label {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .value {{ font: 700 13px Arial, sans-serif; fill: #161616; }}
    .track {{ fill: #e4e6dc; }}
    .card {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .badge-title {{ font: 700 12px Arial, sans-serif; fill: #262626; }}
    .badge-value {{ font: 700 12px Arial, sans-serif; fill: #198038; }}
    .note {{ font: 13px Arial, sans-serif; fill: #393939; }}
  </style>
  <rect class="bg" width="1180" height="900" />
  <text x="54" y="58" class="title">Stress Harness Results</text>
  <text x="54" y="88" class="subtitle">Adversarial decode, randomized specs, tokenizer prefixes, workflow routing, and ControlDelta numerics</text>
  <rect x="54" y="122" width="214" height="82" rx="6" class="card" />
  <text x="78" y="158" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="184" class="metric-label">total cases</text>
  <rect x="292" y="122" width="214" height="82" rx="6" class="card" />
  <text x="316" y="158" class="metric">{fmt(total_failures)}</text>
  <text x="316" y="184" class="metric-label">total failures</text>
  <rect x="530" y="122" width="214" height="82" rx="6" class="card" />
  <text x="554" y="158" class="metric">{fmt(total_ms)}</text>
  <text x="554" y="184" class="metric-label">aggregate runtime ms</text>
  {failure_strip(rows, 54, 252)}
  {bar_rows(rows, "Cases", 54, 460, 520, "Case volume by domain")}
  {bar_rows(rows, "DurationMs", 630, 460, 490, "Runtime by domain, ms")}
  {notes(rows, 54, 795)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
