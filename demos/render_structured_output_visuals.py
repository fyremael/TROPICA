from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "structured_output_summary.csv"
SVG_PATH = ARTIFACT_DIR / "structured_output_visuals.svg"

COLORS = {
    "tiktoken/cl100k_base": "#0072c3",
    "hf/structured-bpe": "#ff832b",
    "schema-controls": "#198038",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for row in raw:
        rows.append(
            {
                "Adapter": row["Adapter"],
                "Suite": row["Suite"],
                "Cases": float(row["Cases"]),
                "Failures": float(row["Failures"]),
                "DurationMs": float(row["DurationMs"]),
                "Outputs": float(row["Outputs"]),
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
    label_w = 218
    bar_w = width - label_w - 74
    row_h = 38
    parts = [f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>']
    for idx, row in enumerate(rows):
        adapter = str(row["Adapter"])
        label = f'{adapter}: {row["Suite"]}'
        value = float(row[metric])
        yy = y + 34 + idx * row_h
        fill_w = int((value / max_value) * bar_w)
        parts.extend(
            [
                f'<text x="{x}" y="{yy + 16}" class="label">{escape(label[:32])}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="22" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{fill_w}" height="22" rx="3" fill="{COLORS.get(adapter, "#6f6f6f")}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 16}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def failure_cards(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Fail-closed surface</text>']
    for idx, row in enumerate(rows):
        adapter = str(row["Adapter"])
        failures = int(float(row["Failures"]))
        xx = x + (idx % 3) * 300
        yy = y + 32 + (idx // 3) * 64
        status = "PASS" if failures == 0 else f"{failures} FAIL"
        color = "#198038" if failures == 0 else "#da1e28"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="270" height="48" rx="6" class="card" />',
                f'<circle cx="{xx + 24}" cy="{yy + 24}" r="8" fill="{color}" />',
                f'<text x="{xx + 42}" y="{yy + 20}" class="badge-title">{escape(adapter)}</text>',
                f'<text x="{xx + 42}" y="{yy + 37}" class="badge-value">{escape(str(row["Suite"]))}: {status}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(row["Cases"]) for row in rows)
    total_failures = sum(float(row["Failures"]) for row in rows)
    max_outputs = max(float(row["Outputs"]) for row in rows)
    adapters = sorted({str(row["Adapter"]) for row in rows})
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="850" viewBox="0 0 1280 850" role="img" aria-labelledby="title desc">
  <title id="title">Structured tool-call output summary</title>
  <desc id="desc">Bounded JSON tool-call outputs compiled to real tokenizer masks.</desc>
  <style>
    .bg {{ fill: #f7f8f3; }}
    .title {{ font: 700 31px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .metric {{ font: 700 27px Arial, sans-serif; fill: #161616; }}
    .metric-label {{ font: 13px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .label {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .value {{ font: 700 13px Arial, sans-serif; fill: #161616; }}
    .track {{ fill: #e4e6dc; }}
    .card {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .badge-title {{ font: 700 12px Arial, sans-serif; fill: #262626; }}
    .badge-value {{ font: 700 12px Arial, sans-serif; fill: #198038; }}
  </style>
  <rect class="bg" width="1280" height="850" />
  <text x="54" y="58" class="title">Structured Tool-Call Masks</text>
  <text x="54" y="88" class="subtitle">Finite JSON schema enumeration, real tokenizer-ID masks, hostile decode, and negative controls</text>
  <rect x="54" y="122" width="210" height="78" rx="6" class="card" />
  <text x="78" y="156" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="182" class="metric-label">cases</text>
  <rect x="290" y="122" width="210" height="78" rx="6" class="card" />
  <text x="314" y="156" class="metric">{fmt(total_failures)}</text>
  <text x="314" y="182" class="metric-label">failures</text>
  <rect x="526" y="122" width="210" height="78" rx="6" class="card" />
  <text x="550" y="156" class="metric">{fmt(max_outputs)}</text>
  <text x="550" y="182" class="metric-label">max compiled outputs</text>
  <rect x="762" y="122" width="410" height="78" rx="6" class="card" />
  <text x="786" y="156" class="metric">{escape(" + ".join(adapters))}</text>
  <text x="786" y="182" class="metric-label">adapter and schema-control surface</text>
  {bar_panel(rows, "Cases", 54, 260, 560, "Case volume by suite")}
  {bar_panel(rows, "Outputs", 674, 260, 520, "Compiled output frontier")}
  {failure_cards(rows, 54, 610)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
