from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "tokenizer_correctness_summary.csv"
SVG_PATH = ARTIFACT_DIR / "tokenizer_correctness_visuals.svg"

COLORS = {
    "tiktoken/cl100k_base": "#0072c3",
    "hf/wordpiece": "#8a3ffc",
    "hf/bpe": "#ff832b",
    "negative-controls": "#198038",
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
                "Nodes": float(row["Nodes"]),
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
    max_value = max(float(r[metric]) for r in rows) or 1.0
    label_w = 230
    bar_w = width - label_w - 78
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
                f'<text x="{x}" y="{yy + 16}" class="label">{escape(label[:34])}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="22" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{fill_w}" height="22" rx="3" fill="{COLORS.get(adapter, "#6f6f6f")}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 16}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def failure_cards(rows, x: int, y: int) -> str:
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Negative controls and failure surface</text>']
    for idx, row in enumerate(rows):
        adapter = str(row["Adapter"])
        failures = int(float(row["Failures"]))
        xx = x + (idx % 2) * 284
        yy = y + 30 + (idx // 2) * 64
        status = "PASS" if failures == 0 else f"{failures} FAIL"
        color = "#198038" if failures == 0 else "#da1e28"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="256" height="48" rx="6" class="card" />',
                f'<circle cx="{xx + 24}" cy="{yy + 24}" r="8" fill="{color}" />',
                f'<text x="{xx + 42}" y="{yy + 20}" class="badge-title">{escape(adapter)}</text>',
                f'<text x="{xx + 42}" y="{yy + 37}" class="badge-value">{escape(str(row["Suite"]))}: {status}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(r["Cases"]) for r in rows)
    total_failures = sum(float(r["Failures"]) for r in rows)
    adapters = sorted({str(r["Adapter"]) for r in rows if not str(r["Adapter"]).startswith("negative")})
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="850" viewBox="0 0 1280 850" role="img" aria-labelledby="title desc">
  <title id="title">Tokenizer correctness summary</title>
  <desc id="desc">Real tokenizer adapter exactness and negative-control results.</desc>
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
  <text x="54" y="58" class="title">Production Tokenizer Correctness</text>
  <text x="54" y="88" class="subtitle">Real adapters, strict round-trip compilation, exact enum generation, and failure-closed controls</text>
  <rect x="54" y="122" width="210" height="78" rx="6" class="card" />
  <text x="78" y="156" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="182" class="metric-label">cases</text>
  <rect x="290" y="122" width="210" height="78" rx="6" class="card" />
  <text x="314" y="156" class="metric">{fmt(total_failures)}</text>
  <text x="314" y="182" class="metric-label">failures</text>
  <rect x="526" y="122" width="420" height="78" rx="6" class="card" />
  <text x="550" y="156" class="metric">{escape(" + ".join(adapters))}</text>
  <text x="550" y="182" class="metric-label">real adapter families</text>
  {bar_panel(rows, "Cases", 54, 260, 560, "Case volume by suite")}
  {bar_panel(rows, "Nodes", 674, 260, 520, "Compiled automaton nodes")}
  {failure_cards(rows, 54, 610)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
