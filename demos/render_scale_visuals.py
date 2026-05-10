from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "scale_summary.csv"
SVG_PATH = ARTIFACT_DIR / "scale_visuals.svg"

COLORS = {
    "Dyck horizon": "#0072c3",
    "JSON properties": "#8a3ffc",
    "Tokenizer enums": "#ff832b",
    "Workflow nodes": "#198038",
    "ControlDelta tokens": "#d12771",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for row in raw:
        rows.append(
            {
                "Track": row["Track"],
                "Size": float(row["Size"]),
                "Cases": float(row["Cases"]),
                "Failures": float(row["Failures"]),
                "DurationMs": float(row["DurationMs"]),
                "Throughput": float(row["Throughput"]),
                "PrimaryMetric": float(row["PrimaryMetric"]),
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


def line_chart(rows, metric: str, x: int, y: int, width: int, height: int, title: str) -> str:
    by_track: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_track[str(row["Track"])].append(row)
    min_size = min(float(r["Size"]) for r in rows)
    max_size = max(float(r["Size"]) for r in rows)
    min_log = math.log2(min_size)
    max_log = math.log2(max_size)
    max_metric = max(float(r[metric]) for r in rows) or 1.0
    parts = [
        f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis" />',
        f'<line x1="{x}" y1="{y + 24}" x2="{x}" y2="{y + height}" class="axis" />',
    ]
    for track, items in by_track.items():
        items = sorted(items, key=lambda item: float(item["Size"]))
        points = []
        for item in items:
            size_pos = (math.log2(float(item["Size"])) - min_log) / max(1e-9, max_log - min_log)
            px = x + size_pos * width
            py = y + height - (float(item[metric]) / max_metric) * (height - 28)
            points.append((px, py))
        point_text = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{COLORS[track]}" stroke-width="3" />')
        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{COLORS[track]}" />')
    parts.extend(
        [
            f'<text x="{x}" y="{y + height + 28}" class="hint">size sweep, log2 -></text>',
            f'<text x="{x + width - 90}" y="{y + height + 28}" class="hint">max {fmt(max_metric)}</text>',
        ]
    )
    return "\n".join(parts)


def latest_cards(rows, x: int, y: int) -> str:
    by_track: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        by_track[str(row["Track"])].append(row)
    parts = [f'<text x="{x}" y="{y}" class="panel-title">Largest-size checkpoints</text>']
    for idx, (track, items) in enumerate(by_track.items()):
        item = max(items, key=lambda row: float(row["Size"]))
        xx = x + (idx % 2) * 276
        yy = y + 30 + (idx // 2) * 76
        status = "PASS" if float(item["Failures"]) == 0 else "FAIL"
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="250" height="58" rx="6" class="card" />',
                f'<rect x="{xx}" y="{yy}" width="7" height="58" rx="3" fill="{COLORS[track]}" />',
                f'<text x="{xx + 18}" y="{yy + 22}" class="badge-title">{escape(track)}</text>',
                f'<text x="{xx + 18}" y="{yy + 43}" class="badge-value">{status} at size {fmt(float(item["Size"]))}</text>',
            ]
        )
    return "\n".join(parts)


def track_legend(rows, x: int, y: int) -> str:
    tracks = []
    for row in rows:
        track = str(row["Track"])
        if track not in tracks:
            tracks.append(track)
    parts = []
    for idx, track in enumerate(tracks):
        xx = x + idx * 210
        parts.extend(
            [
                f'<rect x="{xx}" y="{y}" width="14" height="14" rx="2" fill="{COLORS[track]}" />',
                f'<text x="{xx + 22}" y="{y + 12}" class="legend">{escape(track)}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    total_cases = sum(float(r["Cases"]) for r in rows)
    total_failures = sum(float(r["Failures"]) for r in rows)
    max_size = max(float(r["Size"]) for r in rows)
    max_throughput = max(float(r["Throughput"]) for r in rows)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="900" viewBox="0 0 1280 900" role="img" aria-labelledby="title desc">
  <title id="title">Scale harness summary</title>
  <desc id="desc">Scale sweeps for Dyck horizons, JSON properties, tokenizer enums, workflow nodes, and ControlDelta sequence lengths.</desc>
  <style>
    .bg {{ fill: #f7f8f3; }}
    .title {{ font: 700 31px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .metric {{ font: 700 27px Arial, sans-serif; fill: #161616; }}
    .metric-label {{ font: 13px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .axis {{ stroke: #c6c6c6; stroke-width: 1; }}
    .hint {{ font: 12px Arial, sans-serif; fill: #6f6f6f; }}
    .card {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .badge-title {{ font: 700 13px Arial, sans-serif; fill: #262626; }}
    .badge-value {{ font: 700 13px Arial, sans-serif; fill: #198038; }}
  </style>
  <rect class="bg" width="1280" height="900" />
  <text x="54" y="58" class="title">Scale Harness: Contract Under Growth</text>
  <text x="54" y="88" class="subtitle">Bigger horizons, bigger schemas, bigger enum automata, bigger workflow graphs, longer recurrent scans</text>
  <rect x="54" y="120" width="210" height="78" rx="6" class="card" />
  <text x="78" y="154" class="metric">{fmt(total_cases)}</text>
  <text x="78" y="180" class="metric-label">total cases / token checks</text>
  <rect x="290" y="120" width="210" height="78" rx="6" class="card" />
  <text x="314" y="154" class="metric">{fmt(total_failures)}</text>
  <text x="314" y="180" class="metric-label">failures</text>
  <rect x="526" y="120" width="210" height="78" rx="6" class="card" />
  <text x="550" y="154" class="metric">{fmt(max_size)}</text>
  <text x="550" y="180" class="metric-label">largest scale point</text>
  <rect x="762" y="120" width="260" height="78" rx="6" class="card" />
  <text x="786" y="154" class="metric">{fmt(max_throughput)}</text>
  <text x="786" y="180" class="metric-label">max cases/sec</text>
  {track_legend(rows, 54, 230)}
  {line_chart(rows, "Throughput", 64, 300, 530, 260, "Throughput by scale point")}
  {line_chart(rows, "DurationMs", 690, 300, 500, 260, "Runtime by scale point")}
  {line_chart(rows, "PrimaryMetric", 64, 650, 530, 170, "Primary growth metric")}
  {latest_cards(rows, 690, 640)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
