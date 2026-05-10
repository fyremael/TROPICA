from __future__ import annotations

import csv
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
CSV_PATH = ARTIFACT_DIR / "experiment_summary.csv"
SVG_PATH = ARTIFACT_DIR / "experiment_visuals.svg"

PALETTE = {
    "raw_generator": "#8a3ffc",
    "control_delta_only": "#d12771",
    "external_mask_only": "#0072c3",
    "grammar_only": "#6f6f6f",
    "planner_guided": "#198038",
    "control_delta_plus_external": "#ff832b",
}

ORDER = [
    "raw_generator",
    "control_delta_only",
    "external_mask_only",
    "grammar_only",
    "planner_guided",
    "control_delta_plus_external",
]

LABELS = {
    "raw_generator": "Raw generator",
    "control_delta_only": "ControlDelta only",
    "external_mask_only": "External mask only",
    "grammar_only": "Grammar only",
    "planner_guided": "Planner guided",
    "control_delta_plus_external": "ControlDelta + external",
}


def load_rows() -> list[dict[str, float | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    typed = []
    for row in rows:
        item: dict[str, float | str] = {"Mode": row["Mode"]}
        for key, value in row.items():
            if key != "Mode":
                item[key] = float(value)
        typed.append(item)
    return sorted(typed, key=lambda r: ORDER.index(str(r["Mode"])))


def fmt(value: float) -> str:
    if value < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def bar_panel(rows, metric: str, title: str, x: int, y: int, width: int, max_value: float | None = None) -> str:
    max_seen = max(float(r[metric]) for r in rows)
    scale_max = max_value if max_value is not None else max_seen
    label_w = 172
    bar_w = width - label_w - 72
    row_h = 38
    parts = [
        f'<text x="{x}" y="{y}" class="panel-title">{escape(title)}</text>',
        f'<line x1="{x + label_w}" y1="{y + 16}" x2="{x + label_w + bar_w}" y2="{y + 16}" class="axis" />',
    ]
    for idx, row in enumerate(rows):
        mode = str(row["Mode"])
        value = float(row[metric])
        yy = y + 42 + idx * row_h
        bw = 0 if scale_max == 0 else int((value / scale_max) * bar_w)
        parts.extend(
            [
                f'<text x="{x}" y="{yy + 16}" class="label">{escape(LABELS[mode])}</text>',
                f'<rect x="{x + label_w}" y="{yy}" width="{bar_w}" height="22" rx="3" class="track" />',
                f'<rect x="{x + label_w}" y="{yy}" width="{bw}" height="22" rx="3" fill="{PALETTE[mode]}" />',
                f'<text x="{x + label_w + bar_w + 14}" y="{yy + 16}" class="value">{fmt(value)}</text>',
            ]
        )
    return "\n".join(parts)


def legend(rows, x: int, y: int) -> str:
    parts = []
    for idx, row in enumerate(rows):
        mode = str(row["Mode"])
        xx = x + (idx % 3) * 250
        yy = y + (idx // 3) * 30
        parts.extend(
            [
                f'<rect x="{xx}" y="{yy}" width="14" height="14" rx="2" fill="{PALETTE[mode]}" />',
                f'<text x="{xx + 22}" y="{yy + 12}" class="legend">{escape(LABELS[mode])}</text>',
            ]
        )
    return "\n".join(parts)


def render(rows) -> str:
    invalid_zero = [LABELS[str(r["Mode"])] for r in rows if float(r["InvalidRate"]) == 0.0]
    note_modes = " + ".join(invalid_zero)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="820" viewBox="0 0 1180 820" role="img" aria-labelledby="title desc">
  <title id="title">Planner-guided support decoding experiment summary</title>
  <desc id="desc">Bar charts comparing invalid rate, allowed entropy, winner cardinality, and illegal logit pressure across ablation modes.</desc>
  <style>
    .bg {{ fill: #f7f8f3; }}
    .title {{ font: 700 30px Arial, sans-serif; fill: #161616; }}
    .subtitle {{ font: 16px Arial, sans-serif; fill: #525252; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #262626; }}
    .label {{ font: 14px Arial, sans-serif; fill: #393939; }}
    .value {{ font: 700 14px Arial, sans-serif; fill: #161616; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #393939; }}
    .axis {{ stroke: #c6c6c6; stroke-width: 1; }}
    .track {{ fill: #e4e6dc; }}
    .callout {{ fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; }}
    .callout-text {{ font: 700 15px Arial, sans-serif; fill: #198038; }}
  </style>
  <rect class="bg" width="1180" height="820" />
  <text x="54" y="58" class="title">Control-Delta Support Decoding Ablations</text>
  <text x="54" y="88" class="subtitle">Harness metrics from artifacts/experiment_summary.csv</text>
  <rect x="760" y="30" width="360" height="76" rx="6" class="callout" />
  <text x="782" y="61" class="callout-text">Zero invalid rate</text>
  <text x="782" y="86" class="subtitle">{escape(note_modes)}</text>
  {legend(rows, 54, 122)}
  {bar_panel(rows, "InvalidRate", "InvalidRate, lower is better", 54, 210, 520, 1.0)}
  {bar_panel(rows, "EntropyAllowed", "EntropyAllowed, support uncertainty", 630, 210, 490)}
  {bar_panel(rows, "WinnerCardinality", "WinnerCardinality, exposed choices", 54, 525, 520)}
  {bar_panel(rows, "IllegalLogitPressure", "IllegalLogitPressure outside support", 630, 525, 490)}
</svg>
'''


if __name__ == "__main__":
    rows = load_rows()
    SVG_PATH.parent.mkdir(exist_ok=True)
    SVG_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
