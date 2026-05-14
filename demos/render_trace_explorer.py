from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
TRACE_PATH = ARTIFACT_DIR / "model_integration_traces.jsonl"
UNIFIED_TRACE_PATH = ARTIFACT_DIR / "unified_traces.jsonl"
TRACE_PATHS = (TRACE_PATH, UNIFIED_TRACE_PATH)
HTML_PATH = ARTIFACT_DIR / "trace_explorer.html"


def load_traces(paths: list[Path] | tuple[Path, ...] = TRACE_PATHS) -> list[dict[str, object]]:
    traces = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    traces.append(json.loads(text))
    return traces


def render_html(traces: list[dict[str, object]]) -> str:
    data_json = json.dumps(traces, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TROPICA Trace Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #161616;
      --muted: #525252;
      --line: #d0d7de;
      --blue: #0072c3;
      --red: #da1e28;
      --green: #198038;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 24px 32px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    main {{ display: grid; grid-template-columns: 390px 1fr; gap: 18px; padding: 18px 32px 32px; }}
    section {{ min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
    .section-head {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .section-head h2 {{ margin: 0; font-size: 16px; }}
    button {{ border: 1px solid var(--line); background: #f6f8fa; color: var(--ink); border-radius: 4px; padding: 6px 9px; cursor: pointer; }}
    button:hover {{ border-color: var(--blue); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eaeef2; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #f6f8fa; color: #30363d; z-index: 1; }}
    tr[data-selected="true"] {{ background: #eaf4ff; }}
    .scenario-table tr {{ cursor: pointer; }}
    .scroll {{ max-height: 690px; overflow: auto; }}
    .details {{ padding: 16px; display: grid; gap: 14px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fafbfc; }}
    .metric strong {{ display: block; font-size: 20px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .ok {{ color: var(--green); font-weight: 700; }}
    .bad {{ color: var(--red); font-weight: 700; }}
    pre {{ margin: 0; padding: 12px; overflow: auto; background: #0d1117; color: #f0f6fc; border-radius: 6px; font-size: 12px; line-height: 1.45; }}
    code {{ font-family: Consolas, Monaco, monospace; }}
    .token {{ font-family: Consolas, Monaco, monospace; white-space: pre-wrap; word-break: break-word; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; padding: 14px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TROPICA Trace Explorer</h1>
    <p>Token-by-token evidence for structured decode and unified support scenarios. The decoder may see illegal scores, but selected tokens/actions must remain inside final support.</p>
  </header>
  <main>
    <section>
      <div class="section-head">
        <h2>Scenarios</h2>
        <div>
          <button data-sort="provider">Provider</button>
          <button data-sort="family">Family</button>
          <button data-sort="steps">Steps</button>
          <button data-sort="accepted">Status</button>
        </div>
      </div>
      <div class="scroll">
        <table class="scenario-table">
          <thead><tr><th>Provider</th><th>Family</th><th>Suite</th><th>Steps</th><th>Status</th></tr></thead>
          <tbody id="scenarioRows"></tbody>
        </table>
      </div>
    </section>
    <section>
      <div class="section-head">
        <h2 id="traceTitle">Trace</h2>
      </div>
      <div class="details">
        <div class="metric-grid">
          <div class="metric"><strong id="acceptedMetric">-</strong><span>accepted</span></div>
          <div class="metric"><strong id="stepsMetric">-</strong><span>steps</span></div>
          <div class="metric"><strong id="tokenMetric">-</strong><span>emitted tokens</span></div>
          <div class="metric"><strong id="illegalMetric">-</strong><span>illegal outranks</span></div>
        </div>
        <div>
          <h2>Final Value</h2>
          <pre id="finalValue"></pre>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>Step</th><th>Allowed</th><th>Selected</th><th>Selected Text</th><th>Score</th><th>Top Illegal</th><th>Illegal Score</th><th>Accepting</th>
              </tr>
            </thead>
            <tbody id="eventRows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script id="trace-data" type="application/json">{data_json}</script>
  <script>
    const traces = JSON.parse(document.getElementById("trace-data").textContent);
    let selectedIndex = 0;
    let sortKey = "provider";

    function scenario(trace) {{
      return trace.scenario || {{}};
    }}

    function provider(trace) {{
      const s = scenario(trace);
      return s.provider || "support-contract";
    }}

    function family(trace) {{
      const s = scenario(trace);
      return trace.family || s.family || "model_integration";
    }}

    function suite(trace) {{
      const s = scenario(trace);
      return s.suite || s.name || trace.trace_type || "trace";
    }}

    function label(trace) {{
      return `${{provider(trace)}} / ${{family(trace)}} / ${{suite(trace)}}`;
    }}

    function text(value) {{
      return value === null || value === undefined ? "" : String(value);
    }}

    function fmtScore(value) {{
      return value === null || value === undefined ? "" : Number(value).toFixed(3);
    }}

    function status(trace) {{
      return trace.accepted ? '<span class="ok">PASS</span>' : '<span class="bad">STOP</span>';
    }}

    function sortedIndexes() {{
      return traces.map((_, idx) => idx).sort((a, b) => {{
        const ta = traces[a], tb = traces[b];
        if (sortKey === "steps") return Number(tb.steps || 0) - Number(ta.steps || 0);
        if (sortKey === "accepted") return Number(tb.accepted) - Number(ta.accepted);
        if (sortKey === "family") return family(ta).localeCompare(family(tb));
        return label(ta).localeCompare(label(tb));
      }});
    }}

    function allowedCount(event) {{
      if (event.allowed_count !== undefined && event.allowed_count !== null) return event.allowed_count;
      return (event.final_support || []).length;
    }}

    function selectedValue(event) {{
      if (event.selected_token_id !== undefined && event.selected_token_id !== null) return event.selected_token_id;
      return event.selected;
    }}

    function selectedText(event) {{
      if (event.selected_token_text !== undefined && event.selected_token_text !== null) return event.selected_token_text;
      return event.selected;
    }}

    function eventAccepting(event) {{
      return Boolean(event.accepted || event.accepting);
    }}

    function renderScenarios() {{
      const rows = document.getElementById("scenarioRows");
      rows.innerHTML = "";
      for (const idx of sortedIndexes()) {{
        const trace = traces[idx];
        const tr = document.createElement("tr");
        tr.dataset.selected = String(idx === selectedIndex);
        tr.innerHTML = `<td>${{text(provider(trace))}}</td><td>${{text(family(trace))}}</td><td>${{text(suite(trace))}}</td><td>${{text(trace.steps)}}</td><td>${{status(trace)}}</td>`;
        tr.addEventListener("click", () => {{ selectedIndex = idx; renderScenarios(); renderTrace(); }});
        rows.appendChild(tr);
      }}
    }}

    function renderTrace() {{
      const trace = traces[selectedIndex] || {{}};
      const events = trace.events || [];
      document.getElementById("traceTitle").textContent = label(trace);
      document.getElementById("acceptedMetric").innerHTML = status(trace);
      document.getElementById("stepsMetric").textContent = text(trace.steps);
      document.getElementById("tokenMetric").textContent = text((trace.emitted_token_ids || []).length);
      const illegalOutranks = events.filter(e => e.top_illegal_score !== null && e.top_illegal_score > e.selected_score).length;
      document.getElementById("illegalMetric").textContent = illegalOutranks;
      document.getElementById("finalValue").textContent = trace.value || trace.error || JSON.stringify(trace.parsed || "", null, 2);
      const rows = document.getElementById("eventRows");
      rows.innerHTML = "";
      for (const event of events) {{
        const illegal = event.top_illegal_token_id === null || event.top_illegal_token_id === undefined
          ? ""
          : `${{event.top_illegal_token_id}} ${{event.top_illegal_token_text ? "(" + event.top_illegal_token_text + ")" : ""}}`;
        const tr = document.createElement("tr");
        tr.title = `final_support: ${{JSON.stringify(event.final_support || event.allowed_token_ids || [])}}`;
        tr.innerHTML = `<td>${{event.step}}</td><td>${{allowedCount(event)}}</td><td>${{text(selectedValue(event))}}</td><td class="token">${{text(selectedText(event))}}</td><td>${{fmtScore(event.selected_score)}}</td><td class="token">${{illegal}}</td><td>${{fmtScore(event.top_illegal_score)}}</td><td>${{eventAccepting(event) ? "yes" : ""}}</td>`;
        rows.appendChild(tr);
      }}
    }}

    document.querySelectorAll("button[data-sort]").forEach(button => {{
      button.addEventListener("click", () => {{ sortKey = button.dataset.sort; renderScenarios(); }});
    }});

    renderScenarios();
    renderTrace();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    traces = load_traces()
    if not traces:
        raise SystemExit(f"No traces found in {', '.join(str(path) for path in TRACE_PATHS)}")
    HTML_PATH.parent.mkdir(exist_ok=True)
    HTML_PATH.write_text(render_html(traces), encoding="utf-8")
    print(f"Wrote {HTML_PATH}")
