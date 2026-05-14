# API Quickstart

```python
from cdsd import HostileLogitProvider, StructuredOutputCompiler, StructuredOutputDecoder, TiktokenAdapter, ToolCallSpec

spec = ToolCallSpec(
    "search",
    {
        "type": "object",
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "enum": ["alpha", "beta"]},
            "limit": {"type": "integer", "enum": [1, 3]},
        },
        "additionalProperties": False,
    },
)

compiler = StructuredOutputCompiler(TiktokenAdapter("cl100k_base"), [spec])
decoder = StructuredOutputDecoder(compiler)
result = decoder.decode(HostileLogitProvider(), max_steps=256)

print(result.value)
print(result.parsed)
print(result.events[-1])
```

`HostileLogitProvider` deliberately assigns high scores to illegal token IDs.
The decoder still selects only from `allowed_token_ids(state)`, records a trace
event per emitted token, and returns the completed JSON/tool-call value.

For lower-level planner/guard work, import the formal support contract:

```python
from cdsd import SupportMask
from cdsd.contracts import ensure_selected_in_support, validate_intersection

plan = SupportMask.from_iter(["SEARCH", "ASK"])
guard = SupportMask.from_iter(["SEARCH"])
final = plan & guard

validate_intersection(final, plan, guard)
ensure_selected_in_support("SEARCH", final)
```

The same contract fields are emitted into `unified_traces.jsonl` and rendered in
`trace_explorer.html` after `cdsd-report` runs.
