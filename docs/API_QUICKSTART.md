# API Quickstart

```python
from cdsd import StructuredOutputCompiler, TiktokenAdapter, ToolCallSpec

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
state = compiler.initial_state()

while not compiler.is_accepting(state):
    allowed = compiler.allowed_token_ids(state)
    token_id = min(allowed)
    state = compiler.update(state, token_id)

print(compiler.complete_value(state))
```

The generator may rank allowed tokens however it likes. Tokens outside
`allowed_token_ids(state)` are not candidates.
