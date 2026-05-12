# Offline Model Integration

TROPICA exposes an offline structured decode SDK for model-facing loops. A
model adapter supplies logits for the current prefix and legal token IDs; the
decoder selects only from the legal support and records trace events.

## Provider Contract

```python
from collections.abc import Mapping

class LogitProvider:
    def next_logits(
        self,
        emitted_token_ids: tuple[int, ...],
        allowed_token_ids: set[int],
    ) -> Mapping[int, float]:
        ...
```

The provider may return scores for legal and illegal token IDs. Illegal scores
are useful for diagnostics, but they are never selectable.

## Decode Loop

```python
from cdsd import HostileLogitProvider, StructuredOutputCompiler, StructuredOutputDecoder, TiktokenAdapter, ToolCallSpec

spec = ToolCallSpec(
    "search",
    {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "enum": ["alpha", "beta"]}},
        "additionalProperties": False,
    },
)

compiler = StructuredOutputCompiler(TiktokenAdapter("cl100k_base"), [spec])
result = StructuredOutputDecoder(compiler).decode(HostileLogitProvider(), max_steps=256)
```

`result` contains:

- `value`: completed canonical JSON string
- `parsed`: parsed JSON object
- `emitted_token_ids`: emitted token ID tuple
- `steps`: number of decode steps
- `accepted`: whether decoding reached an accepting state
- `events`: ordered trace events with allowed counts, selected IDs, selected
  scores, and top illegal token metadata when present

Export a JSON-safe trace:

```python
trace = result.to_trace_dict(TiktokenAdapter("cl100k_base"), scenario={"provider": "hostile"})
```

Trace records include the final value, parsed object, emitted token IDs, decoded
text when a tokenizer is provided, and ordered events. Each event includes
allowed token IDs, selected token ID/text, selected score, top illegal token
ID/text/score when present, and accepting state.

## Built-In Offline Providers

- `HostileLogitProvider`: gives illegal token IDs very high scores to prove the
  mask fails closed.
- `ScriptedLogitProvider`: follows a target token sequence for exact deterministic
  examples.
- `CallableLogitProvider`: wraps a user callback for custom offline adapters.
- `HFLocalLogitProvider`: experimental bridge for already-loaded Hugging Face
  model/tokenizer objects. It never downloads models; pass objects you loaded
  yourself.

```python
from cdsd import HFLocalLogitProvider

provider = HFLocalLogitProvider(model=my_model, tokenizer=my_tokenizer, device="cpu")
result = decoder.decode(provider, max_steps=256)
```

Install the optional adapter metadata with:

```bash
python -m pip install -e ".[local-models]"
```

`torch` remains controlled by the user environment. Missing dependencies or
missing model/tokenizer objects raise `LocalModelBridgeError`.

## Evidence

The model-integration report track runs:

- scripted exact generation over every canonical tool-call output
- hostile decode where illegal IDs outrank legal IDs
- callable-provider smoke tests
- negative controls for max-step exhaustion, illegal transitions, truncation,
  and schema mismatch

Run it directly:

```bash
python demos/run_model_integration_harness.py
python demos/render_model_integration_visuals.py
python demos/render_trace_explorer.py
```

Or run it through the full evidence gate:

```bash
cdsd-report --with-pytest --artifacts artifacts --jobs 4
```

The SDK is intentionally offline. Hosted APIs and network model calls are out of
scope for this milestone; a local model can be adapted by implementing
`LogitProvider` or by wrapping already-loaded Hugging Face objects with
`HFLocalLogitProvider`.
