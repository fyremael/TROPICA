# Codex Prompt: Tokenizer Compiler Engineer

Build tokenizer-correct mask compilation.

Tasks:

1. Compile literal strings and enums to token-prefix automata.
2. Support byte-level and BPE tokenizers.
3. Test Unicode, whitespace, quoting, and partial-token prefixes.
4. Provide a deterministic `allowed_token_ids(prefix_state)` API.

Acceptance:

- Enum values are generated exactly and only exactly.
- Tokenizer edge cases are covered by tests.
