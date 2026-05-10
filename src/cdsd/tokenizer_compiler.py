from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class TokenizerPrefixError(ValueError):
    pass


class TokenizerRoundTripError(TokenizerPrefixError):
    pass


class TokenizerCollisionError(TokenizerPrefixError):
    pass


@dataclass
class TokenPrefixState:
    node: int = 0
    emitted: tuple[int, ...] = ()
    complete_value: str | None = None


@dataclass
class _Node:
    edges: dict[int, int] = field(default_factory=dict)
    value: str | None = None


class TokenPrefixAutomaton:
    """Deterministic token-prefix automaton for literal strings/enums."""

    def __init__(self, tokenizer: Tokenizer, literals: list[str] | tuple[str, ...], *, strict_roundtrip: bool = True):
        self.tokenizer = tokenizer
        self.nodes = [_Node()]
        self.literal_ids: dict[str, tuple[int, ...]] = {}
        token_sequences: dict[tuple[int, ...], str] = {}
        seen_literals: set[str] = set()
        for literal in literals:
            if literal in seen_literals:
                raise TokenizerCollisionError(f"Duplicate literal: {literal!r}")
            seen_literals.add(literal)
            ids = tuple(tokenizer.encode(literal))
            if not ids:
                raise TokenizerPrefixError(f"Tokenizer returned an empty tokenization for literal {literal!r}")
            if strict_roundtrip:
                decoded = tokenizer.decode(list(ids))
                if decoded != literal:
                    raise TokenizerRoundTripError(f"Tokenizer round-trip mismatch for {literal!r}: decoded {decoded!r}")
            prior = token_sequences.get(ids)
            if prior is not None and prior != literal:
                raise TokenizerCollisionError(f"Literals {prior!r} and {literal!r} compile to the same token sequence")
            token_sequences[ids] = literal
            self.literal_ids[literal] = ids
            self._insert(literal, ids)

    def _insert(self, literal: str, ids: tuple[int, ...]) -> None:
        node = 0
        for tok_id in ids:
            nxt = self.nodes[node].edges.get(tok_id)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes[node].edges[tok_id] = nxt
                self.nodes.append(_Node())
            node = nxt
        if self.nodes[node].value is not None and self.nodes[node].value != literal:
            raise TokenizerCollisionError("Two literals compiled to the same token sequence")
        self.nodes[node].value = literal

    def initial_state(self) -> TokenPrefixState:
        return TokenPrefixState()

    def _node(self, state: TokenPrefixState) -> _Node:
        if state.node < 0 or state.node >= len(self.nodes):
            raise TokenizerPrefixError(f"Invalid automaton node: {state.node}")
        return self.nodes[state.node]

    def allowed_token_ids(self, state: TokenPrefixState) -> set[int]:
        return set(self._node(state).edges)

    def update(self, state: TokenPrefixState, token_id: int) -> TokenPrefixState:
        try:
            nxt = self._node(state).edges[token_id]
        except KeyError as exc:
            raise TokenizerPrefixError(f"Token id {token_id} is not allowed from automaton node {state.node}") from exc
        emitted = state.emitted + (token_id,)
        return TokenPrefixState(node=nxt, emitted=emitted, complete_value=self.nodes[nxt].value)

    def is_accepting(self, state: TokenPrefixState) -> bool:
        return self._node(state).value is not None


class TiktokenAdapter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenizerPrefixError("tiktoken is required for TiktokenAdapter") from exc
        self.encoding_name = encoding_name
        self.name = f"tiktoken/{encoding_name}"
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return list(self.encoding.encode(text, allowed_special=set(), disallowed_special=()))

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(list(ids))


class HFTokenizerAdapter:
    def __init__(self, tokenizer, name: str = "hf"):
        self.tokenizer = tokenizer
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


class WordPieceTokenizer:
    """Tiny deterministic stand-in for BPE-style tests.

    Longest tokens win. Unknown single characters are emitted as their Unicode
    codepoint offset above the explicit vocab range so edge cases remain testable.
    """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self.inverse = {v: k for k, v in self.vocab.items()}
        self._pieces = sorted(self.vocab, key=len, reverse=True)
        self._unk_base = max(self.vocab.values(), default=0) + 1

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            piece = next((p for p in self._pieces if text.startswith(p, i)), None)
            if piece is None:
                out.append(self._unk_base + ord(text[i]))
                i += 1
            else:
                out.append(self.vocab[piece])
                i += len(piece)
        return out

    def decode(self, ids: list[int]) -> str:
        chars = []
        for tok_id in ids:
            if tok_id in self.inverse:
                chars.append(self.inverse[tok_id])
            else:
                chars.append(chr(tok_id - self._unk_base))
        return "".join(chars)
