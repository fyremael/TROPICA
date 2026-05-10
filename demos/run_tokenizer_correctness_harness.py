from __future__ import annotations

import csv
import importlib.util
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.tokenizer_compiler import (
    HFTokenizerAdapter,
    TiktokenAdapter,
    TokenPrefixAutomaton,
    TokenizerCollisionError,
    TokenizerPrefixError,
    TokenizerRoundTripError,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "tokenizer_correctness_summary.csv"
OUT_MD = ARTIFACT_DIR / "tokenizer_correctness_summary.md"


@dataclass
class TokenizerResult:
    adapter: str
    suite: str
    cases: int
    failures: int
    duration_ms: float
    nodes: float
    notes: str


def require_real_tokenizers() -> None:
    missing = [name for name in ["tiktoken", "tokenizers"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing real tokenizer dependencies: {', '.join(missing)}")


def tiktoken_literals(count: int) -> list[str]:
    base = [
        "a",
        "ab",
        "abc",
        '"red"',
        " leading",
        "trailing ",
        "\tindented",
        "line\nbreak",
        "snowman \u2603",
        "accent \u00e9",
        '{"key":"value"}',
    ]
    generated = [f'{{"enum":"value_{i:04d}","path":"shared/{i:04d}","space":"x {i}"}}' for i in range(count - len(base))]
    return base + generated


def wordpiece_literals(count: int) -> list[str]:
    base = ["a", "ab", "abc", '"value"', "hello worlds"]
    generated = []
    for i in range(count - len(base)):
        if i % 3 == 0:
            generated.append(f"w{i:04d}")
        elif i % 3 == 1:
            generated.append(f"w{i:04d} w{i + 1:04d}")
        else:
            generated.append(f'"v{i:04d}"')
    return base + generated


def bpe_literals(count: int) -> list[str]:
    base = ["a", "ab", "abc", '"value"', '{"k":"v0001"}']
    generated = []
    for i in range(count - len(base)):
        j = i + 10000
        if i % 2 == 0:
            generated.append(f"tok{j:04d} end{j:04d}")
        else:
            generated.append(f'{{"k":"v{j:04d}"}}')
    return base + generated


def make_wordpiece_adapter(literals: list[str]) -> HFTokenizerAdapter:
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {"[UNK]": 0, "world": 1, "##s": 2}
    for literal in literals:
        for word in literal.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    return HFTokenizerAdapter(tokenizer, name="hf/wordpiece")


def make_bpe_adapter(literals: list[str]) -> HFTokenizerAdapter:
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit

    alphabet = sorted({ch for literal in literals for word in literal.split() for ch in word})
    vocab = {"[UNK]": 0}
    for ch in alphabet:
        if ch not in vocab:
            vocab[ch] = len(vocab)
        suffixed = f"{ch}</w>"
        if suffixed not in vocab:
            vocab[suffixed] = len(vocab)
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    return HFTokenizerAdapter(tokenizer, name="hf/bpe")


def find_disallowed(allowed: set[int]) -> int:
    token_id = 0
    while token_id in allowed:
        token_id += 1
    return token_id


def run_positive(adapter, suite: str, literals: list[str]) -> TokenizerResult:
    start = time.perf_counter()
    failures = 0
    nodes = 0
    try:
        automaton = TokenPrefixAutomaton(adapter, literals)
        nodes = len(automaton.nodes)
        for literal in literals:
            state = automaton.initial_state()
            for token_id in adapter.encode(literal):
                if token_id not in automaton.allowed_token_ids(state):
                    failures += 1
                    break
                state = automaton.update(state, token_id)
            if state.complete_value != literal or adapter.decode(list(state.emitted)) != literal:
                failures += 1
    except Exception:
        failures = len(literals)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return TokenizerResult(adapter.name, suite, len(literals), failures, duration_ms, float(nodes), "exact enum generation")


def run_adapter_negatives(adapter, literal: str) -> TokenizerResult:
    start = time.perf_counter()
    failures = 0
    cases = 3
    try:
        automaton = TokenPrefixAutomaton(adapter, [literal])
        root = automaton.initial_state()
        try:
            automaton.update(root, find_disallowed(automaton.allowed_token_ids(root)))
            failures += 1
        except TokenizerPrefixError:
            pass

        ids = adapter.encode(literal)
        if len(ids) < 2:
            failures += 1
        else:
            state = root
            for token_id in ids[:-1]:
                state = automaton.update(state, token_id)
            if automaton.is_accepting(state):
                failures += 1

        state = root
        for token_id in ids:
            state = automaton.update(state, token_id)
        try:
            automaton.update(state, find_disallowed(automaton.allowed_token_ids(state)))
            failures += 1
        except TokenizerPrefixError:
            pass
    except Exception:
        failures += 1
    duration_ms = (time.perf_counter() - start) * 1000.0
    return TokenizerResult(adapter.name, "negative transitions", cases, failures, duration_ms, 0.0, "invalid id, truncated value, illegal suffix")


def run_compiler_negatives() -> TokenizerResult:
    class EmptyTokenizer:
        name = "negative-controls/empty"

        def encode(self, text):
            return []

        def decode(self, ids):
            return ""

    class LossyTokenizer:
        name = "negative-controls/lossy"

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "lossy"

    class CollisionTokenizer:
        name = "negative-controls/collision"

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "ignored"

    start = time.perf_counter()
    failures = 0
    checks = [
        lambda: TokenPrefixAutomaton(EmptyTokenizer(), ["x"]),
        lambda: TokenPrefixAutomaton(LossyTokenizer(), ["exact"]),
        lambda: TokenPrefixAutomaton(CollisionTokenizer(), ["a", "b"], strict_roundtrip=False),
        lambda: TokenPrefixAutomaton(TiktokenAdapter(), ["same", "same"]),
    ]
    expected = [TokenizerPrefixError, TokenizerRoundTripError, TokenizerCollisionError, TokenizerCollisionError]
    for fn, exc_type in zip(checks, expected):
        try:
            fn()
            failures += 1
        except exc_type:
            pass
    duration_ms = (time.perf_counter() - start) * 1000.0
    return TokenizerResult("negative-controls", "compile-time failures", len(checks), failures, duration_ms, 0.0, "empty, lossy, collision, duplicate")


def write_results(results: list[TokenizerResult]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["Adapter", "Suite", "Cases", "Failures", "DurationMs", "Nodes", "Notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Adapter": result.adapter,
                    "Suite": result.suite,
                    "Cases": result.cases,
                    "Failures": result.failures,
                    "DurationMs": f"{result.duration_ms:.3f}",
                    "Nodes": f"{result.nodes:.1f}",
                    "Notes": result.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Adapter | Suite | Cases | Failures | DurationMs | Nodes | Notes |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for result in results:
            fh.write(
                f"| {result.adapter} | {result.suite} | {result.cases} | {result.failures} | "
                f"{result.duration_ms:.3f} | {result.nodes:.1f} | {result.notes} |\n"
            )


if __name__ == "__main__":
    require_real_tokenizers()
    tiktoken_adapter = TiktokenAdapter("cl100k_base")
    wp_literals = wordpiece_literals(1800)
    bpe_literals_ = bpe_literals(1800)
    adapters = [
        (tiktoken_adapter, "tiktoken exact literals", tiktoken_literals(2600), 'enum/shared/prefix/000001/value 1'),
        (make_wordpiece_adapter(wp_literals), "hf wordpiece exact literals", wp_literals, "w0001 w0002"),
        (make_bpe_adapter(bpe_literals_), "hf bpe exact literals", bpe_literals_, "tok0001 end0001"),
    ]
    results: list[TokenizerResult] = []
    for adapter, suite, literals, negative_literal in adapters:
        results.append(run_positive(adapter, suite, literals))
        results.append(run_adapter_negatives(adapter, negative_literal))
    results.append(run_compiler_negatives())
    write_results(results)
    for result in results:
        print(
            f"{result.adapter}: suite={result.suite} cases={result.cases} failures={result.failures} "
            f"duration_ms={result.duration_ms:.3f} nodes={result.nodes:.1f} notes={result.notes}"
        )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    if any(result.failures for result in results):
        raise SystemExit(1)
