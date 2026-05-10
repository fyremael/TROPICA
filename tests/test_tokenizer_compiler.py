import importlib.util

import pytest

from cdsd.tokenizer_compiler import (
    ByteTokenizer,
    HFTokenizerAdapter,
    TiktokenAdapter,
    TokenPrefixAutomaton,
    TokenizerCollisionError,
    TokenizerPrefixError,
    TokenizerRoundTripError,
    WordPieceTokenizer,
)


def test_byte_tokenizer_unicode_enum_exact():
    automaton = TokenPrefixAutomaton(ByteTokenizer(), ["yes", "yeti", "snow ☃"])
    state = automaton.initial_state()
    assert ord("y") in automaton.allowed_token_ids(state)
    for tok_id in ByteTokenizer().encode("yeti"):
        state = automaton.update(state, tok_id)
    assert automaton.is_accepting(state)
    assert state.complete_value == "yeti"
    assert automaton.allowed_token_ids(state) == set()


def test_wordpiece_partial_prefixes_and_whitespace():
    tokenizer = WordPieceTokenizer({"foo": 1, "bar": 2, " ": 3, "\"": 4, "baz": 5})
    automaton = TokenPrefixAutomaton(tokenizer, ['foo bar', '"baz"'])
    state = automaton.initial_state()
    assert automaton.allowed_token_ids(state) == {1, 4}
    state = automaton.update(state, 4)
    assert automaton.allowed_token_ids(state) == {5}


def test_illegal_transition_truncated_sequence_and_suffix_rejection():
    tokenizer = ByteTokenizer()
    automaton = TokenPrefixAutomaton(tokenizer, ["a", "ab"])
    state = automaton.update(automaton.initial_state(), ord("a"))
    assert automaton.is_accepting(state)
    assert ord("b") in automaton.allowed_token_ids(state)
    with pytest.raises(TokenizerPrefixError):
        automaton.update(state, ord("x"))
    truncated = TokenPrefixAutomaton(tokenizer, ["snow"]).update(TokenPrefixAutomaton(tokenizer, ["snow"]).initial_state(), ord("s"))
    assert not TokenPrefixAutomaton(tokenizer, ["snow"]).is_accepting(truncated)


def test_duplicate_empty_lossy_and_collision_controls():
    class EmptyTokenizer:
        def encode(self, text):
            return []

        def decode(self, ids):
            return ""

    class LossyTokenizer:
        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "lower"

    class CollisionTokenizer:
        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "ignored"

    with pytest.raises(TokenizerCollisionError):
        TokenPrefixAutomaton(ByteTokenizer(), ["same", "same"])
    with pytest.raises(TokenizerPrefixError):
        TokenPrefixAutomaton(EmptyTokenizer(), ["x"])
    with pytest.raises(TokenizerRoundTripError):
        TokenPrefixAutomaton(LossyTokenizer(), ["UPPER"])
    with pytest.raises(TokenizerCollisionError):
        TokenPrefixAutomaton(CollisionTokenizer(), ["a", "b"], strict_roundtrip=False)


@pytest.mark.skipif(importlib.util.find_spec("tiktoken") is None, reason="tiktoken not installed")
def test_tiktoken_adapter_exact_shared_prefixes():
    tokenizer = TiktokenAdapter("cl100k_base")
    automaton = TokenPrefixAutomaton(tokenizer, ["a", "ab", '"snowman value"'])
    for literal in automaton.literal_ids:
        state = automaton.initial_state()
        for token_id in tokenizer.encode(literal):
            state = automaton.update(state, token_id)
        assert state.complete_value == literal


@pytest.mark.skipif(importlib.util.find_spec("tokenizers") is None, reason="tokenizers not installed")
def test_hf_tokenizer_adapter_exact_wordpiece():
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "##s": 3, '"value"': 4}
    raw = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = WhitespaceSplit()
    raw.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    tokenizer = HFTokenizerAdapter(raw, name="hf/wordpiece-test")
    automaton = TokenPrefixAutomaton(tokenizer, ["hello worlds", '"value"'])
    for literal in automaton.literal_ids:
        state = automaton.initial_state()
        for token_id in tokenizer.encode(literal):
            state = automaton.update(state, token_id)
        assert state.complete_value == literal
