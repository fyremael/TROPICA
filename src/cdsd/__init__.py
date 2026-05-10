"""Control-Delta Support Decoding."""

from .decoder import SupportDecoder, DecodeTrace
from .masks import SupportMask, intersect_masks, masked_softmax_sample
from .model_integration import CallableLogitProvider, DecodeEvent, HostileLogitProvider, LogitProvider, MaskedDecodeResult, ScriptedLogitProvider, StructuredOutputDecodeError, StructuredOutputDecoder
from .structured_output import StructuredOutputCompiler, ToolCallSpec
from .tokenizer_compiler import HFTokenizerAdapter, TiktokenAdapter, TokenPrefixAutomaton, TokenizerPrefixError

__all__ = [
    "CallableLogitProvider",
    "DecodeEvent",
    "DecodeTrace",
    "HFTokenizerAdapter",
    "HostileLogitProvider",
    "LogitProvider",
    "MaskedDecodeResult",
    "ScriptedLogitProvider",
    "StructuredOutputCompiler",
    "StructuredOutputDecodeError",
    "StructuredOutputDecoder",
    "SupportDecoder",
    "SupportMask",
    "TiktokenAdapter",
    "TokenPrefixAutomaton",
    "TokenizerPrefixError",
    "ToolCallSpec",
    "intersect_masks",
    "masked_softmax_sample",
]
