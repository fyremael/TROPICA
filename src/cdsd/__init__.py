"""Control-Delta Support Decoding."""

from .contracts import (
    EmptySupportViolation,
    Generator,
    Guard,
    IllegalSelectionError,
    IllegalTransitionError,
    Planner,
    PlannerOutput,
    Policy,
    StaleStateError,
    SupportContractError,
    UnifiedTraceEvent,
)
from .decoder import SupportDecoder, DecodeTrace
from .masks import SupportMask, intersect_masks, masked_softmax_sample
from .model_integration import (
    CallableLogitProvider,
    DecodeEvent,
    HFLocalLogitProvider,
    HostileLogitProvider,
    LocalModelBridgeError,
    LogitProvider,
    MaskedDecodeResult,
    ScriptedLogitProvider,
    StructuredOutputDecodeError,
    StructuredOutputDecoder,
)
from .structured_output import StructuredOutputCompiler, ToolCallSpec
from .tokenizer_compiler import HFTokenizerAdapter, TiktokenAdapter, TokenPrefixAutomaton, TokenizerPrefixError

__all__ = [
    "CallableLogitProvider",
    "DecodeEvent",
    "DecodeTrace",
    "EmptySupportViolation",
    "Generator",
    "Guard",
    "HFLocalLogitProvider",
    "HFTokenizerAdapter",
    "HostileLogitProvider",
    "IllegalSelectionError",
    "IllegalTransitionError",
    "LocalModelBridgeError",
    "LogitProvider",
    "MaskedDecodeResult",
    "Planner",
    "PlannerOutput",
    "Policy",
    "ScriptedLogitProvider",
    "StructuredOutputCompiler",
    "StructuredOutputDecodeError",
    "StructuredOutputDecoder",
    "StaleStateError",
    "SupportContractError",
    "SupportDecoder",
    "SupportMask",
    "TiktokenAdapter",
    "TokenPrefixAutomaton",
    "TokenizerPrefixError",
    "ToolCallSpec",
    "UnifiedTraceEvent",
    "intersect_masks",
    "masked_softmax_sample",
]
