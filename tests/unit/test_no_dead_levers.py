"""Deleted levers must stay deleted — a re-introduced env read is a regression."""
import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parents[2] / "vllm_hook_plugins" / "vllm_hook_plugins"

DELETED_ENV_NAMES = [
    "VLLM_HOOK_FUSED_EGRESS",
    "VLLM_HOOK_VEC_EGRESS", "VLLM_HOOK_VEC_PIN", "VLLM_HOOK_VEC_PIN_RING",
    "VLLM_HOOK_PINNED_RETRIEVAL",
    "VLLM_HOOK_ASYNC_RETRIEVAL", "VLLM_HOOK_ASYNC_RETRIEVAL_MAX_PENDING",
    "VLLM_HOOK_ASYNC_RETRIEVAL_MAX_POLLS", "VLLM_HOOK_ASYNC_RETRIEVAL_POLL_S",
    "VLLM_HOOK_FLUSH_ASYNC_D2H", "VLLM_HOOK_FLUSH_ASYNC_D2H_MAX_PENDING",
    "VLLM_HOOK_ASYNC_SAVE",
    # Task A6: subsumed by the off-loop ring-drain consumer's backpressure
    # (GpuCaptureRing / OffLoopRingDrain in graph/gpu_capture_ring.py, graph/ring_drain_hs.py,
    # graph/ring_drain_qk.py).
    "VLLM_HOOK_CAPTURE_STREAMING", "VLLM_HOOK_CAPTURE_BACKPRESSURE",
    "VLLM_HOOK_CAPTURE_BACKPRESSURE_FREE_BYTES",
    "VLLM_HOOK_CAPTURE_RESIDENT_CEILING", "VLLM_HOOK_CAPTURE_RESIDENT_CEILING_FRAC",
    "VLLM_HOOK_CAPTURE_MAX_INFLIGHT", "VLLM_HOOK_CAPTURE_GPU_BUDGET",
    "VLLM_HOOK_CAPTURE_DRAIN_BYTES",
    # Diagnostics removed when capture_ring was promoted to main (2026-09-08).
    "VLLM_HOOK_QK_DEBUG", "VLLM_HOOK_QK_DEBUG_MAX",
    "VLLM_HOOK_STEER_NO_OP", "VLLM_HOOK_STEER_NO_ROUTING",
    "VLLM_HOOK_PROBE_FWD_WAIT", "VLLM_HOOK_PROBE_ROUTE_FWD_WAIT",
]

# Source strings whose CLASS/DEF must be gone entirely (the mechanisms they named were deleted,
# not renamed) — checked as literal substrings since these are unambiguous declaration headers.
DELETED_SOURCE_SYMBOLS = [
    "class CaptureDrainManager", "def maybe_drain", "def note_resident",
    "def _resolve_budget", "def _backpressure_check",
    "class _ByteBudgetQueue", "def init_flush_dispatch",
    # 2026-09-08 diagnostics. NOTE: install.py's `def _dbg` (the VLLM_HOOK_QK_DEBUG printer)
    # went with them, but it CANNOT be guarded by name here — graph/ring_drain_hs.py defines a
    # live, unrelated `def _dbg` for VLLM_HOOK_RING_DEBUG, which this substring check would hit.
    # The VLLM_HOOK_QK_DEBUG entry in DELETED_ENV_NAMES covers the reintroduction case.
    "def _probe_fwd_wait", "def _probe_route_fwd_wait",
]


def _all_source():
    return "\n".join(p.read_text() for p in PKG.rglob("*.py"))


def _identifier_present(name: str, src: str) -> bool:
    """Whole-identifier match: True iff ``name`` appears NOT as a substring of a longer
    identifier. Plain ``name in src`` would false-positive on
    VLLM_HOOK_CAPTURE_BACKPRESSURE_FREE_FRAC (a live, different env var) for the deleted
    VLLM_HOOK_CAPTURE_BACKPRESSURE, since the former starts with the latter."""
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, src) is not None


def test_deleted_levers_are_not_read_anywhere():
    src = _all_source()
    for name in DELETED_ENV_NAMES:
        assert not _identifier_present(name, src), f"{name} was deleted; it must not be read again"


def test_no_module_defines_retrieval_rpcs():
    src = _all_source()
    for name in ("def kick_retrieval", "def try_collect", "def pinned_cpu_list"):
        assert name not in src, f"{name} was deleted with the retrieval movers"


def test_capture_drain_manager_and_dispatch_queue_are_gone():
    """Task A6: the off-loop ring-drain consumer (OffLoopRingDrain, graph/ring_drain_hs.py /
    graph/ring_drain_qk.py) replaced CaptureDrainManager (budget/backpressure/throttle
    methods) and the writer-process dispatch ladder dropped the byte-budget queue in front of
    it. Both mechanisms must be gone from the package source, not merely unused."""
    src = _all_source()
    for symbol in DELETED_SOURCE_SYMBOLS:
        assert symbol not in src, f"{symbol} was deleted; it must not reappear"
