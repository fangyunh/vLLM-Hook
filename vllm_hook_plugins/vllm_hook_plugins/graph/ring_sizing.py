"""Ring sizing.

The shared GPU capture ring is a FIXED size, resolved by ``resolve_ring_bytes_auto``:
``VLLM_HOOK_RING_GPU_BYTES`` (bytes) if set, else the default 4 GiB. The install gate is a *fit
check* — the fixed ring must fit the free margin left by ``gpu_memory_utilization``.

A fixed ring makes the free margin a known constant — which is what the auto
``max_num_batched_tokens`` derivation (compute_safe_max_batched_tokens) sizes its per-step
transient budget against.

Fails loud (never a silent degrade)."""
import os
from typing import Optional, Tuple

# Default fixed GPU capture-ring size: 4 GiB.
DEFAULT_RING_GPU_BYTES = 4 * (1 << 30)

# Auto-cap defaults. SAFETY covers the buffering copies (egress gather + pinned staging +
# not-yet-drained clones) plus any activation the startup profiling under-counts because it never
# runs capture; HEADROOM is fragmentation slack. Both overridable by env for tuning.
DEFAULT_AUTOCAP_SAFETY = 3
DEFAULT_AUTOCAP_HEADROOM_BYTES = 1 << 30   # 1 GiB

_TRUE = ("1", "true", "on", "yes", "auto")
_FALSE = ("", "0", "false", "off", "no")


def resolve_ring_bytes_fixed(total_gpu_bytes: int, fixed_bytes: int, gpu_mem_util: float) -> int:
    """Fixed-size ring path: return ``fixed_bytes`` verbatim after a fit check. The ring must fit the
    free margin ``(1 - gpu_mem_util) x total`` left after vLLM's KV commitment; otherwise raise
    (spec: fail loud, never silent degrade)."""
    fixed_bytes = int(fixed_bytes)
    total_gpu_bytes = int(total_gpu_bytes)
    free_margin = (1.0 - gpu_mem_util) * total_gpu_bytes
    if fixed_bytes > free_margin:
        raise ValueError(
            f"ring {fixed_bytes/(1<<30):.2f} GiB + gpu_memory_utilization={gpu_mem_util} leaves no "
            f"room (free margin {free_margin/(1<<30):.2f} GiB): lower gpu_memory_utilization or "
            f"VLLM_HOOK_RING_GPU_BYTES")
    return fixed_bytes


def resolve_ring_bytes_auto(total_gpu_bytes: int, gpu_mem_util: float) -> int:
    """Resolve the fixed ring byte budget from the environment: ``VLLM_HOOK_RING_GPU_BYTES`` if set,
    else ``DEFAULT_RING_GPU_BYTES`` (4 GiB). Applies the fit-check gate."""
    raw_fixed = os.environ.get("VLLM_HOOK_RING_GPU_BYTES")
    if raw_fixed is not None and raw_fixed.strip() != "":
        return resolve_ring_bytes_fixed(total_gpu_bytes, int(raw_fixed), gpu_mem_util)
    return resolve_ring_bytes_fixed(total_gpu_bytes, DEFAULT_RING_GPU_BYTES, gpu_mem_util)


# ---------------------------------------------------------------------------
# Auto-derive max_num_batched_tokens (the OOM fix).
#
# Under offline lock-step at high batch, a large synchronized prefill's per-step transient (egress
# gather + pinned staging + undrained clones) scales with tokens-processed-per-step, which is ~batch.
# vLLM's startup profiling never runs capture, so this transient is un-budgeted and overflows the thin
# free margin. Capping the scheduler's per-step token budget chunks the prefill and bounds the
# transient INDEPENDENT of batch (chunked prefill is already proven capture-chunk-invariant). These
# pure helpers compute the safe cap; _hook_plugin applies it MIN-ONLY at the create_engine_config seam.
# ---------------------------------------------------------------------------

def per_layer_token_bytes_hs(hidden_size: int, dtype_size: int) -> int:
    """HS: one captured token costs one residual-stream row per layer."""
    return int(hidden_size) * int(dtype_size)


def per_layer_token_bytes_qk(n_q_heads: int, n_kv_heads: int, head_dim: int, dtype_size: int) -> int:
    """QK: the capture buffers hold ALL q and k heads (install.py sizes q_dim/k_dim from the full
    head counts), so one captured token costs (H_q + H_kv) * head_dim elements per layer."""
    return (int(n_q_heads) + int(n_kv_heads)) * int(head_dim) * int(dtype_size)


def compute_safe_max_batched_tokens(
    total_gpu_bytes: int,
    gpu_mem_util: float,
    ring_gpu_bytes: int,
    n_layers_captured: int,
    per_layer_token_bytes: int,
    safety: int = DEFAULT_AUTOCAP_SAFETY,
    headroom_bytes: int = DEFAULT_AUTOCAP_HEADROOM_BYTES,
) -> Optional[int]:
    """Largest per-step token budget whose worst-case (all-token, all-layer) capture transient fits
    the free GPU margin left after ``gpu_memory_utilization`` + the fixed ring + a fragmentation
    headroom.

    ``free_margin = (1 - util) * total - ring - headroom``; ``bytes_per_token = n_layers * per_layer``;
    ``cap = floor(free_margin / (bytes_per_token * safety))``. Returns None (caller makes NO change)
    when there is no usable margin or the inputs are degenerate — never raises (min-only contract)."""
    total_gpu_bytes = int(total_gpu_bytes)
    # Round the (1-util)*total product to an int before the integer subtraction: `1.0 - 0.9` is
    # 0.0999... in float, which would drop the exact 3 GiB margin to a hair under and cost the
    # canonical cap a spurious -1 (4096 -> 4095). The rounding error is < 1 byte, safety-neutral.
    free_margin = round((1.0 - gpu_mem_util) * total_gpu_bytes) - int(ring_gpu_bytes) - int(headroom_bytes)
    bytes_per_token = int(n_layers_captured) * int(per_layer_token_bytes)
    if free_margin <= 0 or bytes_per_token <= 0 or int(safety) <= 0:
        return None
    cap = int(free_margin // (bytes_per_token * int(safety)))
    return cap if cap >= 1 else None


def apply_min_only(current: Optional[int], safe: Optional[int]) -> Optional[int]:
    """The min-only decision: only ever LOWER ``max_num_batched_tokens``.

    ``current`` is vLLM's resolved value (or None if unresolved); ``safe`` is the derived cap (or
    None). Returns the new (lower) int to set, or None meaning "change nothing" — byte-identical.
    When ``safe >= current`` we leave vLLM's value untouched, which is the whole safety argument:
    the plugin bites only in the regime that would OOM."""
    if safe is None:
        return None
    if current is None:
        return int(safe)
    return int(safe) if int(safe) < int(current) else None


def parse_autocap_setting(raw: Optional[str]) -> Tuple[str, Optional[int]]:
    """Parse the tri-state VLLM_HOOK_RING_MAX_BATCHED_TOKENS knob.

    Returns one of:
      ``("off", None)``      — disabled (unset is OFF: opt-in first, matching RING_PER_REQUEST)
      ``("auto", None)``     — derive the cap automatically
      ``("explicit", int)``  — use this int exactly (still applied min-only)
    Truthy spellings (on/true/yes/1/auto) mean 'derive'; an unparseable value is treated as OFF so a
    typo never silently changes the per-step budget."""
    if raw is None:
        return ("off", None)
    s = raw.strip().lower()
    if s in _FALSE:
        return ("off", None)
    if s in _TRUE:
        return ("auto", None)
    try:
        return ("explicit", int(s))
    except ValueError:
        return ("off", None)
