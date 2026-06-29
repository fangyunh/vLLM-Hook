import os
import math
import pickle
import torch
from typing import TYPE_CHECKING, Any, Dict, List
import zstandard as zstd
from vllm.forward_context import get_forward_context
from vllm.distributed import parallel_state as ps

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.workers._common import (
    background_save_loop,
    clear_states_for_req,
    get_query_metadata,
    init_async_save_thread,
    iter_matched_modules,
    iter_matching_req_ids,
    match_attn,
    save_pt_atomic,
    save_safetensors_atomic,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def key_cache_from_layer_kv(kv_cache):
    """Return the KEY cache as ``[num_blocks, block_size, num_kv_heads, head_size]``.

    The KV cache layout differs across vLLM versions / backends, and getting it
    wrong makes ``key_cache[block_ids]`` gather along the WRONG dimension — an
    out-of-bounds index into a size-2 (key/value) axis that triggers an
    unrecoverable device-side assert. vLLM v1 stores ONE tensor shaped
    ``[num_blocks, 2, block_size, num_kv_heads, head_size]`` and splits it with
    ``kv_cache.unbind(1)`` (confirmed in the TRITON/FLASH backends), so the key
    cache is ``kv_cache[:, 0]`` — NOT ``kv_cache[0]`` (which is block 0).

    Handles, in order:
      * a per-virtual-engine list/tuple wrapping ONE kv tensor -> unwrap it;
      * a (key, value) pair already split -> take element 0;
      * tensor ``[num_blocks, 2, block_size, H, D]`` (current vLLM) -> ``[:, 0]``;
      * tensor ``[2, num_blocks, block_size, H, D]`` (legacy) -> ``[0]``;
      * an already-key-only 4-D tensor -> as-is.
    """
    kv = kv_cache
    # Unwrap a single-element per-virtual-engine list -> the kv tensor.
    if isinstance(kv, (list, tuple)) and len(kv) == 1 and hasattr(kv[0], "ndim"):
        kv = kv[0]
    # Already-split (key, value) pair.
    if isinstance(kv, (list, tuple)) and len(kv) == 2 and hasattr(kv[0], "ndim"):
        return kv[0]
    if not hasattr(kv, "ndim"):
        return kv
    if kv.ndim == 5:
        if kv.shape[1] == 2:      # [num_blocks, 2, block_size, H, D] — vLLM unbind(1)
            return kv[:, 0]
        if kv.shape[0] == 2:      # [2, num_blocks, block_size, H, D] — legacy
            return kv[0]
    return kv                     # 4-D: already key-only


def _read_cached_keys(
    module_name,
    attn_metadata,
    req_idx: int,
    num_cached: int,
    total_len: int,
):
    """Read cached prefix keys from vLLM's paged KV cache.

    When prefix caching is active, the hook only fires for non-cached tokens.
    This function reconstructs the missing prefix keys by reading directly from
    vLLM's KV cache blocks, keyed by the block_table entry for this request.

    Returns a tensor of shape (num_cached, num_kv_heads * head_size) on the
    same device as the KV cache, or None on any error (caller falls back to
    new-tokens-only capture).
    """
    try:
        ctx = get_forward_context()
        # kv_cache is bound to the vLLM Attention wrapper in the forward context,
        # not to the PyTorch module. Access via no_compile_layers[layer_name].kv_cache.
        kv_cache = ctx.no_compile_layers[module_name].kv_cache
        # Correct key cache: [num_blocks, block_size, num_kv_heads, head_size]
        # (see key_cache_from_layer_kv for the layout lesson — using kv_cache[0]
        # here gathers a size-2 axis with real block ids -> device-side assert).
        key_cache = key_cache_from_layer_kv(kv_cache)

        num_blocks   = key_cache.shape[0]
        block_size   = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size    = key_cache.shape[3]

        # block_table: [batch_size, max_blocks_per_seq]
        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]  # [num_blocks_needed]

        # Bounds guard: an out-of-range block id makes the gather below trigger an
        # UNRECOVERABLE device-side assert (kills the whole engine). If anything is
        # off (wrong layout, stale block_table), skip prefix-K rather than crash —
        # the caller then keeps the new-tokens-only keys, a safe degradation.
        if block_ids.numel() == 0 or int(block_ids.max()) >= num_blocks \
                or int(block_ids.min()) < 0:
            return None

        # Gather and flatten: [num_blocks_needed * block_size, kv_hidden]
        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)

        # Trim to exact cached token count (last block may be partially filled)
        return prefix_keys[:num_cached].detach()
    except Exception:
        return None


def _k_all_cpu_list(entry: dict) -> list:
    """Return the per-step growing-prefix ``k_all`` as CPU tensors.

    Graph buffer-mode egress (W3 incremental) stores only each step's NEW key rows in
    ``entry["k_all"]`` plus the per-step prefix length in ``entry["k_prefix_ends"]``, so
    the per-step GPU clone is O(1) instead of the O(seq_len) full-prefix clone. Here we
    rebuild the growing prefixes (``full[:L]`` for each recorded ``L``) so the captured
    artifact is byte-identical to the eager path, which stores the full prefixes directly
    (no ``k_prefix_ends`` -> passed through unchanged). ``.cpu()`` first so a drained
    (pinned) + GPU mix concatenates on one device; the per-step slices are CPU views, and
    the downstream pad_sequence/stack copies them exactly as before.
    """
    prefix_ends = entry.get("k_prefix_ends")
    if not prefix_ends:
        return [t.cpu() for t in entry["k_all"]]
    full = torch.cat([t.cpu() for t in entry["k_all"]], dim=0)
    return [full[:int(L)] for L in prefix_ends]


def compute_head_score(q_flat: torch.Tensor, k_flat: torch.Tensor, head: int,
                       conf: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One head's causal attention score: ``softmax((q_h·k_hᵀ)·mult + causal) -> [S_q, S_k]``.

    The v0.6.0 GPU-side score capture: instead of cloning q/k (all heads) and recomputing
    in the analyzer, compute the score for ONE head here and flush only that. Matches the
    analyzer math exactly (``AttntrackerAnalyzer.compute_attention_from_qk`` /
    ``CorerAnalyzer.get_attn_all``) so the captured score == analyzer recompute for the same
    head:

      * ``q_flat`` [S_q, H_q·d] — this pass's post-RoPE queries (all_tokens slice).
      * ``k_flat`` [S_k, H_kv·d] — the FULL key history (prefix reconstructed upstream); S_k ≥ S_q.
      * ``head``   q-head index in [0, H_q); its KV head under GQA is ``head // (H_q//H_kv)``.
      * causal: query row r (absolute pos ``offset+r``, ``offset = S_k - S_q`` = prefix-cached
        count) attends keys ``[0, offset+r]`` — ``tril(diagonal=offset)`` == the analyzer's
        ``triu(diagonal=-(S_k-S_q))``.

    Matmul/softmax in fp32 (accuracy), stored in ``dtype`` (default fp16: [0,1] values fit
    fp16's mantissa better than bf16, same 2 bytes). ``conf`` is the worker ``_conf``.
    """
    H_q = int(conf["num_attention_heads"])
    H_kv = int(conf["num_key_value_heads"])
    d = int(conf["head_dim"])
    mult = float(conf["attention_multiplier"])
    S_q = q_flat.shape[0]
    S_k = k_flat.shape[0]
    g = max(1, H_q // H_kv)
    h = int(head) % H_q
    q_h = q_flat.view(S_q, H_q, d)[:, h, :].float()            # [S_q, d]
    k_h = k_flat.view(S_k, H_kv, d)[:, h // g, :].float()      # [S_k, d]
    s = torch.matmul(q_h, k_h.transpose(0, 1)) * mult          # [S_q, S_k] fp32
    offset = S_k - S_q
    if S_q > 1 or offset < 0:
        # Mask non-causal entries. A single query row (decode / last token) at offset==S_k-1
        # attends to every key, so the mask is a no-op and is skipped for speed.
        mask = torch.ones(S_q, S_k, dtype=torch.bool, device=s.device).tril(diagonal=offset)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1).to(dtype)


def _scores_from_qk_entry(entry: dict, conf: dict, dtype: torch.dtype = torch.float16) -> list:
    """Compute per-pass head scores from an accumulated Q/K entry (buffer-mode score).

    Buffer-mode egress stages Q/K exactly like QK capture (so it reuses the routing +
    growing-``k_all`` reconstruction + prefix-K), then marks the entry ``capture="score"``.
    Here, at retrieval, we recompute the one-head score ON GPU from those staged tensors and
    flush only the score — the analyzer's work, moved to the worker. Mirrors
    ``_k_all_cpu_list``'s prefix reconstruction (``full[:L]`` per recorded ``k_prefix_ends``)
    so each pass's keys match the eager path; aligns 1:1 with the per-pass ``q`` list (both
    appended on ``emit_q`` steps). Returns CPU score tensors.
    """
    head = int(entry.get("head", 0))
    # .cpu() every staged tensor BEFORE the cat/compute: the streaming drain
    # (graph/drain.py) replaces drained list elements in place with pinned-HOST tensors
    # while post-drain clones stay on GPU, so entry["k_all"]/["q"] can be a CUDA/CPU MIX
    # — torch.cat over mixed devices raises. Same fix _k_all_cpu_list uses; computing the
    # score on host at retrieval is value-identical (buffer mode is not perf-profiled).
    q_list = [t.cpu() for t in entry["q"]]
    prefix_ends = entry.get("k_prefix_ends")
    if prefix_ends:
        full_k = torch.cat([t.cpu() for t in entry["k_all"]], dim=0)
        k_list = [full_k[:int(L)] for L in prefix_ends]
    else:
        k_list = [t.cpu() for t in entry["k_all"]]
    out = []
    for q_p, k_p in zip(q_list, k_list):
        q_p = q_p if q_p.dim() == 2 else q_p.unsqueeze(0)
        out.append(compute_head_score(q_p, k_p, head, conf, dtype))
    return out


class ProbeHookQKWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    vLLM does Worker.__bases__ += (ProbeHookQKWorker,) at runtime,
    so self is the Worker instance. Methods are callable via collective_rpc.
    """

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: "ParallelConfig"

    # Default capture phase — matches the old hooks_on=(True, False) registry entry.
    # Can be overridden per-request via extra_args["hooks_on"].
    _default_hooks_on: str = "prefill"

    # Per-request captured QK states (API serving path):
    # internal_req_id -> {module_name -> {"q": [...], "k_all": [...], "layer_num": int}}
    _captured_states: dict = {}
    _hooks_installed: bool = False

    def install_hooks(self):
        """Install forward hooks on all target attention modules. Idempotent.

        Callable via collective_rpc("install_hooks") — the plugin calls this
        lazily on the first request that sets output_qk in extra_args.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}  # RPC path
        self._disk_states = {}      # disk path: same shape, written via flush_disk()
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Worker-wide fallback when extra_args["hookq_mode"] is missing.
        self.hookq_mode = "all_tokens"

        # v0.6.0 score capture: worker-wide defaults (per-request extra_args override).
        # VLLM_HOOK_QK_SCORE=1 flips the worker to flush per-head attention scores
        # instead of Q/K; VLLM_HOOK_QK_SCORE_HEAD picks the (single) head per layer.
        self._score_mode_default = os.environ.get("VLLM_HOOK_QK_SCORE", "0") == "1"
        self._score_head_default = int(os.environ.get("VLLM_HOOK_QK_SCORE_HEAD", "0"))
        self._score_dtype = torch.float16

        # Background I/O thread (shared by all disk-save requests on this worker).
        init_async_save_thread(self, self._background_save_loop, "vllm-hook-qk-io")

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads"))
        num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
        hidden = int(getattr(text_cfg, "hidden_size"))
        head_dim = hidden // num_h
        attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        # Only TP rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        def qkv_hook(input, module_name, attn_module=None):
            # Fast-path: only rank 0 captures (RPC and disk paths both need it).
            if not self._should_capture:
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)

            # Warmup or non-attention passes: nothing to do
            if metadata is None:
                return
            if torch.cuda.is_current_stream_capturing():
                return None

            query_start_loc, seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return

            bs = len(query_start_loc) - 1
            last_indices = query_start_loc

            layer_num = match_attn(module_name)

            # Per-request capture. Each request in the batch may route to
            # either _captured_states (RPC path) or _disk_states (disk path)
            # based on its extra_args.
            try:
                req_ids = self.model_runner.input_batch.req_ids
            except Exception:
                return

            try:
                num_computed = self.model_runner.input_batch.num_computed_tokens_cpu
                num_prompt   = self.model_runner.input_batch.num_prompt_tokens
            except Exception:
                # fall back to the old behavior and capture every chunk if these attributes aren't available on the running vLLM version
                num_computed = None
                num_prompt = None

            for i in range(bs):
                req_id = req_ids[i]
                req_state = self.model_runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    continue
                extra = req_state.sampling_params.extra_args
                if not extra or extra.get("output_qk") is None:
                    continue

                # output_qk accepts three forms, matching the old layer_to_heads config:
                #   True             -> capture all layers
                #   [layer_ids]      -> capture specific layers
                #   {layer: [heads]} -> capture specific layers (heads used downstream by analyzer)
                # The worker only uses the keys for layer filtering — head info is
                # forwarded to the analyzer by the caller, same as the old env-var flow.
                output_spec = extra.get("output_qk")
                if isinstance(output_spec, dict):
                    layer_set = {int(k) for k in output_spec.keys()}
                    if layer_num not in layer_set:
                        continue
                elif isinstance(output_spec, list):
                    if layer_num not in output_spec:
                        continue

                # hooks_on: "prefill" (default) | "decode" | "both"
                # Uses output_token_ids == [] on the worker-side CachedRequestState
                # to detect the first (prefill) pass. This is robust to prefix
                # caching where query_len < seq_len even on the first pass.
                hooks_on = extra.get("hooks_on", self._default_hooks_on)
                is_prefill = len(req_state.output_token_ids) == 0
                if hooks_on != "both":
                    if hooks_on == "prefill" and not is_prefill:
                        continue
                    if hooks_on == "decode" and is_prefill:
                        continue

                # Per-request mode: extra_args["hookq_mode"] overrides the worker default.
                req_mode = extra.get("hookq_mode", self.hookq_mode)
                # v0.6.0: per-request capture mode — "qk" (default) clones Q/K; "score"
                # computes one head's attention score on-GPU and flushes only that.
                cap_mode = extra.get("qk_capture",
                                     "score" if self._score_mode_default else "qk")

                start = int(last_indices[i].item())
                end = int(last_indices[i + 1].item())

                # With chunked-prefill, in last_token mode, only capture on the final chunk of the prefill
                # i.e., when computed-after-step reaches num_prompt_tokens. Score mode is
                # all_tokens (every query row is part of the [S_q,S_k] matrix), so it never
                # skips a chunk.
                if (cap_mode != "score" and is_prefill and req_mode == "last_token" and num_computed is not None and num_prompt is not None):
                    chunk_len = end - start
                    if int(num_computed[i]) + chunk_len < int(num_prompt[i]):
                        # Mid-prefill chunk doesn't need capture
                        continue

                # ---- v0.6.0 score mode: flush one head's [S_q, S_k] score, not Q/K ----
                if cap_mode == "score":
                    score_head = int(extra.get("score_head", self._score_head_default))
                    # q/k are views consumed immediately by the matmul — no clone needed.
                    q_view = input[0][start:end, :].detach()
                    k_view = input[1][start:end, :].detach()
                    # Reconstruct the prefix-cache-trimmed keys so k is the FULL history
                    # (same path as QK mode below); the score needs every key.
                    if seq_lens is not None and attn_module is not None:
                        try:
                            total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], 'item') else int(seq_lens[i])
                            num_cached = total_len - (end - start)
                            if num_cached > 0:
                                PROF.incr("kv.prefix_recon")
                                with PROF.timed("kv.prefix_recon"):
                                    prefix_k = _read_cached_keys(module_name, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
                                if prefix_k is not None:
                                    k_view = torch.cat([prefix_k.to(k_view.device, dtype=k_view.dtype), k_view], dim=0)
                        except Exception:
                            PROF.incr("kv.prefix_recon.errors")
                    score = compute_head_score(q_view, k_view, score_head, self._conf, self._score_dtype)
                    PROF.incr("hook.fire.qk")
                    PROF.gauge("captured.bytes.qk", score.numel() * score.element_size())
                    bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                    if req_id not in bucket:
                        bucket[req_id] = {}
                    layer_states = bucket[req_id]
                    if module_name not in layer_states:
                        layer_states[module_name] = {"scores": [], "head": score_head, "layer_num": layer_num, "hookq_mode": "all_tokens", "capture": "score"}
                    layer_states[module_name]["scores"].append(score)
                    continue

                # Accumulate GPU tensors — clone() copies data immediately so we
                # own the buffer; .cpu() is deferred to retrieval/flush.
                if req_mode == "all_tokens":
                    q_tok = input[0][start:end, :].detach().clone()
                else:
                    q_tok = input[0][end - 1, :].detach().clone()
                k_tok = input[1][start:end, :].detach().clone()

                # Reconstruct full k_all when prefix caching is active.
                # seq_lens[i] = total sequence length (cached + new tokens).
                # query_len = new tokens only (what the hook captured above).
                # If num_cached > 0, read the missing prefix keys directly from
                # vLLM's paged KV cache and prepend them to k_tok.
                if seq_lens is not None and attn_module is not None:
                    try:
                        total_len = int(seq_lens[i].item()) if hasattr(seq_lens[i], 'item') else int(seq_lens[i])
                        query_len = end - start
                        num_cached = total_len - query_len
                        if num_cached > 0:
                            PROF.incr("kv.prefix_recon")
                            with PROF.timed("kv.prefix_recon"):
                                prefix_k = _read_cached_keys(module_name, metadata if not isinstance(metadata, dict) else next(iter(metadata.values())), i, num_cached, total_len)
                            if prefix_k is not None:
                                k_tok = torch.cat([prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok], dim=0)
                    except Exception:
                        PROF.incr("kv.prefix_recon.errors")

                PROF.incr("hook.fire.qk")
                PROF.gauge("captured.bytes.qk",
                           q_tok.numel() * q_tok.element_size()
                           + k_tok.numel() * k_tok.element_size())

                # Route to disk or RPC bucket based on save_to_disk flag.
                bucket = self._disk_states if extra.get("save_to_disk") else self._captured_states
                if req_id not in bucket:
                    bucket[req_id] = {}
                layer_states = bucket[req_id]
                if module_name not in layer_states:
                    layer_states[module_name] = {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": req_mode}
                layer_states[module_name]["q"].append(q_tok)
                layer_states[module_name]["k_all"].append(k_tok)

        # Hook every attention module. Per-request layer filtering via
        # extra_args['output_qk'] happens inside the hook closure.
        self._hooks = []
        matched = []
        for name, module, _ in iter_matched_modules(model, match_attn):
            hook = module.register_forward_hook(
                lambda _m, i, _o, n=name: qkv_hook(i, n, _m)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")

    # ------------------------------------------------------------------
    # v0.3.0 CUDA-graph capture install (graph mode only)
    # ------------------------------------------------------------------

    def graph_install(self):
        """Install the CUDA-graph QK capture path (static buffers + wrap).

        Thin delegating entry called by the Worker.load_model monkey-patch
        (graph/install.py:patch_worker_load_model) AFTER the model is built but
        BEFORE warm-up/compile/capture. It is a strict no-op for the eager
        v0.2.0 path: the load_model patch only reaches here when graph mode is
        armed (VLLM_HOOK_ALLOW_CUDAGRAPH==1) AND this worker is the QK worker.

        The heavy lifting (per-layer hosts, class-level Attention.forward wrap,
        execute_model routing+egress wrapper) lives in graph/install.py. This
        method only:
          * ensures the egress buckets the eager path also uses exist as
            instance dicts (so the graph egress can populate them and
            get_captured_states / flush_disk / _save_safetensors consume them
            UNCHANGED — REUSE CONTRACT), then
          * delegates to graph.install.

        Idempotent: graph.install's installers are themselves guarded, and the
        bucket init below only seeds dicts that are missing.
        """
        # Egress buckets — same instance dicts the eager path (install_hooks)
        # and the RPC retrieval methods (get_captured_states / flush_disk)
        # operate on. The class-level defaults (_captured_states = {}) are
        # SHARED across instances, so we reset to per-instance dicts here,
        # exactly as install_hooks() does, before any capture can fire.
        if not getattr(self, "_captured_states", None):
            self._captured_states = {}  # RPC path
        if not getattr(self, "_disk_states", None):
            self._disk_states = {}      # disk path

        # graph.install builds the hosts, populates self._conf / self.hookq_mode /
        # self._should_capture, wires the wrap, and installs the execute_model
        # wrapper. Import lazily so the eager path never imports the graph stack.
        from vllm_hook_plugins.graph.install import (
            install_execute_model_wrapper,
            install_qk_hosts,
        )
        install_qk_hosts(self)
        install_execute_model_wrapper(self.model_runner, self)

    # ------------------------------------------------------------------
    # API serving: collective_rpc-callable artifact retrieval
    # ------------------------------------------------------------------

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve and remove captured QK states for a completed request.

        Matches by "{external_req_id}-" prefix because vLLM internally
        transforms the user-provided request_id into "{request_id}-{random_suffix}".

        CPU transfer happens here (once per request, not per hook).
        Returns zstd-compressed pickle, or None if nothing was captured.
        """
        from vllm_hook_plugins.graph.drain import drain_barrier
        drain_barrier(self)  # wait for any pending W4 drain before reading buckets
        dm = getattr(self, "_capture_drain", None)
        if dm is not None:
            dm.forget(external_req_id)  # prune a throttled (never-bucketed) id for this req
        for req_id in iter_matching_req_ids(self._captured_states, external_req_id):
            layer_dict = self._captured_states.pop(req_id)
            # Stage 3: release the request's resident bytes (counted at egress) on pop.
            if dm is not None:
                dm.on_pop(req_id, layer_dict)
            cpu_dict = {}
            with PROF.timed("worker.cpu_transfer.qk"):
                for mod_name, entry in layer_dict.items():
                    from torch.nn.utils.rnn import pad_sequence
                    # v0.6.0 score entry. Two producers: eager/op store "scores" directly;
                    # buffer mode stages Q/K + marks capture=="score", so recompute here.
                    if "scores" in entry or entry.get("capture") == "score":
                        scores = entry["scores"] if "scores" in entry \
                            else _scores_from_qk_entry(entry, self._conf,
                                                       getattr(self, "_score_dtype", torch.float16))
                        cpu_dict[mod_name] = {
                            "scores": [t.cpu() for t in scores],
                            "head": entry.get("head", 0),
                            "layer_num": entry["layer_num"],
                            "hookq_mode": entry.get("hookq_mode", "all_tokens"),
                            "capture": "score",
                        }
                        continue
                    mode = entry.get("hookq_mode", self.hookq_mode)
                    if mode == "all_tokens":
                        q_stacked = pad_sequence([t.cpu() for t in entry["q"]], batch_first=True)
                    else:
                        q_stacked = torch.stack([t.cpu() for t in entry["q"]])
                    k_stacked = pad_sequence(_k_all_cpu_list(entry), batch_first=True)
                    cpu_dict[mod_name] = {"q": q_stacked, "k_all": k_stacked, "layer_num": entry["layer_num"], "hookq_mode": mode}
            payload = {"qk_cache": cpu_dict, "config": self._conf}
            with PROF.timed("worker.compress.qk"):
                raw = pickle.dumps(payload)
                PROF.gauge("worker.raw_bytes.qk", len(raw))
                compressed = _ZSTD_COMPRESSOR.compress(raw)
            PROF.gauge("worker.compressed_bytes.qk", len(compressed))
            return compressed
        return None

    def reset_capture_peak_mem(self) -> int:
        """collective_rpc-callable: reset the CUDA peak-allocated high-water mark and return
        current allocated bytes. Read-only test/monitoring introspection (the CB-OOM oracle
        measures peak GPU residency this way; worker-internal CUDA stats have no other
        channel to the driver). Called by string name so no function payload is serialized."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return int(torch.cuda.memory_allocated())
        return 0

    def capture_drain_stats(self) -> dict:
        """collective_rpc-callable: snapshot the drain/throttle manager counters, PROF
        capture counters, and CUDA mem high-water. Read-only introspection for the CB-OOM
        oracle. Called by string name (no cloudpickled function payload)."""
        out = {
            "drain_present": False, "drain_enabled": False, "num_drains": 0,
            "gpu_bytes": 0, "resident_bytes": 0, "ceiling_bytes": 0, "budget_bytes": 0,
            "throttled_set": 0, "max_inflight": 0, "admitted_set": 0,
            "prof_throttled": 0, "prof_hookfire": 0,
            "max_mem_allocated": 0, "mem_allocated": 0, "total_gpu": 0,
        }
        dm = getattr(self, "_capture_drain", None)
        if dm is not None:
            out["drain_present"] = True
            out["drain_enabled"] = bool(getattr(dm, "enabled", False))
            out["num_drains"] = int(getattr(dm, "num_drains", 0))
            out["gpu_bytes"] = int(getattr(dm, "gpu_bytes", 0))
            out["resident_bytes"] = int(getattr(dm, "resident_bytes", 0))
            out["ceiling_bytes"] = int(getattr(dm, "ceiling_bytes", 0))
            out["budget_bytes"] = int(getattr(dm, "budget_bytes", 0))
            out["throttled_set"] = int(len(getattr(dm, "_throttled", ()) or ()))
            out["max_inflight"] = int(getattr(dm, "max_inflight", 0))
            out["admitted_set"] = int(len(getattr(dm, "_admitted", ()) or ()))
        try:
            from vllm_hook_plugins._profiler import PROF
            counters = PROF.snapshot().get("counters", {})
            out["prof_throttled"] = int(counters.get("capture.throttled", 0))
            out["prof_hookfire"] = int(counters.get("hook.fire.qk", 0))
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            out["max_mem_allocated"] = int(torch.cuda.max_memory_allocated())
            out["mem_allocated"] = int(torch.cuda.memory_allocated())
            try:
                _free, _total = torch.cuda.mem_get_info()
                out["total_gpu"] = int(_total)
            except Exception:  # noqa: BLE001
                pass
        return out

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured states without returning them (cleanup on abort/disconnect)."""
        dm = getattr(self, "_capture_drain", None)
        if dm is None:
            clear_states_for_req(self._captured_states, external_req_id)
            clear_states_for_req(self._disk_states, external_req_id)
            return
        # Release resident bytes before dropping each bucket so aborts don't leak the
        # ceiling counter (which would eventually wedge admission). BOTH buckets: a
        # disk-mode abort must release too, else resident_bytes ratchets up monotonically.
        for bucket in (self._captured_states, self._disk_states):
            for req_id in iter_matching_req_ids(bucket, external_req_id):
                dm.on_pop(req_id, bucket.pop(req_id))
        dm.forget(external_req_id)  # prune a throttled (never-bucketed) id for this req

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        """Write captured Q/K for all requests in the batch to one artifact.

        Accepts a list of external_req_ids so all requests sharing a run_id
        are merged into one cpu_cache before writing — matching the old
        execute_model() behavior where the full batch was saved atomically.

        Returns True if any artifacts were written, False if nothing captured.
        """
        from vllm_hook_plugins.graph.drain import drain_barrier
        drain_barrier(self)  # wait for any pending W4 drain before reading buckets
        dm = getattr(self, "_capture_drain", None)
        if dm is not None:
            for _erid in external_req_ids:
                dm.forget(_erid)  # prune throttled (never-bucketed) ids for this batch
        cpu_cache: dict = {"config": self._conf, "qk_cache": {}}
        found_any = False

        with PROF.timed("worker.cpu_transfer.qk"):
            for external_req_id in external_req_ids:
                for req_id in iter_matching_req_ids(self._disk_states, external_req_id):
                    layer_dict = self._disk_states.pop(req_id)
                    # Stage 3: release the request's resident bytes on pop.
                    if dm is not None:
                        dm.on_pop(req_id, layer_dict)
                    if not layer_dict:
                        continue
                    found_any = True
                    for mod_name, entry in layer_dict.items():
                        # v0.6.0 score entry (eager/op store "scores"; buffer stages Q/K
                        # + marks capture=="score" -> recompute). Merge per-pass lists.
                        if "scores" in entry or entry.get("capture") == "score":
                            scores = entry["scores"] if "scores" in entry \
                                else _scores_from_qk_entry(entry, self._conf,
                                                           getattr(self, "_score_dtype", torch.float16))
                            cpu_entry = {
                                "scores": [t.cpu() for t in scores],
                                "head": entry.get("head", 0),
                                "layer_num": entry["layer_num"],
                                "hookq_mode": entry.get("hookq_mode", "all_tokens"),
                                "capture": "score",
                            }
                            if mod_name in cpu_cache["qk_cache"]:
                                if "scores" not in cpu_cache["qk_cache"][mod_name]:
                                    raise ValueError(
                                        f"mixed capture modes (score + qk) for module "
                                        f"{mod_name} in one batch are not supported")
                                cpu_cache["qk_cache"][mod_name]["scores"].extend(cpu_entry["scores"])
                            else:
                                cpu_cache["qk_cache"][mod_name] = cpu_entry
                            continue
                        cpu_entry = {
                            "q": [t.cpu() for t in entry["q"]],
                            "k_all": _k_all_cpu_list(entry),
                            "layer_num": entry["layer_num"],
                            "hookq_mode": entry.get("hookq_mode", self.hookq_mode),
                        }
                        if mod_name in cpu_cache["qk_cache"]:
                            if "q" not in cpu_cache["qk_cache"][mod_name]:
                                raise ValueError(
                                    f"mixed capture modes (qk + score) for module "
                                    f"{mod_name} in one batch are not supported")
                            cpu_cache["qk_cache"][mod_name]["q"].extend(cpu_entry["q"])
                            cpu_cache["qk_cache"][mod_name]["k_all"].extend(cpu_entry["k_all"])
                        else:
                            cpu_cache["qk_cache"][mod_name] = cpu_entry

        if not found_any:
            return False

        tp_rank = int(ps.get_tensor_model_parallel_rank())
        run_dir = os.path.join(hook_dir, run_id, f"tp_rank_{tp_rank}")
        os.makedirs(run_dir, exist_ok=True)

        # v0.6.0 score artifacts are ragged [S_q,S_k] lists — the fixed-shape safetensors
        # format doesn't fit them, so a score cache always saves as .pt (pickle handles
        # the lists). The RPC/probes path is unaffected.
        has_scores = any("scores" in e for e in cpu_cache["qk_cache"].values())

        if os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
            PROF.gauge("async.queue_depth_before_put", self._save_queue.qsize())
            with PROF.timed("worker.queue_put"):
                self._save_queue.put((run_id, cpu_cache, run_dir))
        elif os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1" and not has_scores:
            self._save_safetensors(cpu_cache, run_dir)
        else:
            save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
        return found_any

    def _save_safetensors(self, cpu_cache: dict, run_dir: str):
        # v0.6.0: score caches are ragged [S_q,S_k] lists that don't fit the fixed-shape
        # safetensors format — fall back to .pt (pickle). Also covers the async
        # background-save path, which calls this method directly.
        if any("scores" in e or e.get("capture") == "score"
               for e in cpu_cache["qk_cache"].values()):
            save_pt_atomic(cpu_cache, os.path.join(run_dir, "qk.pt"))
            return
        # safetensors stores fixed-shape tensors only, so per-request variable-length
        # q/k_all are padded into a single batched tensor and unpadded on load using
        # the seq_lens stored in the JSON sidecar. q and k_all may have different
        # lengths when prefix caching is active (q = new tokens, k_all = new+cached),
        # so we track q_seq_lens and k_seq_lens separately.
        from torch.nn.utils.rnn import pad_sequence

        flat_dict: dict = {}
        layer_order: list = []
        batch_size = 0
        q_seq_lens: list = []  # only populated for all_tokens mode (q is 2D then)
        k_seq_lens: list = []  # always populated (k_all is always 2D)

        for mod_name, entry in cpu_cache["qk_cache"].items():
            safe_key_q = mod_name.replace(".", "__") + "__q"
            safe_key_k = mod_name.replace(".", "__") + "__k"
            mode = entry.get("hookq_mode", self.hookq_mode)

            if mode == "all_tokens":
                flat_dict[safe_key_q] = pad_sequence(entry['q'], batch_first=True)
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
                if not q_seq_lens:
                    q_seq_lens = [t.shape[0] for t in entry['q']]
            else:
                # last_token: q is 1D (head_dim,) per request — stack instead of pad.
                flat_dict[safe_key_q] = torch.stack(entry['q'])
                flat_dict[safe_key_k] = pad_sequence(entry['k_all'], batch_first=True)
            if not k_seq_lens:
                k_seq_lens = [t.shape[0] for t in entry['k_all']]

            batch_size = flat_dict[safe_key_q].shape[0]
            layer_order.append({
                "key_q": safe_key_q,
                "key_k": safe_key_k,
                "module_name": mod_name,
                "layer_num": entry["layer_num"],
                "hookq_mode": mode,
            })

        meta = {
            "config": cpu_cache["config"],
            "layer_order": layer_order,
            "batch_size": batch_size,
            "hookq_mode": self.hookq_mode,
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
        }
        if q_seq_lens:
            meta["q_seq_lens"] = q_seq_lens
        if k_seq_lens:
            meta["k_seq_lens"] = k_seq_lens
        save_safetensors_atomic(flat_dict, meta, run_dir, "qk")

    def _background_save_loop(self):
        background_save_loop(self, "qk.pt")

