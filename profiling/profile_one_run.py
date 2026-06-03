"""profile_one_run — run one configured cell and return its metrics.

A cell is fully described by:
  - row:         baseline | plugin_idle | probe_hidden_states | probe_hook_qk | steer_hook_act
  - storage_variant: rpc | disk-pt | disk-pt-async | disk-st | disk-st-async | shm  (ignored for baseline/plugin_idle/steer)
  - mode:        last_token | all_tokens                                            (ignored for steer/baseline)
  - prompt_len:  int (token-exact synthesis)
  - max_tokens:  int (1 = prefill-only)
  - batch_size:  int
  - n_layers:    list[int] | None  (None => all layers)
  - hooks_on:    prefill | decode | both
  - reps:        timed iterations  (warmup runs preceded)
  - warmup:     warmup iterations

Used both directly (single-cell run for debugging) and by ``bench_grid.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure profiling/_common.py is importable and the plugin package is on sys.path
PROFILING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = PROFILING_DIR.parent
# PROJECT_ROOT must be on sys.path so ``from profiling.peers.v010 ...``
# resolves for the v0.1.0 peer rows (which need to be imported as
# fully-qualified package paths to avoid clobbering vllm_hook_plugins).
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))
sys.path.insert(0, str(PROFILING_DIR))

from _common import (  # noqa: E402
    DEFAULT_HOOK_DIR,
    STORAGE_VARIANTS,
    cell_context,
    clean_hook_dir,
    dir_size_kb,
    find_worker_engine_dump,
    flatten_profile_into_row,
    gpu_used_mb,
    mem_snapshot,
    profile_snapshot,
    read_worker_sidecar_profile,
    reset_cuda_peak_memory,
    reset_profiler,
    synth_prompts,
    variant_env,
    wait_for_artifact,
)


# Row name → (worker_name, analyzer_name)
# Rows ending in "_v010" use the vendored v0.1.0 code under
# profiling/peers/v010/ via the V010Engine driver; they share the same
# (worker, analyzer) labels for symmetry with the current rows.
ROW_TO_WORKER = {
    "baseline":                 (None, None),
    "plugin_idle":              (None, None),                 # plugin imported, no worker_extension
    "probe_hidden_states":      ("probe_hidden_states", "hidden_states"),
    "probe_hook_qk":            ("probe_hook_qk",       "attn_tracker"),  # any QK analyzer works
    "steer_hook_act":           ("steer_hook_act",      None),
    # v0.1.0 peer rows — same workers reimplemented with the older
    # file-flag control plane; used for the v0.2.0-vs-v0.1.0 comparison.
    "probe_hidden_states_v010": ("probe_hidden_states", "hidden_states"),
    "probe_hook_qk_v010":       ("probe_hook_qk",       "attn_tracker"),
    "steer_hook_act_v010":      ("steer_hook_act",      None),
}


def _is_v010_row(row: str) -> bool:
    return row.endswith("_v010")


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------


def _write_temp_config(base_cfg_path: Optional[str], *, row: str,
                       layers: Optional[List[int]], mode: str,
                       steer_layer: Optional[int] = None) -> Optional[str]:
    """Write an ephemeral config that overrides layers/mode for this cell.

    Returns the path, or None when no config is needed (baseline / plugin_idle).
    """
    if row in ("baseline", "plugin_idle"):
        return None
    # Strip the _v010 suffix so config synthesis uses the same JSON schema
    # as the v0.2.0 path — only the runtime layer differs.
    base_row = row[:-len("_v010")] if row.endswith("_v010") else row
    if base_cfg_path is None or not os.path.exists(base_cfg_path):
        # Steer needs a config (it specifies the vector); the others can run
        # with a synthesized minimal one when no path is given.
        if base_row == "probe_hidden_states":
            cfg = {"hidden_states": {"layers": layers or [], "mode": mode}}
        elif base_row == "probe_hook_qk":
            cfg = {"hookq": {"hookq_mode": mode},
                   "params": {"important_heads": [[ln, 0] for ln in (layers or [1])]}}
        else:
            return base_cfg_path
    else:
        with open(base_cfg_path) as f:
            cfg = json.load(f)
        if base_row == "probe_hidden_states" and "hidden_states" in cfg:
            cfg["hidden_states"]["layers"] = layers if layers is not None else cfg["hidden_states"].get("layers", [])
            cfg["hidden_states"]["mode"]   = mode
        elif base_row == "probe_hook_qk" and "hookq" in cfg:
            cfg["hookq"]["hookq_mode"] = mode

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="vllm_hook_prof_"
    )
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _build_llm(*, row: str, model: str, dtype: str, hook_dir: str,
               cfg_path: Optional[str], tp_size: int,
               max_model_len: int, gpu_memory_utilization: float,
               enable_prefix_caching: bool, enforce_eager: bool):
    """Build the vLLM engine for this cell. Returns (engine, is_hook_llm)."""
    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32, "auto": "auto"}
    torch_dtype = dtype_map.get(dtype, dtype)

    if row == "baseline":
        from vllm import LLM
        engine = LLM(
            model=model, dtype=torch_dtype,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
        )
        return engine, False

    if row == "plugin_idle":
        # Import the plugin (registers worker_extension default) but don't
        # use the HookLLM wrapper — measures the loaded-but-idle plugin tax.
        import vllm.plugins as vp
        vp.load_general_plugins()
        from vllm import LLM
        engine = LLM(
            model=model, dtype=torch_dtype,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
        )
        return engine, False

    # v0.1.0 peer rows: dispatch to the vendored V010Engine. v0.1.0 has no
    # prefix-cache prefix-recon path, so we force prefix caching off for
    # parity with the original Numerical_Analysis benchmark — even if the
    # caller asked for it on. This is a measurement requirement, not a
    # capability gap to hide.
    if _is_v010_row(row):
        from profiling.peers.v010.driver import build_engine as _build_v010
        engine = _build_v010(
            row,
            model=model,
            hook_dir=hook_dir,
            config_file=cfg_path,
            dtype=torch_dtype,
            tp_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=enforce_eager,
        )
        return engine, True

    # All v0.2.0 hook rows go through HookLLM.
    from vllm_hook_plugins import register_plugins
    from vllm_hook_plugins.hook_llm import HookLLM
    register_plugins()

    worker_name, analyzer_name = ROW_TO_WORKER[row]
    engine = HookLLM(
        model=model,
        worker_name=worker_name,
        analyzer_name=analyzer_name,
        config_file=cfg_path,
        hook_dir=hook_dir,
        dtype=torch_dtype,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=enforce_eager,
        enable_hook=True,
    )
    return engine, True


# ---------------------------------------------------------------------------
# Per-rep run
# ---------------------------------------------------------------------------


def _one_rep(engine, is_hook_llm: bool, *, row: str, prompts: List[str],
             max_tokens: int, hook_dir: str, storage_variant: str,
             hooks_on: str, analyzer_spec: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    if is_hook_llm and row == "probe_hook_qk":
        sp.extra_args = {"hooks_on": hooks_on}
    elif is_hook_llm and row == "probe_hidden_states":
        sp.extra_args = {"hooks_on": hooks_on}

    save_to_disk = (storage_variant.startswith("disk-")
                    or storage_variant == "shm")
    run_id = str(uuid.uuid4())

    # Reset CUDA peak counters + take a baseline snapshot so the post-rep
    # snapshot reflects ONLY this rep's allocations (not the cumulative
    # high-water mark from warmup or prior reps).
    reset_cuda_peak_memory()
    mem_pre = mem_snapshot()

    t0 = time.perf_counter()
    if is_hook_llm:
        outputs = engine.generate(prompts, sp,
                                  save_to_disk=(storage_variant.startswith("disk-")),
                                  run_id=run_id if save_to_disk else None)
    else:
        outputs = engine.generate(prompts, sp)
    t1 = time.perf_counter()

    # Wait for the async artifact (if any).
    use_st = os.environ.get("VLLM_HOOK_USE_SAFETENSORS", "0") == "1"
    if save_to_disk and os.environ.get("VLLM_HOOK_ASYNC_SAVE", "0") == "1":
        wait_for_artifact(hook_dir, run_id, use_safetensors=use_st, timeout_s=30.0)
    t_wait = time.perf_counter()

    # Analyze (when an analyzer is configured).
    if is_hook_llm and getattr(engine, "analyzer", None) is not None and row != "steer_hook_act":
        try:
            if storage_variant.startswith("disk-"):
                _ = engine.analyze(analyzer_spec=analyzer_spec or {"reduce": "none"},
                                   run_id=run_id)
            else:
                probes = getattr(outputs[0], "probes", None)
                if probes is not None:
                    _ = engine.analyze(probes=probes,
                                       analyzer_spec=analyzer_spec or {"reduce": "none"})
        except Exception as exc:
            print(f"[profile_one_run] analyze failed (continuing): {exc}")
    t_anal = time.perf_counter()

    n_prompt_tokens = sum(len(o.prompt_token_ids)        for o in outputs)
    n_gen_tokens    = sum(len(o.outputs[0].token_ids)    for o in outputs)

    artifact_kb = 0.0
    if save_to_disk:
        artifact_kb = dir_size_kb(os.path.join(hook_dir, run_id))

    # Post-rep memory snapshot — peaks include the entire rep (generate +
    # analyze + flush) because we don't reset between phases here.
    mem_post = mem_snapshot()

    return {
        "gen_lat":          t1 - t0,
        "wait_lat":         t_wait - t1,
        "analyze_lat":      t_anal - t_wait,
        "total_lat":        t_anal - t0,
        "n_prompt_tokens":  n_prompt_tokens,
        "n_gen_tokens":     n_gen_tokens,
        "prefill_tok_per_sec": n_prompt_tokens / max(1e-6, (t1 - t0)),
        "decode_tok_per_sec":  n_gen_tokens    / max(1e-6, (t1 - t0)),
        "peak_gpu_mb":      gpu_used_mb(),
        "artifact_kb":      artifact_kb,
        "run_id":           run_id,
        # Advanced per-rep memory metrics (MB).
        "cuda_alloc_mb_pre":      mem_pre["cuda_alloc_mb"],
        "cuda_alloc_mb_post":     mem_post["cuda_alloc_mb"],
        "cuda_peak_alloc_mb":     mem_post["cuda_peak_alloc_mb"],
        "cuda_reserved_mb_post":  mem_post["cuda_reserved_mb"],
        "cuda_peak_reserved_mb":  mem_post["cuda_peak_reserved_mb"],
        # Hook-induced allocation spike: how much above the pre-rep working set
        # this rep climbed. The interesting "what does the hook cost in GPU
        # memory?" number, isolated from KV-cache + weights.
        "cuda_alloc_delta_mb":    max(0.0, mem_post["cuda_peak_alloc_mb"]
                                            - mem_pre["cuda_alloc_mb"]),
        "host_rss_mb":            mem_post["host_rss_mb"],
        "host_rss_delta_mb":      max(0.0, mem_post["host_rss_mb"]
                                            - mem_pre["host_rss_mb"]),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_cell(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one cell to completion. Returns the aggregated row dict.

    ``cfg`` keys (all optional except ``row``):
      row, model, dtype, tp_size, storage_variant, mode, hooks_on,
      prompt_len, max_tokens, batch_size, n_layers, base_config,
      reps, warmup, hook_dir, max_model_len, gpu_memory_utilization,
      enable_prefix_caching, enforce_eager
    """
    row             = cfg["row"]
    model           = cfg.get("model", "Qwen/Qwen2-1.5B-Instruct")
    dtype           = cfg.get("dtype", "float16")
    tp_size         = int(cfg.get("tp_size", 1))
    storage_variant = cfg.get("storage_variant", "rpc")
    mode            = cfg.get("mode", "last_token")
    hooks_on        = cfg.get("hooks_on", "prefill")
    prompt_len      = int(cfg.get("prompt_len", 64))
    max_tokens      = int(cfg.get("max_tokens", 1))
    batch_size      = int(cfg.get("batch_size", 1))
    n_layers        = cfg.get("n_layers", None)
    base_cfg_path   = cfg.get("base_config", None)
    reps            = int(cfg.get("reps", 5))
    warmup          = int(cfg.get("warmup", 2))
    hook_dir        = cfg.get("hook_dir", DEFAULT_HOOK_DIR)
    max_model_len   = int(cfg.get("max_model_len", 2048))
    gpu_util        = float(cfg.get("gpu_memory_utilization", 0.5))
    prefix_cache    = bool(cfg.get("enable_prefix_caching", False))
    enforce_eager   = bool(cfg.get("enforce_eager", True))
    analyzer_spec   = cfg.get("analyzer_spec", None)

    # Apply storage env (except for baseline/plugin_idle which don't read it).
    storage_for_env = "rpc" if storage_variant not in STORAGE_VARIANTS else storage_variant
    if row in ("baseline", "plugin_idle"):
        # no-op env context; baseline/plugin_idle never write artifacts
        from contextlib import nullcontext
        env_ctx = nullcontext()
    else:
        env_ctx = variant_env(storage_for_env)

    with env_ctx:
        if hook_dir:
            os.makedirs(hook_dir, exist_ok=True)
            clean_hook_dir(hook_dir)

        cfg_path = _write_temp_config(
            base_cfg_path, row=row, layers=n_layers, mode=mode,
        )
        engine, is_hook_llm = _build_llm(
            row=row, model=model, dtype=dtype, hook_dir=hook_dir,
            cfg_path=cfg_path, tp_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=prefix_cache,
            enforce_eager=enforce_eager,
        )

        # Prompts
        tokenizer = engine.tokenizer if is_hook_llm else engine.get_tokenizer()
        prompts = synth_prompts(prompt_len, tokenizer, n=batch_size)

        # Warmup
        for _ in range(warmup):
            _one_rep(engine, is_hook_llm, row=row, prompts=prompts,
                     max_tokens=max_tokens, hook_dir=hook_dir,
                     storage_variant=storage_variant, hooks_on=hooks_on,
                     analyzer_spec=analyzer_spec)

        # Timed reps — reset profiler so the snapshot is per-cell.
        reset_profiler()
        cell_start_ts = time.time()
        peak_gpu = 0.0
        per_rep: List[Dict[str, Any]] = []
        for i in range(reps):
            ret = _one_rep(engine, is_hook_llm, row=row, prompts=prompts,
                           max_tokens=max_tokens, hook_dir=hook_dir,
                           storage_variant=storage_variant, hooks_on=hooks_on,
                           analyzer_spec=analyzer_spec)
            ret["rep_idx"] = i
            per_rep.append(ret)
            peak_gpu = max(peak_gpu, ret["peak_gpu_mb"])

        snap = profile_snapshot()

        # Try to grab the worker's PROF snapshot via the safetensors sidecar
        # BEFORE we tear the engine down — for disk-st* cells this is
        # synchronously available with the latest rep's artifacts. If the
        # cell didn't write safetensors (rpc / disk-pt / shm), the engine-dump
        # poll below picks it up after `del engine` triggers atexit.
        worker_snap = None
        if per_rep and storage_variant.startswith("disk-st"):
            last_rid = per_rep[-1].get("run_id")
            worker_snap = read_worker_sidecar_profile(hook_dir, last_rid)

        # Cleanup
        if cfg_path and cfg_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(cfg_path)
            except OSError:
                pass
        del engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Fallback worker-snapshot source: the EngineCore subprocess writes
        # ``profile-worker-<pid>-<seq>.json`` to VLLM_HOOK_PROFILE_DIR via
        # its atexit handler. Used when the sidecar path didn't fire
        # (rpc / disk-pt / shm storage variants, or no artifact written).
        # We poll briefly because the worker process exits asynchronously.
        if worker_snap is None:
            worker_snap = find_worker_engine_dump(
                os.environ.get("VLLM_HOOK_PROFILE_DIR"),
                since_ts=cell_start_ts,
                timeout_s=5.0,
            )

    # Aggregate
    row_dict = cell_context(
        model=model, dtype=dtype, tp_size=tp_size,
        worker=ROW_TO_WORKER[row][0], analyzer=ROW_TO_WORKER[row][1],
        storage_variant=storage_variant, mode=mode, hooks_on=hooks_on,
        prompt_len=prompt_len, max_tokens=max_tokens, batch_size=batch_size,
        n_layers_captured=(len(n_layers) if isinstance(n_layers, list) else None),
        enable_prefix_caching=prefix_cache, enforce_eager=enforce_eager,
        profile_tier=(2 if os.environ.get("VLLM_HOOK_PROFILE_FINE") == "1" else 1),
        row=row,
    )

    def _mean(key: str) -> float:
        vals = [r[key] for r in per_rep if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _max(key: str) -> float:
        vals = [r[key] for r in per_rep if r.get(key) is not None]
        return max(vals) if vals else 0.0

    row_dict.update({
        "rep_count":           len(per_rep),
        "gen_lat_mean":        _mean("gen_lat"),
        "wait_lat_mean":       _mean("wait_lat"),
        "analyze_lat_mean":    _mean("analyze_lat"),
        "total_lat_mean":      _mean("total_lat"),
        "prefill_tok_per_sec": _mean("prefill_tok_per_sec"),
        "decode_tok_per_sec":  _mean("decode_tok_per_sec"),
        "peak_gpu_mb":         peak_gpu,
        "artifact_kb_mean":    _mean("artifact_kb"),
        # Advanced memory aggregates — max across reps captures the
        # worst-case spike; mean is what a steady stream would see.
        "cuda_alloc_mb_max":         _max("cuda_alloc_mb_post"),
        "cuda_peak_alloc_mb_max":    _max("cuda_peak_alloc_mb"),
        "cuda_peak_alloc_mb_mean":   _mean("cuda_peak_alloc_mb"),
        "cuda_reserved_mb_max":      _max("cuda_reserved_mb_post"),
        "cuda_peak_reserved_mb_max": _max("cuda_peak_reserved_mb"),
        "cuda_alloc_delta_mb_max":   _max("cuda_alloc_delta_mb"),
        "cuda_alloc_delta_mb_mean":  _mean("cuda_alloc_delta_mb"),
        "host_rss_mb_max":           _max("host_rss_mb"),
        "host_rss_delta_mb_max":     _max("host_rss_delta_mb"),
    })
    row_dict = flatten_profile_into_row(row_dict, snap)
    # Overlay the worker-process snapshot. Driver and worker timer names are
    # disjoint (driver: hookllm.*, rpc.*, analyzer.*, io.*; worker:
    # hook.fire.*, worker.*, async.*) so there are no collisions in the
    # ``timer.*`` namespace. The ``gauge.mem.*`` columns DO overlap — the
    # worker's values overwrite the driver's, which is what we want because
    # the worker process is the one that actually has a CUDA context and
    # thus the only place ``mem.cuda_alloc_mb`` is meaningful.
    if worker_snap:
        row_dict = flatten_profile_into_row(row_dict, worker_snap)
    return row_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True, choices=list(ROW_TO_WORKER))
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--storage", default="rpc", choices=list(STORAGE_VARIANTS))
    p.add_argument("--mode", default="last_token", choices=["last_token", "all_tokens"])
    p.add_argument("--hooks-on", default="prefill", choices=["prefill", "decode", "both"])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-layers", default=None,
                   help="Comma-separated layer indices, or 'all'")
    p.add_argument("--base-config", default=None)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--hook-dir", default=DEFAULT_HOOK_DIR)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--no-enforce-eager", action="store_true")
    p.add_argument("--output-json", default=None,
                   help="Append the cell record as one JSONL line.")
    args = p.parse_args()

    n_layers: Optional[List[int]] = None
    if args.n_layers and args.n_layers != "all":
        n_layers = [int(x) for x in args.n_layers.split(",") if x.strip()]

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    cfg = dict(
        row=args.row, model=args.model, dtype=args.dtype, tp_size=args.tp_size,
        storage_variant=args.storage, mode=args.mode, hooks_on=args.hooks_on,
        prompt_len=args.prompt_len, max_tokens=args.max_tokens,
        batch_size=args.batch_size, n_layers=n_layers,
        base_config=args.base_config, reps=args.reps, warmup=args.warmup,
        hook_dir=args.hook_dir, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=(not args.no_enforce_eager),
    )
    row = run_cell(cfg)
    print(json.dumps(row, indent=2, default=str))
    if args.output_json:
        from _common import append_jsonl
        append_jsonl(row, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
