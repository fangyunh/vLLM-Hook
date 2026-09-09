"""AOT-engaged evidence for the graph-vs-eager parity harnesses (collective_rpc).

WHY THIS EXISTS
  "Does buffer mode support AOT compile?" needs PROOF that AOT actually engaged, not
  just that the run didn't crash. vLLM's AOT compile happens in the model-runner (worker)
  process, and its evidence lives in `vllm.compilation.counter.compilation_counter`:
      num_aot_compiles         > 0  -> a fresh AOT compile ran this boot
      num_aot_artifacts_loaded > 0  -> a cached AOT artifact was loaded ("Directly load
                                       AOT compilation from path ...")
  Either proves AOT is real. The counter is process-local to the worker, so the driver
  reads it via collective_rpc — the SAME cross-process trick decode_verify uses for PROF.

  On a single cold boot with VLLM_USE_AOT_COMPILE=1 the AOT compile still runs even under
  VLLM_DISABLE_COMPILE_CACHE=1 (only the disk *save* is skipped), so num_aot_compiles>0 is
  the decisive single-boot signal — no warm-cache second boot needed offline.

REQUIREMENTS (set by the run_*_aot_full.sh wrappers)
  - This file's dir on PYTHONPATH so the worker can unpickle `read_aot_counters`.
  - VLLM_ALLOW_INSECURE_SERIALIZATION=1 so collective_rpc may ship a plain callable.
"""


def read_aot_counters(self=None):
    """Runs IN the worker process (collective_rpc target). Return live AOT counters."""
    from vllm.compilation.counter import compilation_counter
    return {
        "num_aot_compiles": int(getattr(compilation_counter, "num_aot_compiles", 0)),
        "num_aot_artifacts_loaded": int(
            getattr(compilation_counter, "num_aot_artifacts_loaded", 0)),
        "num_aot_artifacts_saved": int(
            getattr(compilation_counter, "num_aot_artifacts_saved", 0)),
    }


def report_aot_counters(llm, tag="aot"):
    """Driver-side: read the worker AOT counters, print AOT_ENGAGED evidence. Never raises.

    Prints two greppable lines:
        [tag] AOT_COUNTERS num_aot_compiles=N num_aot_artifacts_loaded=M num_aot_artifacts_saved=K
        [tag] AOT_ENGAGED: YES|NO|UNKNOWN
    Returns the aggregated counter dict (or None if the RPC produced nothing).
    """
    rows = None
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc(read_aot_counters)
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] collective_rpc via {type(h).__name__} failed: {e}", flush=True)
            rows = None
            continue
        if rows:
            break
    if not rows:
        print(f"[{tag}] AOT_ENGAGED: UNKNOWN (collective_rpc returned nothing)", flush=True)
        return None

    agg = {"num_aot_compiles": 0, "num_aot_artifacts_loaded": 0, "num_aot_artifacts_saved": 0}
    for r in rows:
        for k in agg:
            agg[k] = max(agg[k], int((r or {}).get(k, 0)))
    engaged = agg["num_aot_compiles"] > 0 or agg["num_aot_artifacts_loaded"] > 0
    print(f"[{tag}] AOT_COUNTERS num_aot_compiles={agg['num_aot_compiles']} "
          f"num_aot_artifacts_loaded={agg['num_aot_artifacts_loaded']} "
          f"num_aot_artifacts_saved={agg['num_aot_artifacts_saved']}", flush=True)
    print(f"[{tag}] AOT_ENGAGED: {'YES' if engaged else 'NO'}", flush=True)
    return agg
