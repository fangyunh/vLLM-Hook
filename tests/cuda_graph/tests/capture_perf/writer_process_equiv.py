"""Rank-1c equivalence oracle: WRITER_PROCESS off (inline) vs on (process) -> identical on-disk artifacts.

WHY THIS IS DIFFERENT FROM THE PARITY ORACLES (qk_parity.py / hs_parity.py):
  The parity oracles compare GRAPH capture vs EAGER capture within 1e-2. This oracle
  compares the SAME FULL+buffer DISK save path against ITSELF -- once serialized inline on
  the engine loop (VLLM_HOOK_WRITER_PROCESS=0), once by the spawned *writer child process*
  (VLLM_HOOK_WRITER_PROCESS=1). Both paths call the SAME pure graph/artifact_writer.write_artifact,
  so the on-disk artifact must be IDENTICAL. Only WHO runs the serialize (and how the cpu_cache
  crosses to it -- torch.mp shm handoff vs same-process) changes; the bytes must not.

  Gate:
    * .safetensors -> raw-byte compare (the format materializes each tensor contiguously,
      so identical bytes <=> identical shape/dtype/values/order). STRICT.
    * .json sidecar -> structural compare with the volatile "profile" key stripped (the
      inline path bakes the WORKER profiler, the child bakes ITS OWN profiler -> different
      numbers, same everything else).
    * .pt -> LOAD both and torch.equal every tensor (torch.save can embed storage-SHARING
      metadata that the shm round-trip may not reproduce even when every value is identical,
      so raw .pt bytes are not a fair gate; value identity is).

  Both runs capture in SEPARATE subprocesses (GPU is exclusive_process on LSF -> one engine
  per process), same FULL cudagraph + buffer capture, same prompts, same seed, each into its
  OWN fresh hook_dir. Only VLLM_HOOK_WRITER_PROCESS differs.

PATH ASSERTION (hollow-green guard): the wp=1 run MUST actually have used the writer process,
  not silently fallen back to the inline save. With wp=1 the writer child is the ONLY async
  route, so the writer branch is the ONLY producer of the ``worker.queue_put`` timer AND the
  child (not the worker) does the disk write, so the WORKER profiler shows ``worker.queue_put``
  count > 0 and ``worker.disk_write.*`` count == 0. Inline fallback would instead show
  queue_put == 0 and a nonzero in-worker disk_write. So (queue_put > 0 AND in-worker
  disk_write == 0) proves the writer PROCESS did the writes.

USAGE (driven by run_writer_process.sh as three sequential processes per leg):
    python writer_process_equiv.py capture --worker qk --gran last_token --fmt st --wp 0 \
        --hookdir /dev/shm/wp_a --out a.pkl
    python writer_process_equiv.py capture --worker qk --gran last_token --fmt st --wp 1 \
        --hookdir /dev/shm/wp_b --out b.pkl
    python writer_process_equiv.py compare --a a.pkl --b b.pkl

Exit code: capture -> 0 on success; compare -> 0 if artifacts identical AND the wp=1 path
fired, else 1.
"""
import os
# MUST precede the plugin import: beat hook_llm.py's setdefault("TORCHDYNAMO_DISABLE","1")
# so FULL cudagraph mode can actually compile.
os.environ["TORCHDYNAMO_DISABLE"] = "0"

import argparse
import getpass
import glob
import json
import pickle
import shutil
import sys
import time

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16

# A few fixed prompts -> one finished-request DISK save (one run_dir) each, so the writer
# queue + submit-order are exercised across several artifacts.
_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
    {"name": "third", "text": "The mitochondria is the powerhouse of"},
]

# per-worker wiring: (worker_name, analyzer_name, config subdir, cache key, env switch, artifact stem)
_WORKERS = {
    "qk": {
        "worker_name": "probe_hook_qk",
        "analyzer_name": "attn_tracker",
        "cfg_dir": "attention_tracker",
        "capture_env": "VLLM_HOOK_QK_CAPTURE",
        "stem": "qk",
    },
    "hs": {
        "worker_name": "probe_hidden_states",
        "analyzer_name": "hidden_states",
        "cfg_dir": "hidden_states",
        "capture_env": "VLLM_HOOK_HS_CAPTURE",
        "stem": "hidden_states",
    },
}


def _collect_artifacts(hook_dir, stem, fmt):
    """Walk hook_dir; return an ORDERED list of artifact records keyed by
    (run_dir ordinal by mtime, filename). Ordinal (not run_id string) so the two runs line up
    even if vLLM assigns different request/run ids. Reads raw bytes; .json is parsed +
    profile-stripped; .pt kept as raw bytes (loaded at compare time)."""
    run_dirs = []
    for d in glob.glob(os.path.join(hook_dir, "*")):
        if os.path.isdir(d):
            run_dirs.append(d)
    run_dirs.sort(key=lambda d: (os.path.getmtime(d), d))

    records = []
    for ordinal, rd in enumerate(run_dirs):
        # each run_dir has tp_rank_{n}/ subdir(s)
        for tp_dir in sorted(glob.glob(os.path.join(rd, "tp_rank_*"))):
            tp_name = os.path.basename(tp_dir)
            names = []
            if fmt == "st":
                names = [f"{stem}.safetensors", f"{stem}.json"]
            else:
                names = [f"{stem}.pt"]
            for name in names:
                fp = os.path.join(tp_dir, name)
                if not os.path.exists(fp):
                    continue
                key = f"{ordinal:03d}/{tp_name}/{name}"
                if name.endswith(".json"):
                    with open(fp) as f:
                        obj = json.load(f)
                    # Strip fields that legitimately vary RUN-TO-RUN (the two paths are separate
                    # engine boots, not the same cpu_cache serialized twice): the per-process
                    # profiler snapshot and the HS peak-GPU-MB metadata. Everything else
                    # (config, layer_order, seq_lens, batch_size, tp_rank) is deterministic.
                    obj.pop("profile", None)
                    obj.pop("peak_gpu_mb", None)
                    records.append({"key": key, "kind": "json", "obj": obj})
                elif name.endswith(".safetensors"):
                    with open(fp, "rb") as f:
                        records.append({"key": key, "kind": "st", "bytes": f.read()})
                else:  # .pt
                    with open(fp, "rb") as f:
                        records.append({"key": key, "kind": "pt", "bytes": f.read()})
    records.sort(key=lambda r: r["key"])
    return records


def _wait_settled(hook_dir, stem, fmt, want_files, timeout_s=90.0):
    """Poll until the count of final artifact files is >= want_files AND stable across 3
    consecutive 1 s polls. Atomic tmp+rename means a file at its final path is complete, so a
    stable settled count is a safe read barrier for BOTH the inline save and the async writer
    child (which keeps draining while the engine -- its parent -- is alive here)."""
    if fmt == "st":
        pat = os.path.join(hook_dir, "*", "tp_rank_*", f"{stem}.safetensors")
    else:
        pat = os.path.join(hook_dir, "*", "tp_rank_*", f"{stem}.pt")
    stable = 0
    last = -1
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = len(glob.glob(pat))
        if n >= want_files and n == last:
            stable += 1
            if stable >= 3:
                return n
        else:
            stable = 0
        last = n
        time.sleep(1.0)
    return len(glob.glob(pat))


def _read_writer_counters(llm, prof_dir):
    """Dump the WORKER profiler and return the timer call-counts we key the path assertion on.
    worker.queue_put fires only on the writer-submit branch; worker.disk_write.{safetensors,pt}
    fires only when the WORKER itself serializes inline -- NOT when the child process does."""
    try:
        llm.llm.collective_rpc("dump_profiler")
    except Exception as e:  # noqa: BLE001
        print(f"[writer-equiv:capture] dump_profiler RPC failed: {e!r}", flush=True)
    keys = ("worker.queue_put", "worker.disk_write.safetensors", "worker.disk_write.pt")
    out = {k: 0 for k in keys}
    for fp in glob.glob(os.path.join(prof_dir, "*.json")):
        try:
            snap = json.load(open(fp))
        except Exception:  # noqa: BLE001
            continue
        for k in keys:
            t = (snap.get("timers", {}) or {}).get(k)
            if isinstance(t, dict):
                out[k] += int(t.get("count", 0))
    return out


def capture(worker, gran, fmt, wp, hook_dir, out_path):
    spec = _WORKERS[worker]

    # Fresh hook_dir so the two runs don't cross-contaminate and no stale run_dirs are collected.
    if os.path.isdir(hook_dir):
        shutil.rmtree(hook_dir, ignore_errors=True)
    os.makedirs(hook_dir, exist_ok=True)

    # The variable under test -- set BEFORE the engine (and its worker subprocess) boots.
    os.environ["VLLM_HOOK_WRITER_PROCESS"] = "1" if str(wp) == "1" else "0"
    # wp=0: writer process OFF -> flush_disk falls straight to the inline save (the control).
    # wp=1: the writer child is the ONLY async route, so the path assertion below is exact
    #       (queue_put can come from nothing but the writer branch).
    os.environ["VLLM_HOOK_USE_SAFETENSORS"] = "1" if fmt == "st" else "0"
    # FULL cudagraph + buffer capture (the Rank-1c target path).
    os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
    os.environ[spec["capture_env"]] = "buffer"

    # Worker profiler on -> timer call-counts for the path assertion. Profiling only times/counts;
    # it never touches captured values, and its snapshot lands ONLY in the .json "profile" key
    # (stripped before compare) -- so byte-identity of the payload is preserved.
    prof_dir = os.path.abspath(out_path + "_prof")
    os.makedirs(prof_dir, exist_ok=True)
    os.environ["VLLM_HOOK_PROFILE"] = "1"
    os.environ["VLLM_HOOK_PROFILE_DIR"] = prof_dir

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    cfg_name = (f"{_MODEL.split('/')[-1]}.json" if gran == "last_token"
                else f"{_MODEL.split('/')[-1]}_alltok.json")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/{spec['cfg_dir']}/{cfg_name}")

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[writer-equiv:capture] worker={worker} gran={gran} fmt={fmt} "
          f"WRITER_PROCESS={os.environ['VLLM_HOOK_WRITER_PROCESS']} "
          f"cudagraph_mode={cudagraph_mode} capture=buffer config={config_file} "
          f"hook_dir={hook_dir}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name=spec["worker_name"],
        analyzer_name=spec["analyzer_name"],
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=12,
        extra_args={"hooks_on": "both"},
    )

    for case in _CASES:
        llm.generate(case["text"], sp, save_to_disk=True)
        print(f"[writer-equiv:capture] case={case['name']} generated (save_to_disk)", flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass

    # Read barrier: wait for the writer child / inline save to land all artifacts while the
    # engine (the child's parent) is still alive.
    n = _wait_settled(hook_dir, spec["stem"], fmt, want_files=len(_CASES))
    print(f"[writer-equiv:capture] settled: {n} artifact file(s) present "
          f"(wanted >= {len(_CASES)})", flush=True)

    records = _collect_artifacts(hook_dir, spec["stem"], fmt)
    counters = _read_writer_counters(llm, prof_dir)
    print(f"[writer-equiv:capture] collected {len(records)} artifact record(s); "
          f"counters queue_put={counters['worker.queue_put']} "
          f"disk_write.st={counters['worker.disk_write.safetensors']} "
          f"disk_write.pt={counters['worker.disk_write.pt']}", flush=True)

    payload = {"records": records,
               "wp": os.environ["VLLM_HOOK_WRITER_PROCESS"],
               "fmt": fmt,
               "counters": counters}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[writer-equiv:capture] wrote {out_path}", flush=True)
    return 0


def _pt_tensors_equal(a_bytes, b_bytes):
    """Load two .pt blobs and torch.equal every tensor pairwise (structural value identity).
    Returns (ok, detail)."""
    import io
    a = torch.load(io.BytesIO(a_bytes), map_location="cpu", weights_only=False)
    b = torch.load(io.BytesIO(b_bytes), map_location="cpu", weights_only=False)

    mism = []
    # Run-to-run-volatile metadata carried inside cpu_cache (see JSON strip above).
    _skip_keys = {"peak_gpu_mb"}

    def walk(pa, pb, path):
        if isinstance(pa, torch.Tensor) or isinstance(pb, torch.Tensor):
            if not (isinstance(pa, torch.Tensor) and isinstance(pb, torch.Tensor)):
                mism.append(f"{path}: tensor-presence mismatch")
                return
            if pa.shape != pb.shape or pa.dtype != pb.dtype:
                mism.append(f"{path}: shape/dtype {tuple(pa.shape)},{pa.dtype} vs "
                            f"{tuple(pb.shape)},{pb.dtype}")
                return
            if not torch.equal(pa, pb):
                mism.append(f"{path}: values differ")
            return
        if isinstance(pa, dict) and isinstance(pb, dict):
            ka, kb = set(pa) - _skip_keys, set(pb) - _skip_keys
            if ka != kb:
                mism.append(f"{path}: dict keys {sorted(map(str, ka))} vs {sorted(map(str, kb))}")
                return
            for k in ka:
                walk(pa[k], pb[k], f"{path}.{k}")
            return
        if isinstance(pa, (list, tuple)) and isinstance(pb, (list, tuple)):
            if len(pa) != len(pb):
                mism.append(f"{path}: len {len(pa)} vs {len(pb)}")
                return
            for i, (x, y) in enumerate(zip(pa, pb)):
                walk(x, y, f"{path}[{i}]")
            return
        if pa != pb:
            mism.append(f"{path}: {pa!r} vs {pb!r}")

    walk(a, b, "cpu_cache")
    return (len(mism) == 0, mism)


def compare(a_path, b_path):
    with open(a_path, "rb") as f:
        A = pickle.load(f)
    with open(b_path, "rb") as f:
        B = pickle.load(f)

    ra = {r["key"]: r for r in A["records"]}
    rb = {r["key"]: r for r in B["records"]}
    keys = sorted(set(ra) & set(rb))
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))

    overall_ok = True
    total = matched = 0
    if only_a or only_b:
        overall_ok = False
        print(f"[writer-equiv] ARTIFACT SET MISMATCH only_inline={only_a} only_process={only_b}")
    if not keys:
        print("[writer-equiv] NO common artifacts to compare")
        overall_ok = False

    for key in keys:
        a, b = ra[key], rb[key]
        total += 1
        if a["kind"] != b["kind"]:
            overall_ok = False
            print(f"[writer-equiv] {key}: KIND MISMATCH {a['kind']} vs {b['kind']}")
            continue
        if a["kind"] == "st":
            if a["bytes"] == b["bytes"]:
                matched += 1
            else:
                overall_ok = False
                print(f"[writer-equiv] {key}: SAFETENSORS BYTES DIFFER "
                      f"(len {len(a['bytes'])} vs {len(b['bytes'])})")
        elif a["kind"] == "json":
            if a["obj"] == b["obj"]:
                matched += 1
            else:
                overall_ok = False
                print(f"[writer-equiv] {key}: JSON SIDECAR DIFFERS (profile-stripped)")
                print(f"    inline : {a['obj']}")
                print(f"    process: {b['obj']}")
        else:  # pt
            ok, detail = _pt_tensors_equal(a["bytes"], b["bytes"])
            if ok:
                matched += 1
            else:
                overall_ok = False
                print(f"[writer-equiv] {key}: PT TENSORS DIFFER: {detail[:5]}")

    # Path assertion (hollow-green guard): the wp=1 side must have used the writer PROCESS.
    proc_side = next((d for d in (A, B) if str(d.get("wp")) == "1"), None)
    path_ok = True
    if proc_side is None:
        path_ok = False
        print("[writer-equiv] PATH ASSERTION FAILED: neither payload is the wp=1 run.")
    else:
        c = proc_side.get("counters", {})
        qp = int(c.get("worker.queue_put", 0))
        dw = int(c.get("worker.disk_write.safetensors", 0)) + int(c.get("worker.disk_write.pt", 0))
        print(f"[writer-equiv] path-check (wp=1): worker.queue_put={qp} "
              f"in-worker disk_write={dw} (expect queue_put>0 AND disk_write==0)")
        if qp <= 0 or dw != 0:
            path_ok = False
            print("[writer-equiv] PATH ASSERTION FAILED: the wp=1 run did NOT serialize in the "
                  "writer process (queue_put==0 => wrong branch, or in-worker disk_write>0 => "
                  "inline fallback). Byte-equality would be a hollow green.")

    print("=" * 64)
    print(f"[writer-equiv] {matched}/{total} artifacts identical (inline-save vs process-save)")
    if overall_ok and path_ok and total > 0:
        print("[writer-equiv] VERDICT: PASS -- writer-process save is byte-identical to the "
              "inline save, and the writer process was proven to have done the writes.")
        return 0
    print(f"[writer-equiv] VERDICT: FAIL -- "
          f"{'artifact divergence' if not overall_ok else 'writer path not exercised'} "
          f"({matched}/{total} identical).")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--worker", choices=["qk", "hs"], required=True)
    pc.add_argument("--gran", choices=["last_token", "all_tokens"], required=True)
    pc.add_argument("--fmt", choices=["st", "pt"], required=True)
    pc.add_argument("--wp", choices=["0", "1"], required=True)
    pc.add_argument("--hookdir", required=True)
    pc.add_argument("--out", required=True)
    pk = sub.add_parser("compare")
    pk.add_argument("--a", required=True)
    pk.add_argument("--b", required=True)
    args = p.parse_args()

    if args.cmd == "capture":
        sys.exit(capture(args.worker, args.gran, args.fmt, args.wp, args.hookdir, args.out))
    else:
        sys.exit(compare(args.a, args.b))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
