"""microbench_serialize — measure current vs proposed HTTP wire formats.

Today (vLLM-Hook): tensor.tolist() → nested JSON list of floats.
Proposed (vllm-lens-style): tensor.numpy().tobytes() → zstd → base64.

The plan.html §7 claim is ~25-30× smaller wire, with the asyncio loop
freed in ms instead of seconds. This microbench verifies that on
realistic probe shapes.

Standalone — no vLLM, no GPU required. Run with regular python.
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import statistics
import sys
import time
from typing import Dict, List, Tuple

import torch
import zstandard as zstd


_ZSTD = zstd.ZstdCompressor(level=3)


def _make_probes(n_layers: int, batch: int, seq_len: int, hidden: int,
                 dtype: torch.dtype = torch.float16) -> Dict:
    """Synthesize a fake hidden-states probe of the requested shape."""
    cache = {}
    for i in range(n_layers):
        cache[f"model.layers.{i}"] = {
            "hidden_states": torch.randn(batch, seq_len, hidden, dtype=dtype),
            "layer_num": i + 1,
            "hs_mode": "all_tokens",
        }
    return {"hs_cache": cache, "config": {"hidden_size": hidden, "num_layers": n_layers}}


# --- vLLM-Hook today (tolist → json) ---------------------------------------

def _serialize_tolist_json(probes: Dict) -> bytes:
    out = {}
    for key, cache in probes.items():
        if key == "config":
            out[key] = cache
            continue
        if not isinstance(cache, dict):
            continue
        out[key] = {}
        for mod_name, entry in cache.items():
            out[key][mod_name] = {
                k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in entry.items()
            }
    return json.dumps(out).encode()


# --- Proposed (bytes → zstd → base64) --------------------------------------

def _serialize_bytes_zstd_b64(probes: Dict) -> bytes:
    out = {}
    for key, cache in probes.items():
        if key == "config":
            out[key] = cache
            continue
        if not isinstance(cache, dict):
            continue
        out[key] = {}
        for mod_name, entry in cache.items():
            sub = {}
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    raw = v.contiguous().cpu().numpy().tobytes()
                    sub[k] = {
                        "shape": list(v.shape),
                        "dtype": str(v.dtype).removeprefix("torch."),
                        "b64":   base64.b64encode(_ZSTD.compress(raw)).decode(),
                    }
                else:
                    sub[k] = v
            out[key][mod_name] = sub
    return json.dumps(out).encode()


# --- Proposed v2 (msgpack / raw binary) ------------------------------------

def _serialize_pickle_zstd(probes: Dict) -> bytes:
    """Internal-RPC-style: pickle + zstd, no base64. Only useful for
    same-process or binary transports; included as the ceiling."""
    return _ZSTD.compress(pickle.dumps(probes))


# ---------------------------------------------------------------------------


def bench(probes: Dict, n_reps: int = 5) -> Dict[str, Dict[str, float]]:
    impls = {
        "tolist_json":        _serialize_tolist_json,
        "bytes_zstd_b64":     _serialize_bytes_zstd_b64,
        "pickle_zstd_binary": _serialize_pickle_zstd,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in impls.items():
        sizes = []
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            blob = fn(probes)
            t1 = time.perf_counter()
            sizes.append(len(blob))
            times.append((t1 - t0) * 1000.0)
        out[name] = {
            "size_kb_mean": statistics.mean(sizes) / 1024.0,
            "time_ms_mean": statistics.mean(times),
            "time_ms_min":  min(times),
            "time_ms_p99":  sorted(times)[min(n_reps - 1, int(n_reps * 0.99))],
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers",   type=int, nargs="+", default=[4, 16, 32])
    p.add_argument("--seq-len",    type=int, nargs="+", default=[64, 256])
    p.add_argument("--hidden",     type=int, default=4096)
    p.add_argument("--batch",      type=int, default=1)
    p.add_argument("--reps",       type=int, default=5)
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    print(f"{'layers':>7} {'seq':>5} {'impl':>22} {'size_kb':>10} "
          f"{'time_ms':>10} {'vs_tolist':>10}")
    rows = []
    for n_layers in args.n_layers:
        for seq_len in args.seq_len:
            probes = _make_probes(n_layers, args.batch, seq_len, args.hidden)
            results = bench(probes, n_reps=args.reps)
            base_size = results["tolist_json"]["size_kb_mean"]
            base_time = results["tolist_json"]["time_ms_mean"]
            for impl, stats in results.items():
                ratio = base_size / stats["size_kb_mean"] if stats["size_kb_mean"] > 0 else float("inf")
                print(f"{n_layers:>7d} {seq_len:>5d} {impl:>22s} "
                      f"{stats['size_kb_mean']:>10.1f} "
                      f"{stats['time_ms_mean']:>10.2f} "
                      f"{ratio:>9.1f}x")
                rows.append({"n_layers": n_layers, "seq_len": seq_len,
                             "hidden": args.hidden, "batch": args.batch,
                             "impl": impl,
                             "size_kb_mean": stats["size_kb_mean"],
                             "time_ms_mean": stats["time_ms_mean"],
                             "time_ms_p99":  stats["time_ms_p99"],
                             "vs_tolist_size_ratio": ratio,
                             "vs_tolist_time_ratio":
                                 base_time / stats["time_ms_mean"]
                                 if stats["time_ms_mean"] > 0 else float("inf")})

    if args.output:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _common import save_csv  # noqa: E402
        save_csv(rows, args.output)
        print(f"\n[microbench] CSV → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
