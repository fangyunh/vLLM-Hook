"""microbench_storage_variants — isolate the storage cost from forward cost.

Builds a synthetic captured-states dict that mimics what the worker would
have collected from a real forward pass, then exercises each storage path
directly:

  - rpc-like:    pickle + zstd compress (then decompress)
  - disk-pt:     torch.save / torch.load
  - disk-pt-async: same as disk-pt; the queue back-pressure is the only
                  difference and is measured by bench_grid against real
                  forwards
  - disk-st:     safetensors.torch.save_file / safe_open
  - disk-st-async: see disk-pt-async note

No vLLM forward involved — pure egress timings on a fixed payload.
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import tempfile
import time
from typing import Dict, List

import torch
import zstandard as zstd

_ZSTD = zstd.ZstdCompressor(level=3)


def make_payload(n_layers: int, batch: int, seq_len: int, hidden: int,
                 dtype: torch.dtype = torch.float16) -> Dict:
    cache = {}
    for i in range(n_layers):
        cache[f"model.layers.{i}"] = {
            "hidden_states": [torch.randn(seq_len, hidden, dtype=dtype) for _ in range(batch)],
            "layer_num": i + 1,
            "hs_mode":  "all_tokens",
        }
    return {"hs_cache": cache, "config": {"hidden_size": hidden, "num_layers": n_layers}}


def bench_rpc(payload: Dict, reps: int) -> Dict[str, float]:
    enc_times, dec_times, sizes = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        blob = _ZSTD.compress(pickle.dumps(payload))
        t1 = time.perf_counter()
        _restored = pickle.loads(zstd.ZstdDecompressor().decompress(blob))
        t2 = time.perf_counter()
        sizes.append(len(blob))
        enc_times.append((t1 - t0) * 1000.0)
        dec_times.append((t2 - t1) * 1000.0)
    return {
        "size_kb":        statistics.mean(sizes) / 1024.0,
        "encode_ms_mean": statistics.mean(enc_times),
        "decode_ms_mean": statistics.mean(dec_times),
    }


def bench_pt(payload: Dict, reps: int, tmpdir: str) -> Dict[str, float]:
    enc_times, dec_times, sizes = [], [], []
    path = os.path.join(tmpdir, "hs.pt")
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.save(payload, path)
        t1 = time.perf_counter()
        _ = torch.load(path, map_location="cpu", weights_only=False)
        t2 = time.perf_counter()
        sizes.append(os.path.getsize(path))
        enc_times.append((t1 - t0) * 1000.0)
        dec_times.append((t2 - t1) * 1000.0)
    return {
        "size_kb":        statistics.mean(sizes) / 1024.0,
        "encode_ms_mean": statistics.mean(enc_times),
        "decode_ms_mean": statistics.mean(dec_times),
    }


def bench_safetensors(payload: Dict, reps: int, tmpdir: str) -> Dict[str, float]:
    from safetensors.torch import save_file
    from safetensors import safe_open
    from torch.nn.utils.rnn import pad_sequence

    # Flatten to padded tensors as the worker does.
    flat: Dict[str, torch.Tensor] = {}
    seq_lens = None
    for name, entry in payload["hs_cache"].items():
        stacked = pad_sequence(entry["hidden_states"], batch_first=True)
        flat[name.replace(".", "__")] = stacked
        if seq_lens is None:
            seq_lens = [t.shape[0] for t in entry["hidden_states"]]

    enc_times, dec_times, sizes = [], [], []
    path = os.path.join(tmpdir, "hs.safetensors")
    for _ in range(reps):
        t0 = time.perf_counter()
        save_file(flat, path)
        t1 = time.perf_counter()
        with safe_open(path, framework="pt") as sf:
            for k in flat:
                _ = sf.get_tensor(k)
        t2 = time.perf_counter()
        sizes.append(os.path.getsize(path))
        enc_times.append((t1 - t0) * 1000.0)
        dec_times.append((t2 - t1) * 1000.0)
    return {
        "size_kb":        statistics.mean(sizes) / 1024.0,
        "encode_ms_mean": statistics.mean(enc_times),
        "decode_ms_mean": statistics.mean(dec_times),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--seq-len",  type=int, default=512)
    p.add_argument("--hidden",   type=int, default=4096)
    p.add_argument("--reps",     type=int, default=10)
    p.add_argument("--tmp-dir",  default="/dev/shm/vllm_hook_microbench")
    p.add_argument("--output",   default=None)
    args = p.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    payload = make_payload(args.n_layers, args.batch, args.seq_len, args.hidden)

    print(f"payload: layers={args.n_layers} batch={args.batch} "
          f"seq_len={args.seq_len} hidden={args.hidden}")

    rows = []
    for name, fn in (("rpc",     lambda: bench_rpc(payload, args.reps)),
                     ("disk-pt", lambda: bench_pt(payload, args.reps, args.tmp_dir)),
                     ("disk-st", lambda: bench_safetensors(payload, args.reps, args.tmp_dir))):
        stats = fn()
        print(f"  {name:>10s}  size_kb={stats['size_kb']:>9.1f}  "
              f"encode_ms={stats['encode_ms_mean']:>7.2f}  "
              f"decode_ms={stats['decode_ms_mean']:>7.2f}")
        rows.append({"variant": name, **stats,
                     "n_layers": args.n_layers, "batch": args.batch,
                     "seq_len": args.seq_len, "hidden": args.hidden})

    if args.output:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _common import save_csv  # noqa: E402
        save_csv(rows, args.output)
        print(f"[microbench] CSV → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
