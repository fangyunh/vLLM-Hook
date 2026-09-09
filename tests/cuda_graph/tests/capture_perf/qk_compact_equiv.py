"""Standalone CPU equivalence: compact k_all roundtrip == the old padded k_all.

The worker used to send k_stacked = pad_sequence(_k_all_cpu_list(entry)) — a dense
[num_steps, max_len, k_dim] tensor (O(seq^2), ~93% zeros). Compact transfer sends
(full, prefix_ends) and the driver rebuilds pad_sequence([full[:L] for L in ends]). This proves
the rebuild is byte-identical to the old blob, across ragged growing-prefix shapes.

Run: python tests/cuda_graph/tests/capture_perf/qk_compact_equiv.py
"""
import torch
from torch.nn.utils.rnn import pad_sequence


def _k_all_cpu_list(entry):
    """Mirror of probe_hookqk_worker._k_all_cpu_list (graph buffer-mode: deltas + prefix_ends)."""
    prefix_ends = entry.get("k_prefix_ends")
    if not prefix_ends:
        return [t.cpu() for t in entry["k_all"]]
    full = torch.cat([t.cpu() for t in entry["k_all"]], dim=0)
    return [full[:int(L)] for L in prefix_ends]


def _k_all_compact(entry):
    """Mirror of probe_hookqk_worker._k_all_compact."""
    prefix_ends = entry.get("k_prefix_ends")
    if not prefix_ends:
        return None
    full = torch.cat([t.cpu() for t in entry["k_all"]], dim=0)
    return full, [int(L) for L in prefix_ends]


def _reconstruct(full, ends):
    """Mirror of _hook_plugin._reconstruct_compact_qk's rebuild."""
    return pad_sequence([full[:int(L)] for L in ends], batch_first=True)


def run_case(name, deltas):
    """deltas: per-step NEW key row counts. Build a graph-style entry + compare."""
    k_dim = 8
    torch.manual_seed(0)
    rows = []
    cum = 0
    k_all, prefix_ends = [], []
    for d in deltas:
        blk = torch.randn(d, k_dim)
        k_all.append(blk)
        cum += d
        prefix_ends.append(cum)
    entry = {"k_all": k_all, "k_prefix_ends": prefix_ends}

    old = pad_sequence(_k_all_cpu_list(entry), batch_first=True)   # worker's old k_stacked
    full, ends = _k_all_compact(entry)
    new = _reconstruct(full, ends)                                 # driver rebuild

    assert old.shape == new.shape, f"{name}: shape {tuple(old.shape)} vs {tuple(new.shape)}"
    assert torch.equal(old, new), f"{name}: value mismatch max|Δ|={(old-new).abs().max()}"
    # compact bytes vs padded bytes (the whole point): full is O(seq), padded is O(seq^2)
    padded_elems = old.numel()
    compact_elems = full.numel()
    print(f"  PASS {name}: steps={len(deltas)} shape={tuple(old.shape)} "
          f"compact={compact_elems} vs padded={padded_elems} "
          f"({padded_elems/max(1,compact_elems):.1f}x)")


def main():
    print("[qk_compact_equiv] compact (full+ends) rebuild vs old padded k_all")
    run_case("uniform decode x32", [1] * 32)              # last_token decode: +1 key/step
    run_case("prefill+decode", [64] + [1] * 31)           # 64-tok prefill then decode
    run_case("chunked prefill", [16, 16, 16, 16] + [1] * 12)
    run_case("single step", [5])
    run_case("ragged", [3, 1, 7, 2, 9, 1, 1, 40])
    print("[qk_compact_equiv] ALL PASS — compact rebuild is byte-identical; compact is O(seq).")


if __name__ == "__main__":
    main()
