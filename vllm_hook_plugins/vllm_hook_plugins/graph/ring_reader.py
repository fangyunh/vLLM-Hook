"""Read-side reconstruction for the capture-ring raw dump: pair the row-major raw file with the
`ring_metadata` sidecar to rebuild each request's per-layer tensors, byte-identical to what was
written. Pure host-side (no GPU): `numpy.memmap` + `torch.from_numpy`.

Contract (see `ring_metadata.py::LayerEntry`): `read_sidecar` returns entries grouped into
`StepMeta`s, but step index/position is NOT meaningful — reconstruction keys on each entry's
`file_row`, the row offset into THAT ENTRY'S OWN raw file (the per-layer file for the multi-layer
reader; the single shared file for `load_ring_artifact`), NEVER on `logical_start` (the ring-wide
reservation position — only equal to `file_row` while every layer receives every row) nor on list
position. `file_row` is always populated (`LayerEntry.__post_init__` / `read_sidecar`'s "fr"-else-"o"
fallback resolve it), so this module reads it unconditionally; `_file_row()` below adds one more layer
of defensiveness for an entry that somehow lacks the attribute. Multiple blocks for the same
(req_id, layer) are sorted by `file_row` (ascending) before concatenation, so the reconstructed token
order is correct regardless of the order entries appear in the sidecar — the reader does not rely on
the upstream drain writing monotonically.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .ring_metadata import read_qk_sidecar, read_sidecar


def _file_row(e) -> int:
    """The row offset into `e`'s OWN raw file: prefer the explicit per-layer `file_row`, else fall
    back to `logical_start`. `LayerEntry.__post_init__` / `read_sidecar` already guarantee
    `file_row` is populated on every entry this module sees in practice, but keying through this
    helper (rather than `e.file_row` directly) keeps the reader correct even against an entry that
    somehow lacks the attribute — "keys on file_row when present, else logical_start", literally."""
    fr = getattr(e, "file_row", None)
    return e.logical_start if fr is None else fr


# Map the header's dtype string to a numpy dtype. bfloat16 has no native numpy dtype: read its raw
# bytes as uint16 and reinterpret via `torch.view(torch.bfloat16)` after the tensor is built (numpy
# has no bf16 type to view into directly).
_NUMPY_DTYPE_BY_NAME = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def load_ring_artifact(raw_path: str, meta_path: str) -> dict:
    """Reconstruct `{req_id: {layer: Tensor}}` from a raw ring dump + its metadata sidecar.

    `raw_path` is the row-major dump of ring rows in logical order (row `i` at byte offset
    `i * row_bytes`); `meta_path` is the `ring_metadata` sidecar written alongside it.
    """
    header, steps = read_sidecar(meta_path)
    dtype_name = header["dtype"]
    row_shape = tuple(header["row_shape"])
    is_bf16 = dtype_name == "bfloat16"
    if is_bf16:
        np_dtype = np.uint16
    else:
        try:
            np_dtype = _NUMPY_DTYPE_BY_NAME[dtype_name]
        except KeyError:
            raise ValueError(f"unsupported dtype {dtype_name!r} in ring header")

    mmap = np.memmap(raw_path, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)

    # Flatten in write order (steps are organizational only; grouping below is keyed on
    # file_row, not on this order).
    entries = [e for s in steps for e in s.entries]

    # Collect every block per (req_id, layer) tagged with its file_row, so multi-block concatenation
    # can be sorted into file row order below — self-contained-correct regardless of the order
    # entries happen to appear in the sidecar (do not rely on writer monotonicity).
    blocks: dict = {}
    for e in entries:
        fr = _file_row(e)
        block = np.array(mmap[fr : fr + e.n_rows])  # copy out of the mmap
        tensor = torch.from_numpy(block)
        if is_bf16:
            tensor = tensor.view(torch.bfloat16)
        blocks.setdefault((e.req_id, e.layer), []).append((fr, tensor))

    out: dict = {}
    for (req_id, layer), parts in blocks.items():
        parts.sort(key=lambda p: p[0])
        tensors = [t for _, t in parts]
        merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        out.setdefault(req_id, {})[layer] = merged

    return out


def _np_dtype_for(dtype_name: str):
    """numpy dtype for a header dtype string; bfloat16 reads as uint16 (reinterpreted after)."""
    if dtype_name == "bfloat16":
        return np.uint16, True
    try:
        return _NUMPY_DTYPE_BY_NAME[dtype_name], False
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype_name!r} in ring header")


def load_multilayer_qk_ring_artifact(run_dir: str, meta_path: str | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: {"q", "k_all", "k_full", "k_prefix_ends", "hookq_mode"}}}``
    from the QK capture-ring dump: one q raw file + one k raw file per layer
    (``qk_q_layer_<L>.raw`` / ``qk_k_layer_<L>.raw``, row-major, written by
    :class:`MultiLayerQKRingDrain`) + ONE shared QK sidecar (``qk_ring_meta.jsonl``).

    Both per-layer files grow in lockstep with the shared ring cursor, so a
    ``QKStepEntry.k_start`` / ``q_start`` is the row offset into ITS file (same invariant as
    :func:`load_multilayer_ring_artifact`, split across the q and k files). Per (req, layer):

      * ``k_full`` = the forwarded key history (cat of each step's ``k_file[k_start:k_start+k_rows]``,
        in ``k_start`` / logical order);
      * ``q``      = cat of the emit_q ``q_file[q_start:q_start+q_rows]`` slices;
      * ``k_all``  = ``[k_full[:L] for L in k_prefix_ends]`` — the growing-prefix reconstruction the
        worker's ``_k_all_cpu_list`` produces (byte-identical to the eager path).

    v1 PREFIX-CACHE / hooks_on=decode LIMIT (deferred, see the module + QKStepEntry docstrings): the
    trimmed cached prefix ``[0, num_computed)`` is NOT written into the ring, so when a request's
    FIRST capture step has ``num_computed > 0`` this raises ``NotImplementedError`` rather than
    returning a k_all that is short by the cached prefix. Fresh prefills (clean / hooks_on=both) have
    first-step ``num_computed == 0`` and reconstruct exactly.
    """
    if meta_path is None:
        meta_path = os.path.join(run_dir, "qk_ring_meta.jsonl")
    header, entries = read_qk_sidecar(meta_path)
    q_dtype, q_is_bf16 = _np_dtype_for(header["dtype"])
    k_dtype, k_is_bf16 = q_dtype, q_is_bf16  # q and k share the model dtype
    q_row_shape = tuple(header["q_row_shape"])
    k_row_shape = tuple(header["k_row_shape"])

    q_mmaps: dict = {}
    k_mmaps: dict = {}

    def _mm(cache: dict, fname: str, layer: int, np_dtype, row_shape):
        mm = cache.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, fname)
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            cache[layer] = mm
        return mm

    # Group entries by (req_id, layer), preserving per-entry step order via k_start.
    grouped: dict = {}
    for e in entries:
        grouped.setdefault((e.req_id, e.layer), []).append(e)

    out: dict = {}
    for (req_id, layer), es in grouped.items():
        es.sort(key=lambda e: e.k_start)
        # First capture step = smallest k_start. v1 does not reconstruct a trimmed prefix.
        if es and es[0].num_computed > 0:
            raise NotImplementedError(
                f"QK capture-ring prefix reconstruction is deferred (v1): request {req_id!r} "
                f"layer {layer} first-step num_computed={es[0].num_computed} > 0 "
                f"(prefix caching or hooks_on=decode). k_full would be short by the cached prefix; "
                f"refusing to return a wrong k_all. Use a fresh prefill / hooks_on in "
                f"{{prefill, both}}, or extend the QK ring path to prepend cached keys.")

        qmm = _mm(q_mmaps, f"qk_q_layer_{layer}.raw", layer, q_dtype, q_row_shape)
        kmm = _mm(k_mmaps, f"qk_k_layer_{layer}.raw", layer, k_dtype, k_row_shape)

        k_parts, q_parts, prefix_ends = [], [], []
        for e in es:
            kb = torch.from_numpy(np.array(kmm[e.k_start:e.k_start + e.k_rows]))
            if k_is_bf16:
                kb = kb.view(torch.bfloat16)
            k_parts.append(kb)
            if e.q_rows > 0 and e.q_start >= 0:
                qb = torch.from_numpy(np.array(qmm[e.q_start:e.q_start + e.q_rows]))
                if q_is_bf16:
                    qb = qb.view(torch.bfloat16)
                q_parts.append(qb)
            if e.prefix_end >= 0:
                prefix_ends.append(int(e.prefix_end))

        k_full = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=0)
        q_cat = (q_parts[0] if len(q_parts) == 1
                 else (torch.cat(q_parts, dim=0) if q_parts else k_full.new_empty((0,) + q_row_shape)))
        k_all = [k_full[:L] for L in prefix_ends]
        out.setdefault(req_id, {})[layer] = {
            "q": q_cat,
            "k_all": k_all,
            "k_full": k_full,
            "k_prefix_ends": prefix_ends,
            "hookq_mode": header.get("hookq_mode"),
        }
    return out


def load_multilayer_ring_artifact(run_dir: str, meta_path: str | None = None) -> dict:
    """Reconstruct ``{req_id: {layer: Tensor}}`` from the HS capture-ring dump: one raw file per
    layer (``hs_layer_<L>.raw``, row-major, written by ``MultiLayerRingDrain``) + ONE shared
    ``ring_metadata`` sidecar (``hs_ring_meta.jsonl``).

    Each ``LayerEntry``'s ``file_row`` is the row offset into ITS OWN layer's file (same invariant as
    the single-file ``load_ring_artifact``, extended to per-layer files) — NOT ``logical_start``, the
    ring-wide reservation position, which only coincides with ``file_row`` today because every
    installed layer's file receives every step's rows (see ``LayerEntry``). Multi-block
    ``(req_id, layer)`` groups are sorted by ``file_row`` before concatenation.
    """
    if meta_path is None:
        meta_path = os.path.join(run_dir, "hs_ring_meta.jsonl")
    header, steps = read_sidecar(meta_path)
    dtype_name = header["dtype"]
    row_shape = tuple(header["row_shape"])
    is_bf16 = dtype_name == "bfloat16"
    if is_bf16:
        np_dtype = np.uint16
    else:
        try:
            np_dtype = _NUMPY_DTYPE_BY_NAME[dtype_name]
        except KeyError:
            raise ValueError(f"unsupported dtype {dtype_name!r} in ring header")

    # One memmap per layer file, opened lazily on first reference (a request may touch only a
    # subset of layers; non-referenced layer files are never opened).
    mmaps: dict = {}

    def _mm(layer: int):
        mm = mmaps.get(layer)
        if mm is None:
            raw = os.path.join(run_dir, f"hs_layer_{layer}.raw")
            mm = np.memmap(raw, dtype=np_dtype, mode="r").reshape((-1,) + row_shape)
            mmaps[layer] = mm
        return mm

    entries = [e for s in steps for e in s.entries]
    blocks: dict = {}
    for e in entries:
        mm = _mm(e.layer)
        fr = _file_row(e)
        block = np.array(mm[fr: fr + e.n_rows])  # copy out of the mmap
        tensor = torch.from_numpy(block)
        if is_bf16:
            tensor = tensor.view(torch.bfloat16)
        blocks.setdefault((e.req_id, e.layer), []).append((fr, tensor))

    out: dict = {}
    for (req_id, layer), parts in blocks.items():
        parts.sort(key=lambda p: p[0])
        tensors = [t for _, t in parts]
        merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        out.setdefault(req_id, {})[layer] = merged

    return out
