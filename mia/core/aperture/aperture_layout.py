"""File names and row geometry of the shared files a hybrid delivery reads, per capture kind (HS, QK)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

QK_SIDECAR_NAME = "qk_aperture_meta.jsonl"
HS_SIDECAR_NAME = "hs_aperture_meta.jsonl"


@dataclass(frozen=True)
class ApertureLayout:
    kind: str                                # "hs" | "qk"; also the run-dir lock kind
    files: Tuple[Tuple[str, str], ...]       # per layer: (file prefix, header key of its row shape); files[0] anchors
    layer_base: int                          # 1 (HS) or 0 (QK)
    index_codec: str                         # "runs-v1" | "qk-rows-v1"
    index_name: str
    segment_fmt: str
    segment_glob: str
    segment_re: "re.Pattern[str]"
    delivery_format: str
    rank_marker_format: str
    trim_marker_fmt: str
    trim_marker_glob: str
    trim_marker_format: str
    sidecar_name: str
    default_dump_dir: str

    @property
    def anchor(self) -> str:
        """The file whose rows ARE the request's rows (HS: its one file; QK: k -- every q row is a k row)."""
        return self.files[0][0]

    @property
    def prefixes(self) -> Tuple[str, ...]:
        return tuple(p for p, _k in self.files)

    def layer_file(self, prefix: str, layer: int) -> str:
        return f"{prefix}_{int(layer)}.raw"

    def layer_re(self, prefix: str) -> "re.Pattern[str]":
        return re.compile(rf"^{re.escape(prefix)}_(\d+)\.raw$")

    def row_bytes(self, header: Mapping) -> Dict[str, int]:
        """``{prefix: bytes per row}`` from an index or sidecar header (bfloat16 counts 2 bytes)."""
        name = str(header["dtype"])
        item = int(np.dtype("uint16" if name == "bfloat16" else name).itemsize)
        out: Dict[str, int] = {}
        for prefix, key in self.files:
            width = 1
            for d in header[key]:
                width *= int(d)
            out[prefix] = width * item
        return out


HS_LAYOUT = ApertureLayout(
    kind="hs", files=(("hs_layer", "row_shape"),), layer_base=1, index_codec="runs-v1",
    index_name="hs_run_index.jsonl", segment_fmt="hs_run_index.seg.%06d.jsonl",
    segment_glob="hs_run_index.seg.*.jsonl", segment_re=re.compile(r"hs_run_index\.seg\.(\d{6})\.jsonl$"),
    delivery_format="hs-layer-runs-v1", rank_marker_format="hs-delivery-rank-v1",
    trim_marker_fmt="hs_trim.w%02dof%02d.json", trim_marker_glob="hs_trim.w*of*.json",
    trim_marker_format="hs-trim-floor-v1", sidecar_name=HS_SIDECAR_NAME, default_dump_dir="./hs_aperture_dump")

QK_LAYOUT = ApertureLayout(
    kind="qk", files=(("qk_k_layer", "k_row_shape"), ("qk_q_layer", "q_row_shape")), layer_base=0,
    index_codec="qk-rows-v1", index_name="qk_run_index.jsonl", segment_fmt="qk_run_index.seg.%06d.jsonl",
    segment_glob="qk_run_index.seg.*.jsonl", segment_re=re.compile(r"qk_run_index\.seg\.(\d{6})\.jsonl$"),
    delivery_format="qk-layer-runs-v1", rank_marker_format="qk-delivery-rank-v1",
    trim_marker_fmt="qk_trim.w%02dof%02d.json", trim_marker_glob="qk_trim.w*of*.json",
    trim_marker_format="qk-trim-floor-v1", sidecar_name=QK_SIDECAR_NAME, default_dump_dir="./qk_aperture_dump")

_BY_KIND = {"hs": HS_LAYOUT, "qk": QK_LAYOUT}


def layout(kind) -> ApertureLayout:
    """The descriptor for ``kind`` ("hs" / "qk"); a descriptor is returned as it is."""
    if isinstance(kind, ApertureLayout):
        return kind
    try:
        return _BY_KIND[str(kind)]
    except KeyError:
        raise ValueError(f"no hybrid layout for capture kind {kind!r}; want one of {sorted(_BY_KIND)}") from None
