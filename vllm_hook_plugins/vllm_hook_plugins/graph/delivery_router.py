"""Per-request delivery router: given the PREDICTED artifact bytes and whether the chosen analyzer is
REDUCIBLE server-side, pick the transport (rpc vs disk) and where to analyze (inflight/from_disk/none).
Thresholds are CALIBRATED from a profile, never guessed. Pure logic — no torch/GPU."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    transport: str        # 'rpc' | 'disk'
    analyze_where: str    # 'none' (deliver raw) | 'inflight' (host buf) | 'from_disk'

def decide_route(predicted_bytes: int, reducible: bool, t_rpc: int, t_analyze: int) -> RouteDecision:
    if reducible:
        # ship the small RESULT: analyze in-flight if the artifact is small enough to hold, else
        # stream to disk and analyze from disk (avoid holding a huge artifact in host RAM).
        return (RouteDecision("rpc", "inflight") if predicted_bytes <= t_analyze
                else RouteDecision("disk", "from_disk"))
    # deliver the RAW artifact: RPC if small, else disk + offload.
    return (RouteDecision("rpc", "none") if predicted_bytes <= t_rpc
            else RouteDecision("disk", "none"))
