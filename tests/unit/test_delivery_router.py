from vllm_hook_plugins.graph.delivery_router import decide_route, RouteDecision

T_RPC, T_AN = 1_000_000, 1_000_000
def test_reducible_small_analyzes_inflight_over_rpc():
    assert decide_route(500_000, True, T_RPC, T_AN) == RouteDecision("rpc", "inflight")
def test_reducible_large_analyzes_from_disk():
    assert decide_route(5_000_000, True, T_RPC, T_AN) == RouteDecision("disk", "from_disk")
def test_rawneeded_small_rpc():
    assert decide_route(500_000, False, T_RPC, T_AN) == RouteDecision("rpc", "none")
def test_rawneeded_large_disk_offload():
    assert decide_route(5_000_000, False, T_RPC, T_AN) == RouteDecision("disk", "none")
def test_boundary_is_le_threshold_small():
    assert decide_route(T_RPC, False, T_RPC, T_AN).transport == "rpc"      # == threshold => small
