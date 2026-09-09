from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing

def _ring(n_slots=8):
    # row_bytes irrelevant to the pure math; device cpu, no real tensor needed for logic
    return GpuCaptureRing(row_bytes=16, n_slots=n_slots, device="cpu")

def test_reserve_advances_and_reports_free():
    r = _ring(8)
    assert r.free_rows() == 8 and r.pending_rows() == 0
    start = r.reserve(3)
    assert start == 0
    assert r.pending_rows() == 3 and r.free_rows() == 5

def test_reserve_refuses_when_insufficient_free():
    r = _ring(4)
    assert r.reserve(4) == 0
    assert r.reserve(1) is None        # full -> backpressure signal
    r.advance_drain(2)                 # consumer freed 2
    assert r.reserve(2) == 4           # logical cursor is monotonic

def test_physical_slots_wrap():
    r = _ring(4)
    r.reserve(3)                       # slots 0,1,2 written
    r.advance_drain(3)                 # freed
    start = r.reserve(2)               # logical 3,4 -> physical 3,0
    assert start == 3
    assert r.physical_slots(start, 2) == [3, 0]

def test_drained_segments_split_on_wrap():
    r = _ring(4)
    r.reserve(3); r.advance_drain(1)   # drain=1, write=3 -> physical [1,3)
    assert r.drained_segments() == [(1, 3)]
    r.reserve(2)                       # write=5 -> physical write%4=1, drain=1 => [1..? ] pending=4
    # pending [drain=1, write=5) physical wraps: [1,4) and [0,1)
    segs = r.drained_segments()
    assert segs == [(1, 4), (0, 1)]

def test_sentinel_is_outside_reservable_range():
    r = _ring(4)
    assert 0 <= r.SENTINEL
    # a full reservation never returns the sentinel physical slot
    r.reserve(4)
    assert r.SENTINEL not in r.physical_slots(0, 4)
