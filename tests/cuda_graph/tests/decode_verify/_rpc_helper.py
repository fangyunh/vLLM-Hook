"""Importable so vLLM's EngineCore subprocess can resolve it for collective_rpc.

A __main__-level function can't be unpickled in the worker process (its __main__
is vLLM's, not the test script), so the PROF reader lives here and the test dir
is put on PYTHONPATH.
"""


def read_prof_counters(self=None):
    """Return the worker process's live PROF counters (e.g. steer.fire)."""
    from vllm_hook_plugins._profiler import PROF
    return dict(PROF.counters)
