"""v0.1.0 driver — minimal HookLLM-equivalent shim.

v0.1.0 controls the worker via three env vars + two filesystem paths:

    VLLM_HOOK_DIR    — root dir for artifacts
    VLLM_HOOK_FLAG   — touched while hooks are active, removed otherwise
    VLLM_RUN_ID      — file containing newline-separated run_ids;
                       worker reads the last line each pass

Plus a few worker-specific env vars:

    VLLM_HOOK_LAYERS       (HS)  semicolon-separated layer indices
    VLLM_HOOK_HS_MODE      (HS)  last_token | all_tokens
    VLLM_HOOK_LAYER_HEADS  (QK)  layer:head1,head2;layer2:...
    VLLM_HOOKQ_MODE        (QK)  last_token | all_tokens
    VLLM_ACTSTEER_CONFIG   (steer) path to a JSON file with the steering block

The worker is registered via ``LLM(worker_cls=<v010 worker class>)`` — NOT
worker_extension_cls. The two are different vLLM extension points and v0.1.0
predates the mixin path entirely.

This driver exposes a ``HookLLM``-compatible interface so ``profile_one_run``
can use it without branching: ``.generate(prompts, sp, save_to_disk, run_id)``,
``.analyze(...)``, ``.tokenizer``, ``.llm_engine``, ``.analyzer``.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


# Map row name → (worker import path, analyzer class, default analyzer kwargs).
# Worker import path is "module.ClassName" so we can pass it to LLM(worker_cls=...).
V010_WORKERS = {
    "probe_hidden_states_v010": {
        "worker": "profiling.peers.v010.workers.probe_hidden_states_worker.ProbeHiddenStatesWorker",
        "analyzer": "profiling.peers.v010.analyzers.hidden_states_analyzer.HiddenStatesAnalyzer",
    },
    "probe_hook_qk_v010": {
        "worker": "profiling.peers.v010.workers.probe_hookqk_worker.ProbeHookQKWorker",
        "analyzer": "profiling.peers.v010.analyzers.attention_tracker_analyzer.AttntrackerAnalyzer",
    },
    "steer_hook_act_v010": {
        "worker": "profiling.peers.v010.workers.steer_activation_worker.SteerHookActWorker",
        "analyzer": None,  # steering is in-place; no analyzer in v0.1.0
    },
}


def _resolve(dotted: str):
    mod, cls = dotted.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(mod), cls)


def _layer_to_heads_string(layer_to_heads: Dict[int, List[int]]) -> str:
    return ";".join(
        f"{layer}:{','.join(map(str, heads))}"
        for layer, heads in sorted(layer_to_heads.items())
    )


def _populate_env_from_config(row: str, config_file: Optional[str]) -> Dict[str, Optional[str]]:
    """Set the env vars v0.1.0 expects, given the same JSON config files
    the v0.2.0 path uses. Returns a snapshot for restoration on teardown."""
    keys = (
        "VLLM_HOOK_LAYERS", "VLLM_HOOK_HS_MODE",
        "VLLM_HOOK_LAYER_HEADS", "VLLM_HOOKQ_MODE",
        "VLLM_ACTSTEER_CONFIG",
    )
    snap = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)

    if not config_file:
        return snap
    with open(config_file) as f:
        cfg = json.load(f)

    if row.startswith("probe_hidden_states") and "hidden_states" in cfg:
        hs = cfg["hidden_states"]
        layers = hs.get("layers", [])
        os.environ["VLLM_HOOK_LAYERS"] = ";".join(map(str, layers))
        os.environ["VLLM_HOOK_HS_MODE"] = hs.get("mode", "last_token")

    if row.startswith("probe_hook_qk"):
        important_heads = (cfg.get("params") or {}).get("important_heads")
        if important_heads:
            l2h: Dict[int, List[int]] = {}
            for layer_idx, head_idx in important_heads:
                l2h.setdefault(layer_idx, []).append(head_idx)
            os.environ["VLLM_HOOK_LAYER_HEADS"] = _layer_to_heads_string(l2h)
        os.environ["VLLM_HOOKQ_MODE"] = (cfg.get("hookq") or {}).get("hookq_mode", "all_tokens")

    if row.startswith("steer_hook_act") and "steering" in cfg:
        os.environ["VLLM_ACTSTEER_CONFIG"] = os.path.abspath(config_file)

    return snap


def _restore_env(snap: Dict[str, Optional[str]]) -> None:
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class V010Engine:
    """v0.1.0-equivalent engine, shaped like ``vllm_hook_plugins.HookLLM``
    so ``profile_one_run`` can use it transparently.

    Each ``generate(run_id=...)`` call:
      1. Appends ``run_id`` to ``RUN_ID.txt`` (matching v0.1.0's
         ``_setup_hooks``).
      2. Touches ``EXTRACT.flag``.
      3. Calls vLLM's ``llm.generate(prompts, sp)``.
      4. Removes ``EXTRACT.flag`` (matching ``_cleanup_hooks``).

    The artifact lands at ``<hook_dir>/<run_id>/tp_rank_<r>/`` per the v0.1.0
    worker's ``execute_model`` post-pass.
    """

    def __init__(
        self,
        *,
        row: str,
        model: str,
        hook_dir: str,
        config_file: Optional[str],
        dtype,
        tp_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
        enforce_eager: bool,
    ):
        if row not in V010_WORKERS:
            raise ValueError(f"V010Engine: unknown row {row!r}")

        from vllm import LLM  # heavy; imported lazily

        os.makedirs(hook_dir, exist_ok=True)
        self.row = row
        self._hook_dir = hook_dir
        self._hook_flag = os.path.join(hook_dir, "EXTRACT.flag")
        self._run_id_file = os.path.join(hook_dir, "RUN_ID.txt")

        # Wipe any stale flag/run_id from a previous cell.
        for p in (self._hook_flag, self._run_id_file):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Env vars must be set BEFORE LLM construction — workers read them in
        # load_model() during EngineCore subprocess startup.
        os.environ["VLLM_HOOK_DIR"]  = os.path.abspath(self._hook_dir)
        os.environ["VLLM_HOOK_FLAG"] = os.path.abspath(self._hook_flag)
        os.environ["VLLM_RUN_ID"]    = os.path.abspath(self._run_id_file)
        self._cfg_env_snap = _populate_env_from_config(row, config_file)

        worker_cls = V010_WORKERS[row]["worker"]

        self.llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            worker_cls=worker_cls,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        # Build analyzer if the row has one. v0.1.0 analyzers take
        # (hook_dir, layer_to_heads) so we reconstruct layer_to_heads from
        # the same config the env-var population used.
        self.analyzer = None
        analyzer_path = V010_WORKERS[row]["analyzer"]
        if analyzer_path:
            analyzer_cls = _resolve(analyzer_path)
            layer_to_heads: Dict[int, List[int]] = {}
            if config_file:
                with open(config_file) as f:
                    cfg = json.load(f)
                for layer_idx, head_idx in (cfg.get("params") or {}).get("important_heads", []):
                    layer_to_heads.setdefault(layer_idx, []).append(head_idx)
            self.analyzer = analyzer_cls(self._hook_dir, layer_to_heads)

    # ------------------------------------------------------------------
    # HookLLM-compatible API
    # ------------------------------------------------------------------

    def generate(self, prompts, sampling_params=None, *, save_to_disk: bool = True,
                 run_id: Optional[str] = None, **_):
        """Drive the v0.1.0 flag-file protocol around one generate call."""
        from vllm import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(prompts, str):
            prompts = [prompts]

        rid = run_id or str(uuid.uuid4())
        # Append run_id; worker reads the LAST line each pass.
        with open(self._run_id_file, "a") as f:
            f.write(rid + "\n")
        # Touch the flag — worker checks os.path.exists(flag) per pass.
        open(self._hook_flag, "a").close()
        try:
            return self.llm.generate(prompts, sampling_params)
        finally:
            try:
                os.remove(self._hook_flag)
            except OSError:
                pass

    def analyze(self, analyzer_spec: Optional[Dict[str, Any]] = None,
                run_id: Optional[str] = None, probes: Any = None,
                **_) -> Optional[Dict[str, Any]]:
        if self.analyzer is None:
            return None
        return self.analyzer.analyze(analyzer_spec or {"reduce": "none"})

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Best-effort: remove the flag, restore prior env, drop artifacts."""
        for p in (self._hook_flag, self._run_id_file):
            try:
                os.remove(p)
            except OSError:
                pass
        _restore_env(self._cfg_env_snap)
        for var in ("VLLM_HOOK_DIR", "VLLM_HOOK_FLAG", "VLLM_RUN_ID"):
            os.environ.pop(var, None)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def build_engine(row: str, **kwargs) -> V010Engine:
    """Factory used by profile_one_run._build_llm() for v010 rows."""
    return V010Engine(row=row, **kwargs)
