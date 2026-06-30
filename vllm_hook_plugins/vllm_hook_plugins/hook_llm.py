import copy
import os
import json
import uuid
from typing import Optional, Dict, List
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams
from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.run_utils import dispatch_disk_analyze
from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.shm_utils import teardown_shm

class HookLLM:
    def __init__(
        self,
        model: str,
        worker_name: str = None,
        analyzer_name: str = None,
        config_file: str = None,
        download_dir: Optional[str] = None,
        enable_hook: bool = True,
        hook_dir: str = None,
        enforce_eager: bool = True,
        **vllm_kwargs
    ):

        self.model_name = model
        self.worker_name = worker_name
        self.analyzer_name = analyzer_name
        self.enable_hook = enable_hook
        self.enforce_eager = enforce_eager

        # Expand ~ so HF Hub never sees a literal tilde — it doesn't expand it
        # itself and silently builds a bogus cache path under CWD.
        if download_dir is not None:
            download_dir = os.path.expanduser(download_dir)
        self._download_dir = download_dir

        if hook_dir is not None:
            HOOK_DIR = hook_dir
        else:
            fallback_root = download_dir or os.path.expanduser('~/.cache')
            HOOK_DIR = os.path.join(fallback_root, '_v1_qk_peeks')
        os.makedirs(HOOK_DIR, exist_ok=True)
        self._hook_dir = HOOK_DIR

        self.layer_to_heads = {}
        self._output_layers = None       # set by load_config for HS worker
        self._hookq_mode = "all_tokens"  # default; overridable in config
        self._qk_capture = "qk"          # v0.6.0: "qk" | "score"; overridable in config
        self._score_head = 0             # v0.6.0: head index for score capture
        self._hs_mode = "last_token"     # default; overridable in config
        self._steering_config: Optional[Dict] = None  # set by load_config for steer worker
        if config_file:
            self.load_config(config_file)

        # (Optional) pre-allocate shared memory before worker subprocess is spawned.
        self._hook_shm = None
        if os.environ.get("VLLM_HOOK_USE_SHM", "0") == "1":
            from vllm_hook_plugins.shm_utils import setup_shm
            self._hook_shm = setup_shm(config_file, worker_name)

        worker = None
        if worker_name:
            import vllm.plugins
            vllm.plugins.load_general_plugins()
            worker = PluginRegistry.get_worker(worker_name).path

        llm_kwargs = dict(vllm_kwargs)
        # Only forward download_dir when explicitly set — otherwise let HF Hub
        # use its own default cache (~/.cache/huggingface/hub/), which is where
        # the model snapshots and blobs actually live on this system.
        if download_dir is not None:
            llm_kwargs['download_dir'] = download_dir
        self.llm = LLM(
            model=model,
            worker_extension_cls=worker,
            enforce_eager=enforce_eager,
            **llm_kwargs,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        # v0.5.7 D2: cache model dims for the auto-select size model (full/unsharded counts,
        # matching the worker _conf; head_dim = hidden // H_q). None disables auto-select.
        self._model_dims = None
        try:
            tc = self.llm_engine.model_config.hf_text_config
            H_q = int(getattr(tc, "num_attention_heads"))
            H_kv = int(getattr(tc, "num_key_value_heads", H_q))
            hidden = int(getattr(tc, "hidden_size"))
            self._model_dims = {"H_q": H_q, "H_kv": H_kv, "d": hidden // H_q}
        except Exception:
            self._model_dims = None
        self._autosel_log_n = 0  # rate-limit the per-request auto-select log

        self.analyzer = None
        # v0.5.7 D2: the analyzer's capability gates auto-select (default "qk" => never
        # substitute score). Read the class attr without instantiating heavy deps.
        self._analyzer_accepts = "qk"
        if analyzer_name:
            analyzer_cls = PluginRegistry.get_analyzer(analyzer_name).analyzer
            self._analyzer_accepts = getattr(analyzer_cls, "ACCEPTS", "qk")
            self.analyzer = analyzer_cls(self._hook_dir, self.layer_to_heads)


    def load_config(self, config_file: str):
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        if "params" in config_data and "important_heads" in config_data["params"]:
            self.important_heads = config_data["params"]["important_heads"]
            self.layer_to_heads = {}
            for layer_idx, head_idx in self.important_heads:
                if layer_idx not in self.layer_to_heads:
                    self.layer_to_heads[layer_idx] = []
                self.layer_to_heads[layer_idx].append(head_idx)

        if "hookq" in config_data:
            self._hookq_mode = config_data["hookq"].get("hookq_mode", self._hookq_mode)
            # v0.6.0: capture="score" flushes one head's attention score instead of Q/K.
            self._qk_capture = config_data["hookq"].get("capture", self._qk_capture)
            self._score_head = int(config_data["hookq"].get("score_head", self._score_head))

        if "steering" in config_data:
            # vector_path is taken as-is from the config (legacy: usually relative
            # to the project root / current working directory).
            self._steering_config = dict(config_data["steering"])

        if "hidden_states" in config_data:
            hs_cfg = config_data["hidden_states"]
            layers = hs_cfg.get("layers", [])
            self._hs_mode = hs_cfg.get("mode", "last_token")
            self._output_layers = layers if layers else True

    def _build_extra_args(self, save_to_disk: bool, run_id: str,
                            request_extra_args: Optional[dict] = None) -> dict:
        """Build extra_args for the probe worker based on worker_name and config."""
        extra = {}
        request_extra_args = request_extra_args or {}
        if self.worker_name == "probe_hidden_states":
            extra["output_hidden_states"] = self._output_layers if self._output_layers else True
            extra["hs_mode"] = self._hs_mode
        elif self.worker_name == "probe_hook_qk":
            # Pass layer_to_heads dict for head-level metadata; worker uses keys for layer
            # filtering AND (score mode) as the per-layer head set to score.
            extra["output_qk"] = self.layer_to_heads if self.layer_to_heads else True
            extra["hookq_mode"] = self._hookq_mode
            # Effective capture: a per-request override (explicit user value or the D2
            # auto-select decision stashed on request_extra_args) wins over the config
            # default. v0.6.0/D2 score mode flushes per-head attention scores instead of Q/K.
            cap = request_extra_args.get("qk_capture", self._qk_capture)
            if cap == "score":
                extra["qk_capture"] = "score"
                extra["score_head"] = self._score_head
        elif self.worker_name == "steer_hook_act":
            # Start from the instance-default steer config (loaded from config_file),
            # then apply any per-request overrides from extra_args["steer"].
            base = dict(self._steering_config) if self._steering_config else {}
            override = request_extra_args.get("steer")
            if isinstance(override, dict):
                base.update(override)
            extra["steer"] = base or True
        if save_to_disk:
            extra["save_to_disk"] = True
            extra["run_id"] = run_id
            extra["hook_dir"] = self._hook_dir
        return extra

    # ---- v0.5.7 D2: automatic QK-vs-score selection at admission ----------------------
    def _prompt_token_len(self, prompt) -> Optional[int]:
        """Token length of a prompt for the size model (best-effort; None disables select)."""
        try:
            if isinstance(prompt, str):
                return len(self.tokenizer.encode(prompt))
            if isinstance(prompt, (list, tuple)):
                return len(prompt)  # already a token-id list
            if isinstance(prompt, dict):
                toks = prompt.get("prompt_token_ids")
                if toks is not None:
                    return len(toks)
                txt = prompt.get("prompt")
                if isinstance(txt, str):
                    return len(self.tokenizer.encode(txt))
        except Exception:
            return None
        return None

    def _qk_capture_for(self, prompt_len: int, mode: str) -> str:
        """Pick the smaller capture representation ("qk" | "score") for one request.

        Size model (element counts; both representations store 2 bytes/elem, so counts
        compare directly), summed over the request's analyzer layers with per-layer head
        count ``n``. ``S`` = prompt_len; the prefill pass dominates (decode terms factor out
        of the crossover, so max_tokens does not move the decision):

          all_tokens: QK = S·(H_q+H_kv)·d   score = n·S²    -> score iff S < (H_q+H_kv)·d/n
          last_token: QK ≈ (H_q + S·H_kv)·d score = n·S     -> score iff n < ~H_kv·d

        No analyzer head set (``layer_to_heads`` empty) => score would need ALL heads and is
        ~never smaller, so keep raw QK.
        """
        dims = self._model_dims
        l2h = self.layer_to_heads
        if not dims or not l2h:
            return "qk"
        H_q, H_kv, d = dims["H_q"], dims["H_kv"], dims["d"]
        S = int(prompt_len)
        qk_elems = sc_elems = 0
        for _layer, heads in l2h.items():
            n = len(heads) or 1
            if mode == "all_tokens":
                qk_elems += S * (H_q + H_kv) * d
                sc_elems += n * S * S
            else:  # last_token
                qk_elems += (H_q + S * H_kv) * d
                sc_elems += n * S
        return "score" if sc_elems < qk_elems else "qk"

    def _maybe_auto_select(self, prompt, sp, extra: dict) -> None:
        """Set ``extra["qk_capture"]`` to the size-model-optimal representation in place.

        Fires only for the QK worker when the analyzer can consume score and neither the
        user (per-request ``extra``) nor the config (``self._qk_capture``) pinned a
        representation. The decision is immutable for the request's lifetime. Disable with
        ``VLLM_HOOK_QK_AUTO_SELECT=0``.
        """
        if self.worker_name != "probe_hook_qk":
            return
        # Opt-in (default OFF): default-ON would silently flip every attn_tracker QK user
        # (and the QK parity oracles) to score whenever it is smaller — e.g. last_token,
        # which ~always picks score. Enable deliberately with VLLM_HOOK_QK_AUTO_SELECT=1.
        if os.environ.get("VLLM_HOOK_QK_AUTO_SELECT", "0") != "1":
            return
        if self._analyzer_accepts not in ("score", "either"):
            return
        if "qk_capture" in extra:           # explicit user override -> respect it
            return
        if self._qk_capture != "qk":        # config pinned a representation -> respect it
            return
        if self._model_dims is None:
            return
        prompt_len = self._prompt_token_len(prompt)
        if prompt_len is None:
            return
        pick = self._qk_capture_for(prompt_len, self._hookq_mode)
        extra["qk_capture"] = pick
        if self._autosel_log_n < 16:
            self._autosel_log_n += 1
            print(f"[hookllm/D2] auto-select qk_capture={pick} (S={prompt_len} "
                  f"mode={self._hookq_mode} accepts={self._analyzer_accepts})", flush=True)

    def generate(
        self,
        prompts: List[str],
        sampling_params=None,
        use_hook: Optional[bool] = None,
        save_to_disk: bool = False,
        run_id: Optional[str] = None,
        **kwargs
    ):
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)

        if isinstance(sampling_params, list):
            # list[SamplingParams] allows different hook params within requests
            if len(sampling_params) != len(prompts):
                raise ValueError(
                    f"sampling_params list length ({len(sampling_params)}) "
                    f"must match prompts length ({len(prompts)})"
                )
            sp_list = list(sampling_params)
        else:
            sp_list = [sampling_params] * len(prompts)

        if hook and self.worker_name:
            if run_id is None:
                run_id = str(uuid.uuid4())
            with PROF.timed("hookllm.build_extra"):
                new_sp_list = []
                for prompt, sp in zip(prompts, sp_list):
                    sp = copy.copy(sp)
                    extra = dict(sp.extra_args or {})
                    # v0.5.7 D2: decide qk vs score for THIS request (immutable) before
                    # _build_extra_args fills output_qk/score_head from the choice.
                    self._maybe_auto_select(prompt, sp, extra)
                    # Build per-request so _build_extra_args can merge any
                    # per-request "steer" override (e.g. {"coefficient": C})
                    # into the instance-default steer config.
                    defaults = self._build_extra_args(save_to_disk, run_id,
                                                       request_extra_args=extra)
                    for k, v in defaults.items():
                        extra.setdefault(k, v)  # per-request args win for other keys
                    if "steer" in defaults:
                        # _build_extra_args already merged the per-request override
                        # into the default; overwrite the raw partial with the result.
                        extra["steer"] = defaults["steer"]
                    sp.extra_args = extra
                    new_sp_list.append(sp)
                sp_list = new_sp_list
            # Store last run_id so analyze() can find the artifact without
            # the caller needing to track it.
            self._last_run_id = run_id
        else:
            # clear extra_args when use_hook=False
            new_sp_list = []
            for sp in sp_list:
                if sp.extra_args:
                    sp = copy.copy(sp)
                    sp.extra_args = None
                new_sp_list.append(sp)
            sp_list = new_sp_list

        PROF.incr("hookllm.generate.calls")
        PROF.gauge("hookllm.prompts", len(prompts))
        with PROF.timed("hookllm.generate"):
            if all(sp is sp_list[0] for sp in sp_list):
                # collapse to a single sp if they are all the same
                outputs = self.llm.generate(prompts, sp_list[0])
            else:
                outputs = self.llm.generate(prompts, sp_list)

        if hook and self.worker_name and not save_to_disk and len(outputs) > 1 and getattr(outputs[0], "probes", None) is not None:
            with PROF.timed("hookllm.merge_probes"):
                # Merge per-request probes onto outputs[0] so callers always use
                # output[0].probes regardless of batch size. q/k_all become lists
                # of per-request tensors (unbind dim 0), matching the disk format
                # that unpack_qk expects. k_all varies in seq_len so can't be cat'd.
                all_probes = [o.probes for o in outputs]
                merged = {k: v for k, v in all_probes[0].items() if k not in ("qk_cache", "hs_cache")}
                TENSOR_KEYS = ("q", "k_all", "hidden_states", "scores")
                for cache_key in ("qk_cache", "hs_cache"):
                    if cache_key not in all_probes[0]:
                        continue
                    merged[cache_key] = {}
                    # Union of layers across all requests (heterogeneous per-request layer
                    # sets are allowed; a request missing a layer just contributes nothing).
                    layers = []
                    for p in all_probes:
                        for layer in p.get(cache_key, {}):
                            if layer not in layers:
                                layers.append(layer)
                    for layer in layers:
                        # v0.5.7 D2: a batch may mix representations for one module — auto-select
                        # can pick "scores" for one request and raw q/k_all for another. Collect
                        # whatever each request actually has (skip, don't raise, on a missing key)
                        # so the convenience merge never crashes; each representation lands under
                        # its own key as a per-request list. Non-tensor metadata (layer_num,
                        # hookq_mode, capture, heads) is taken from whichever request carries it.
                        entry = {}
                        for p in all_probes:
                            le = p.get(cache_key, {}).get(layer, {})
                            for k, v in le.items():
                                if k not in TENSOR_KEYS:
                                    entry.setdefault(k, v)
                        for tensor_key in TENSOR_KEYS:
                            vals = []
                            for p in all_probes:
                                le = p.get(cache_key, {}).get(layer, {})
                                v = le.get(tensor_key)
                                # RPC format: q/k_all/hidden_states are STACKED tensors,
                                # scores is a per-pass LIST — both index [0] for pass 0, but
                                # never use bool(v) (ambiguous on a multi-element tensor).
                                if v is not None and len(v) > 0:
                                    vals.append(v[0])
                            if vals:
                                entry[tensor_key] = vals
                        merged[cache_key][layer] = entry
                outputs[0].probes = merged

        return outputs

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        probes: Optional[Dict] = None,
        run_id: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Run the configured analyzer.

        Two paths:
        - In-memory: pass ``probes=output.probes`` from a generate() call that
          used save_to_disk=False. The analyzer receives the RPC artifacts directly.
        - Disk: pass ``run_id`` for single-pass analyzers (HS, AttnTracker), or
          ``run_ids=[doc_run_id, na_run_id]`` for CoRer. Omitting both falls back
          to the last generate()'s run_id.
        """
        if self.analyzer is None:
            print("No analyzer configured")
            return None

        PROF.incr("hookllm.analyze.calls")

        if probes is not None:
            with PROF.timed("hookllm.analyze"):
                with PROF.timed("analyzer.kernel"):
                    return self.analyzer.analyze(analyzer_spec=analyzer_spec, probes=probes)

        # Disk path: resolve run_id from args or _last_run_id fallback.
        effective_run_id = run_id or getattr(self, "_last_run_id", None)
        with PROF.timed("hookllm.analyze"):
            return dispatch_disk_analyze(self.analyzer, analyzer_spec,
                                         run_id=effective_run_id, run_ids=run_ids)

    def close(self):
        """Release resources owned by this wrapper."""
        # Profile dump is handled by atexit in _profiler.py — see _atexit_dump.
        # Calling PROF.dump() in __del__ is unreliable: __del__ fires during
        # interpreter shutdown when sys.meta_path is None and silent
        # ImportErrors swallow the dump entirely.
        teardown_shm(getattr(self, "_hook_shm", None))
        self._hook_shm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
