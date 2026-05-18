import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.v1.worker.gpu_worker import Worker as V1Worker

from vllm_hook_plugins.utils.TokenHighlighter.utils import (
    ForwardAttrCapture,
    apply_soft_removal,
    build_highlighter_record,
    flag_driver_tokens,
    get_attn_key_value_weights,
    lm_head_weight,
    install_persistent_forward_attr_hooks,
    forward_attr_capture_is_ready,
    iter_prompt_batches,
    locate_last_attention,
    merge_real_and_teacher_captures,
    prompt_prefill_done,
    record_batch_meta_from_runner,
    register_ephemeral_forward_attr_hooks,
    request_has_prefill_prompts,
    resolve_grad_model_source,
    save_highlighter_sequences,
    slice_real_prefill_capture,
    teacher_forced_input_ids,
)


class HighlighterWorker(V1Worker):
    """
    Prefill-time worker that computes gradients with 
    respect to the affirmation loss for each prompt.

    Record schema produced per prompt:
    {
      "token_ids": [...],
      "token_scores": [...]  # gradient L2 norm per prompt token
    }
    """

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self._model = getattr(self.model_runner, "model", None)
        if self._model is None:
            raise RuntimeError("HighlighterWorker requires model.")

        self._tokenizer = getattr(self.model_runner, "tokenizer", None)
        if self._tokenizer is None:
            tokenizer_group = getattr(self.model_runner, "tokenizer_group", None)
            self._tokenizer = getattr(tokenizer_group, "tokenizer", None) if tokenizer_group else None

        self._device = next(self._model.parameters()).device
        self.target_phrase = os.environ.get(
            "VLLM_HIGHLIGHTER_TARGET_PHRASE",
            "Sure! I can help with that",
        )
        self.threshold_k = float(os.environ.get("VLLM_HIGHLIGHTER_THRESHOLD_K", "2.0"))
        self.soft_beta = float(os.environ.get("VLLM_HIGHLIGHTER_BETA", "0.1"))
        self._target_tensor = None
        self.target_ids = []
        self._pending_soft = []
        self._grad_model = None
        self._grad_tokenizer = None
        self._grad_device = os.environ.get("VLLM_HIGHLIGHTER_GRAD_DEVICE", "cpu")
        getter = getattr(self._model, "get_input_embeddings", None)
        if callable(getter):
            self._input_embeddings = getter()
        elif getattr(getattr(self._model, "model", None), "embed_tokens", None) is not None:
            self._input_embeddings = self._model.model.embed_tokens
        elif getattr(self._model, "embed_tokens", None) is not None:
            self._input_embeddings = self._model.embed_tokens
        else:
            raise RuntimeError("Input embeddings not found in the model.")
        self._soft_hook = self._input_embeddings.register_forward_hook(self._embedding_soft_hook)

        self._forward_attr_cap = ForwardAttrCapture()
        self._forward_attr_pending_prompts: list[list[int]] = []
        self._forward_attr_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._forward_attr_hooks.extend(
            install_persistent_forward_attr_hooks(
                self._model, self._forward_attr_cap, self.model_runner
            )
        )

        return result

    def _embedding_soft_hook(self, _module, _inputs, output):
        # Apply soft-removal on the matching prefill prompt only.
        if not self._pending_soft or not torch.is_tensor(output):
            return output

        token_ids = None
        if _inputs and torch.is_tensor(_inputs[0]):
            ids = _inputs[0]
            token_ls = ids.tolist() 
            token_ids = ids[0].tolist() if ids.size(0) == 1 and ids.dim() == 2 else token_ls
        if token_ids is None:
            return output

        for idx, pending in enumerate(self._pending_soft):
            prompt_ids = pending["prompt_ids"]
            prompt_len = pending["prompt_len"]
            if len(token_ids) < prompt_len or token_ids[:prompt_len] != prompt_ids:
                continue

            soft_prompt = pending["soft_prompt_embeds"]
            seq_len = output.size(-2) if output.dim() == 3 else output.size(0)
            if output.dim() in (2, 3) and seq_len >= prompt_len:
                slot = output[:prompt_len] if output.dim() == 2 else output[0, :prompt_len]
                slot.copy_(soft_prompt)
                del self._pending_soft[idx]
                return output

        return output

    def _ensure_grad_model(self):
        if self._grad_model is not None:
            return
        model_cfg = getattr(getattr(self.model_runner, "vllm_config", None), "model_config", None)
        model_id = getattr(model_cfg, "model", None)
        if not model_id:
            raise RuntimeError("Unable to resolve model id for autograd scorer.")
        model_source = resolve_grad_model_source(model_id, self.model_runner, self.hook_dir)

        # Load scorer model and tokenizer from the same snapshot path as serving model.
        self._grad_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        ).to(self._grad_device)
        self._grad_tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._grad_model.eval()
        for param in self._grad_model.parameters():
            param.requires_grad_(False)
        if self._tokenizer is None: self._tokenizer = self._grad_tokenizer

    def _run_forward_attr_teacher_forward(
        self, prompt_ids: list[int], *, num_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Synthetic serving forward on ``prompt + target[:-1]`` for attribution positions.

        Uses the same fused-``qkv_proj`` vs split-``q_proj``/``k_proj``/``v_proj`` hooks as real
        prefill. Prompt-prefix activations are merged from the scheduler capture afterward.
        """
        runner = self.model_runner
        if not hasattr(runner, "_dummy_run"):
            raise RuntimeError(
                "forward_attr requires the vLLM model_runner (serving model forward)."
            )

        from vllm.config import CUDAGraphMode

        input_ids = teacher_forced_input_ids(prompt_ids, self._target_tensor, self._device)
        n = int(input_ids.numel()) if num_tokens is None else num_tokens
        input_ids = input_ids[:n]

        runner.input_ids.gpu[:n].copy_(input_ids)
        runner.positions[:n].copy_(torch.arange(n, device=self._device, dtype=torch.long))

        captured: dict[str, torch.Tensor] = {}
        hooks = register_ephemeral_forward_attr_hooks(self._model, captured)
        try:
            with torch.no_grad():
                hidden_states, _ = runner._dummy_run(
                    n,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=True,
                    profile_seq_lens=n,
                )
                captured["h_L"] = hidden_states[:n].detach()
        finally:
            for h in hooks:
                h.remove()

        if not {"Q", "K", "V", "h_L"}.issubset(captured):
            raise RuntimeError("forward_attr teacher-forced forward failed to capture Q/K/V/h_L.")
        return captured

    def _score_prompt_tokens_forward_attr(
        self,
        prompt_ids: list[int],
        real_capture: dict[str, torch.Tensor],
    ) -> list[float]:
        prompt_len = len(prompt_ids)
        teacher = self._run_forward_attr_teacher_forward(prompt_ids)
        captured = merge_real_and_teacher_captures(prompt_len, real_capture, teacher)
        return self._get_grad_influences(
            captured,
            prompt_len,
            locate_last_attention(self._model),
            model=self._model,
        )

    def _get_grad_influences(self, captured, prompt_len, last_attn, model=None):
        """
        Computes dL/dh_i^(L-1) (gradient of loss with respect to hidden states before last layer) for each input token i.
        via the product-rule expansion of d/dh_i^(L-1) [ alpha_ji * v_i ] through the last attention layer:
 
            dL/dh_i^(L-1) = sum_j alpha_ji [  W_V^T W_O^T g_j              (value path)
                                             + (l_ji - l_bar_j) W_K^T q_j / sqrt(d)  (key path) ]
 
        where:
            g_j = dL/dh_j = W_U^T (p_j - e_yj)                LM-head gradient at generation position j
            l_ji = g_j^T W_O v_i                              alignment of v_i with the loss gradient
            l_bar_j = sum_m alpha_jm l_jm = dL/ds_bar_j       softmax baseline (current expected alignment)
            delta_ji = dL/ds_ji = alpha_ji (l_ji - l_bar_j)   softmax-adjusted attention sensitivity
 
        Value path: Effect of altering v_i (hence x_i via W_V) on affirmation loss L,
        given g_j (loss gradient at generation position j), attenuated by attention weights alpha_ji 
        (assumed fixed) before back-projecting to input embedding spaces via W_V^T W_O^T.
        Key path: Effect of altering k_i (hence x_i via W_K) on affirmation loss 
        through pre-softmax scores s_ji where delta_ji is large when j already attends to i
        and v_i is unusually well-aligned with back-projected dL/dh_j relative to generation token j's attended set. 
        
        Influence score = ||dL/dh_i^(L-1)||_2, approximating ||dL/dx_i||_2
        via the residual-stream identity approximation dh_i^(L-1)/dx_i ≈ I.
        """
        model = model if model is not None else self._model
        device = next(model.parameters()).device
        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)

        target = self._target_tensor.squeeze(0).to(device)
        n_gen_toks = target.size(0) # Number of target tokens in affirmation response
        start = prompt_len - 1 # Logit at start position predicts first target token (y_1)
        end = start + n_gen_toks # Logit at end position predicts last target token (y_m)

        # Captured activations — raw shapes from projection outputs:
        # h_L (last layer hidden states): (S_seq, d_model),  Q (query matrix): (S_seq, n_heads*d_head)
        # K (key matrix):   (S_seq, n_kv_heads*d_head),  V (value matrix): (S_seq, n_kv_heads*d_head)
        h_L = captured["h_L"].squeeze(0)
        Q = captured["Q"].squeeze(0)
        K = captured["K"].squeeze(0)
        V = captured["V"].squeeze(0)
        S_seq, d_model = h_L.shape
        d_head = Q.size(-1) // n_heads
        sqrt_d = d_head ** 0.5

        W_U = lm_head_weight(model)
        W_O, W_V, W_K = get_attn_key_value_weights(last_attn)

        # Compute LM head gradient with respect to last layer hidden states
        logits = h_L[start:end] @ W_U.T
        p = torch.softmax(logits, dim=-1)
        p[torch.arange(n_gen_toks, device=device), target] -= 1.0
        g = p @ W_U

        # Reshape projection matrices for batched matrix operations
        Q = Q.view(S_seq, n_heads, d_head).transpose(0, 1).contiguous()
        K = K.view(S_seq, n_kv_heads, d_head).transpose(0, 1).contiguous()
        V = V.view(S_seq, n_kv_heads, d_head).transpose(0, 1).contiguous()
        if n_kv_heads != n_heads:
            # Replicate KV matrices for Grouped Query Attention (common across some model families)
            rep = n_heads // n_kv_heads
            K = K.repeat_interleave(rep, dim=0)
            V = V.repeat_interleave(rep, dim=0)

        # Reshape weight matrices per-head for independent attention head computations
        W_O = W_O.view(d_model, n_heads, d_head).transpose(0, 1).contiguous()
        W_V = W_V.view(n_kv_heads, d_head, d_model)
        W_K = W_K.view(n_kv_heads, d_head, d_model)
        if n_kv_heads != n_heads:
            rep = n_heads // n_kv_heads
            W_V = W_V.repeat_interleave(rep, dim=0)
            W_K = W_K.repeat_interleave(rep, dim=0)

        # Re-compute last layer attention scores for generation positions
        Q_gen = Q[:, start:end, :]
        scores = (Q_gen @ K.transpose(-1, -2)) / sqrt_d
        gen_pos = start + torch.arange(n_gen_toks, device=device)
        all_pos = torch.arange(S_seq, device=device)
        # Causal mask: set attention score a_ji = 0 (after softmax) 
        # for i > t + start (generation token at position start + t cannot look ahead)
        causal = (all_pos.unsqueeze(0) > gen_pos.unsqueeze(1))
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
        A = torch.softmax(scores, dim=-1)
        # Extract attention matrix columns corresponding to prompt tokens
        A_in = A[:, :, :prompt_len]

        # Project LM head gradient with respect to last layer hidden state
        # of generation token j (dL/dh_j = g_j) through W_O per head (to d_head space)
        # g_a[h,j] = W_O^(h)^T (dL/dh_j)
        g_a = torch.einsum("md,hdj->hmj", g, W_O)

        # Compute gradient contribution along value path:
        # dL/dx_i \approx sum_j alpha_ji (W_O^(h)^T (dL/dh_j)) @ W_V^(h)
        # for all input tokens i and generation tokens j
        value_grad = (A_in.transpose(-1, -2) @ g_a) @ W_V
        value_grad = value_grad.sum(dim=0) # Sum over heads

        # Compute gradient contribution along key path:
        # dL/dx_i \approx sum_j alpha_ji (l_ji - l_bar_j) W_K^T q_j / sqrt(d)
        # l_full[j,i] is alignment between back-projected g_j and v_i
        l_full = g_a @ V.transpose(-1, -2) # l_full[j,i] = g_j^T W_O v_i
        # Compute expected alignment baseline over all tokens generation token j attends to:
        # l_bar_j = sum_m alpha_jm l_jm
        # under current attention distribution in matrix A, for each generation token
        baseline = (A * l_full).sum(dim=-1, keepdim=True)
        # Compute softmax-adjusted attention sensitivity for each prompt token i:
        # delta_ji = dL/ds_ji = alpha_ji (l_ji - l_bar_j)
        # delta > 0: increasing alpha_ji increases loss
        delta = A_in * (l_full[:, :, :prompt_len] - baseline)
        # Compute ds_ji/d_h_i^(L-1) = (W_K^T q_j) / sqrt(d)
        # Sensitivity-weighted (delta_ji) sum of query vectors q_j for each input token i
        # Corresponds to a vector in head-space for each input token i that
        # indicates how altering k_i affects affirmation loss (dL/dk_i) 
        key_grad = (delta.transpose(-1, -2) @ Q_gen) / sqrt_d
        key_grad = (key_grad @ W_K).sum(dim=0)

        # Sum to yield (approximate) total gradient dL/x_i
        total = value_grad + key_grad
        return total.norm(p=2, dim=-1).detach().cpu().tolist()

    def _score_prompt_tokens_autograd(self, prompt_ids):
        self._ensure_grad_model()
        target = self._target_tensor.to(self._grad_device)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self._grad_device).unsqueeze(0)
        prompt_len = prompt_tensor.size(1)
        target_prefix = target[:, :-1]
        input_ids = torch.cat([prompt_tensor, target_prefix], dim=1)

        with torch.enable_grad():
            embed_layer = self._grad_model.get_input_embeddings()
            all_embeds = embed_layer(input_ids).detach()
            prompt_embeds = all_embeds[:, :prompt_len, :].clone().requires_grad_(True)
            suffix_embeds = all_embeds[:, prompt_len:, :]
            inputs_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
            attention_mask = torch.ones(
                (1, inputs_embeds.size(1)), dtype=torch.long, device=self._grad_device
            )

            outputs = self._grad_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            start = prompt_len - 1
            end = start + target.size(1)
            target_logits = logits[:, start:end, :].contiguous()

            # Paper-equivalent affirmation loss: -log P(y|x).
            token_log_probs = F.log_softmax(target_logits, dim=-1)
            selected_log_probs = token_log_probs.gather(
                dim=-1, index=target.unsqueeze(-1)
            ).squeeze(-1)
            affirmation_loss = -selected_log_probs.sum()
            affirmation_loss.backward()

            grad = prompt_embeds.grad
            if grad is None:
                raise RuntimeError("Autograd failed: prompt embedding gradient is None.")
            gradient_norms = grad.norm(p=2, dim=-1).squeeze(0)
            return gradient_norms.detach().cpu().tolist()

    def _ensure_target_tensor(self) -> None:
        if self._target_tensor is not None:
            return
        target_ids_env = os.environ.get("VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS", "").strip()
        if target_ids_env:
            self.target_ids = [int(tok.strip()) for tok in target_ids_env.split(",") if tok.strip()]
        else:
            tok = self._grad_tokenizer if self._grad_tokenizer is not None else self._tokenizer
            self.target_ids = tok.encode(self.target_phrase, add_special_tokens=False)
        self._target_tensor = (
            torch.tensor(self.target_ids, dtype=torch.long, device=self._device).unsqueeze(0)
            if self.target_ids
            else None
        )

    def _trace_record(self, prompt_ids: list[int], score_list: list[float]) -> dict:
        return build_highlighter_record(
            prompt_ids,
            score_list,
            tokenizer=self._tokenizer,
            soft_beta=self.soft_beta,
            threshold_k=self.threshold_k,
        )

    def _queue_soft_removal(self, prompt_ids: list[int], score_list: list[float]) -> int:
        prompt_tensor = torch.tensor(
            prompt_ids, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        prompt_len = prompt_tensor.size(1)
        prompt_embeds = self._input_embeddings(prompt_tensor).detach()
        soft_prompt_embeds = apply_soft_removal(
            prompt_embeds,
            score_list,
            soft_beta=self.soft_beta,
            threshold_k=self.threshold_k,
        )
        soft_indices = flag_driver_tokens(score_list, threshold_k=self.threshold_k)
        self._pending_soft.append({
            "prompt_ids": list(prompt_ids),
            "prompt_len": prompt_len,
            "soft_prompt_embeds": soft_prompt_embeds,
        })
        print(
            f"[highlighter] soft-removal applied to {len(soft_indices)}/{prompt_len} prompt tokens "
            f"(beta={self.soft_beta})"
        )
        return len(soft_indices)

    def _forward_attr_soft_reprefill(self, prompt_ids: list[int], score_list: list[float]) -> None:
        """Re-run prompt prefill with softened embeddings (updates KV before decode)."""
        self._queue_soft_removal(prompt_ids, score_list)
        try:
            self._run_forward_attr_teacher_forward(prompt_ids, num_tokens=len(prompt_ids))
        finally:
            if self._pending_soft:
                self._pending_soft.pop()

    def _forward_attr_finish_after_prefill(self) -> None:
        self._ensure_target_tensor()
        sequences = []
        for prompt_ids in self._forward_attr_pending_prompts:
            real_capture = slice_real_prefill_capture(
                prompt_ids, self._forward_attr_cap.live, self._forward_attr_cap.meta
            )
            score_list = self._score_prompt_tokens_forward_attr(
                prompt_ids, real_capture=real_capture
            )
            self._forward_attr_soft_reprefill(prompt_ids, score_list)
            sequences.append(self._trace_record(prompt_ids, score_list))

        self._forward_attr_pending_prompts = []
        self._forward_attr_cap.live = {}
        self._forward_attr_cap.meta = {}
        save_highlighter_sequences(self.hook_dir, self.run_id_file, sequences)

    def _compute_affirmation_gradients(self, request) -> None:
        self._ensure_target_tensor()
        sequences = []
        prompt_batches = iter_prompt_batches(request)

        with torch.inference_mode(False), torch.enable_grad():
            for prompt_ids in prompt_batches:
                score_list = self._score_prompt_tokens_autograd(prompt_ids)
                self._queue_soft_removal(prompt_ids, score_list)
                sequences.append(self._trace_record(prompt_ids, score_list))

        self._model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_highlighter_sequences(self.hook_dir, self.run_id_file, sequences)

    def shutdown(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        super().shutdown()


    def _forward_attr_should_capture(
        self, request: object | None, has_new_prompts: bool
    ) -> bool:
        if has_new_prompts:
            return True
        if not self._forward_attr_pending_prompts:
            return False
        return any(
            not prompt_prefill_done(self.model_runner, prompt_ids)
            for prompt_ids in self._forward_attr_pending_prompts
        )

    def _forward_attr_pending_prefill_done(self) -> bool:
        return all(
            prompt_prefill_done(self.model_runner, prompt_ids)
            for prompt_ids in self._forward_attr_pending_prompts
        )

    def execute_model(self, *args, **kwargs):
        request = args[0] if args else kwargs.get("execute_model_req")
        num_scheduled = int(getattr(request, "total_num_scheduled_tokens", 0) or 0)
        has_new_prompts = request is not None and request_has_prefill_prompts(request)
        can_hook = bool(self.hook_flag and os.path.exists(self.hook_flag))
        scorer = os.environ.get("VLLM_HIGHLIGHTER_SCORER", "autograd")

        if can_hook and scorer != "forward_attr" and has_new_prompts:
            self._compute_affirmation_gradients(request)

        use_forward_attr = (
            can_hook
            and scorer == "forward_attr"
            and num_scheduled > 0
            and self._forward_attr_should_capture(request, has_new_prompts)
        )

        if use_forward_attr and has_new_prompts:
            self._ensure_target_tensor()
            for batch in iter_prompt_batches(request):
                if batch not in self._forward_attr_pending_prompts:
                    self._forward_attr_pending_prompts.append(batch)

        if use_forward_attr:
            self._forward_attr_cap.live = {}
            self._forward_attr_cap.meta = {}
            self._forward_attr_cap.active = True

        try:
            result = super().execute_model(*args, **kwargs)
        finally:
            self._forward_attr_cap.active = False

        if self._forward_attr_pending_prompts and scorer == "forward_attr":
            record_batch_meta_from_runner(self.model_runner, self._forward_attr_cap.meta)
            if forward_attr_capture_is_ready(self._forward_attr_cap):
                if self._forward_attr_pending_prefill_done():
                    self._forward_attr_finish_after_prefill()
            elif num_scheduled > 0 and self._forward_attr_pending_prefill_done():
                raise RuntimeError(
                    "forward_attr: prefill completed but Q/K/V hooks did not capture "
                    "activations during execute_model (check enforce_eager=True)."
                )

        return result
