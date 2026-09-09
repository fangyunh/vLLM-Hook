"""v0.5.7 D2 — auto-select size-model unit test (no GPU/engine).

Validates HookLLM._qk_capture_for picks the smaller capture representation per the size
table in docs/v0.5.7_D2_plan.md, including the all_tokens crossover boundary:

  all_tokens: score smaller iff  S < (H_q + H_kv)·d / n
  last_token: score smaller iff  n < ~H_kv·d   (so ~always score)

Builds a bare HookLLM via __new__ (sets only _model_dims + layer_to_heads) so it runs on
a login node with no model load. Run:  python auto_select_unittest.py
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_USE_V1", "1")

from vllm_hook_plugins.hook_llm import HookLLM


def _llm(H_q, H_kv, d, layer_to_heads):
    o = HookLLM.__new__(HookLLM)
    o._model_dims = {"H_q": H_q, "H_kv": H_kv, "d": d}
    o.layer_to_heads = layer_to_heads
    return o


def main():
    ok = True

    def check(label, llm, S, mode, expect):
        nonlocal ok
        got = llm._qk_capture_for(S, mode)
        good = got == expect
        ok = ok and good
        print(f"  [{label}] S={S} mode={mode}: got={got} expect={expect} "
              f"{'OK' if good else 'FAIL'}")

    # Granite-3.1: H_q=32, H_kv=8, d=128 -> (H_q+H_kv)·d = 5120 (the documented n=1 crossover).
    g_1head = _llm(32, 8, 128, {10: [6]})          # n=1 per layer
    check("granite alltok below xover", g_1head, 4096, "all_tokens", "score")
    check("granite alltok above xover", g_1head, 6000, "all_tokens", "qk")
    check("granite alltok at  xover-1", g_1head, 5119, "all_tokens", "score")
    check("granite alltok at  xover",   g_1head, 5120, "all_tokens", "qk")  # not strictly <
    check("granite lasttok always",     g_1head, 6000, "last_token", "score")
    check("granite lasttok tiny S",     g_1head, 4,    "last_token", "score")

    # n=4 heads in the layer -> all_tokens crossover at 5120/4 = 1280.
    g_4head = _llm(32, 8, 128, {10: [6, 9, 1, 20]})
    check("granite n=4 below", g_4head, 1000, "all_tokens", "score")
    check("granite n=4 above", g_4head, 1500, "all_tokens", "qk")

    # Multi-layer with mixed head counts (decision summed over layers). Two layers, n=1 and
    # n=40 -> total score = (1+40)·S²; total qk = 2·S·5120. score < qk iff 41·S < 2·5120
    # => S < 249.7.
    g_multi = _llm(32, 8, 128, {6: [0], 7: list(range(40))})
    check("granite multi below", g_multi, 240, "all_tokens", "score")
    check("granite multi above", g_multi, 260, "all_tokens", "qk")

    # Qwen2-1.5B: H_q=12, H_kv=2, d=128 -> (12+2)·128 = 1792 (n=1 all_tokens crossover).
    qwen = _llm(12, 2, 128, {11: [0]})
    check("qwen alltok below", qwen, 1500, "all_tokens", "score")
    check("qwen alltok above", qwen, 2000, "all_tokens", "qk")
    check("qwen lasttok",      qwen, 2000, "last_token", "score")

    # No analyzer head set -> always raw QK (score would need all heads).
    none = _llm(32, 8, 128, {})
    check("no heads -> qk (alltok)", none, 10, "all_tokens", "qk")
    check("no heads -> qk (lasttok)", none, 10, "last_token", "qk")

    # No model dims -> always qk (size model unavailable).
    nodims = HookLLM.__new__(HookLLM)
    nodims._model_dims = None
    nodims.layer_to_heads = {10: [6]}
    check("no dims -> qk", nodims, 100, "all_tokens", "qk")

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
