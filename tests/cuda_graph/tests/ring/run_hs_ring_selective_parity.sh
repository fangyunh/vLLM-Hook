#!/bin/bash
# run_hs_ring_selective_parity.sh -- GPU oracle for Lever C (selective drain,
# VLLM_HOOK_DRAIN_SELECTIVE, plan Task 16, branch capture_ring, base 4fdab35). Every unit test
# Task 15 wrote takes the CPU branch of _read_segments (graph/ring_drain_hs.py); the CUDA branch
# (stream wait, the copy loop, record_stream, event record+sync) has ZERO executed coverage on a
# driver-less box. This job is the only oracle that runs it. See hs_ring_selective_parity.py's
# module docstring for the full rationale (tolerance choice, why concurrent batching, why the
# config file's layer list is irrelevant to what gets installed).
#
# Six legs, ALL in this one job (the brief's four A-D, plus two the reviews required: F, G):
#   A  selective OFF (PINNED =0; the flag defaults ON) -> regression guard for Tasks 14-15
#   B  selective ON,  all layers (2 concurrent reqs)     -> byte-identical AND rows_skipped==0
#   C  selective ON,  uniform subset 4-of-28             -> byte-identical AND rows_skipped>0
#   D  selective ON,  heterogeneous disjoint layer sets  -> byte-identical, rows_skipped>0, each
#                                                            request's own layer set intact
#   F  selective ON + VLLM_HOOK_RING_PER_REQUEST=1       -> NEGATIVE CONTROL: the refusal must
#                                                            fire (selective=False, a reason set,
#                                                            the install-time print present)
#   G  selective ON, per-request hooks_on differs so     -> byte-identical; each layer's plan is
#      per-layer plans are empty/present/absent across      empty, present, or newly-present at
#      steps (pinned-buffer resize stress)                  DIFFERENT steps (see the leg's block)
#
# All 7 use the SAME config file (Qwen2-1.5B-Instruct_alltok.json, all 28 layers, all_tokens) --
# install_hs_hosts sizes the ring from the MODEL's layer count, not the config's `layers` list, so
# every leg installs all 28 layers regardless; it is each case's own `output_hidden_states` extra_
# arg that picks a subset. This is what makes rows_skipped meaningful on the subset legs.
#
# CROSS-ARM GATE (added after the review round -- see the Task 16 fix report; ported from
# hs_ring_hetero_parity.py's leg 7). A per-leg allclose against eager cannot distinguish "matched
# because the mechanism is right" from "matched because misaddressed contents happened to land
# within tolerance" -- selective drain's own failure mode is rows copied from the WRONG ring
# offset, which is shape-correct and layer-set-correct. Two runs AFTER leg E: legA vs legD (full
# drain vs selective heterogeneous compaction) asserts the two
# legs' per-layer error profiles against eager AGREE on their shared case::layer cells. Hard-fails
# on the DEFECT SIGNATURE (cell-set/shape change, graph_max|v| moving >1%, error-to-signal ratio
# moving >10x) rather than bit-identity, mirroring leg 7's own re-specification after a hard
# bit-identity gate went intermittently RED on a healthy tree (see crossarm()'s docstring in
# hs_ring_selective_parity.py for the full calibration this reuses verbatim).
#
# MANDATORY non-vacuity discipline (every leg claiming to exercise the lever, A excepted): the
# python process itself hard-asserts get_drain_row_counts()["selective"] is True via
# VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=1 -- a leg cannot pass while silently full-draining. Legs
# B/C/D/G additionally hard-assert rows_skipped via VLLM_HOOK_SELPAR_REQUIRE_SKIPPED
# (zero for B, positive for C/D/G). Leg F hard-asserts the OPPOSITE: selective must read False
# WITH a reason naming "per-request" (VLLM_HOOK_SELPAR_REQUIRE_REFUSED=1). Every one of these
# raises RuntimeError (nonzero process exit) INSIDE the capture subprocess on violation -- never a
# printed number a human is trusted to notice.
#
# Leg G's construction (why it produces empty/shrink/grow, not asserted by a dedicated counter --
# read as a WORKLOAD-DESIGN argument, verified after the fact from the log's per-layer shapes):
# three concurrent requests, same layer COUNT (2 each) but disjoint layer NUMBERS and DIFFERENT
# hooks_on: A={1,2}/hooks_on=prefill (captures ONLY the prefill step -- its layers' plans exist at
# step 0 and are ABSENT at every decode step after), B={3,4}/hooks_on=both (present every step, the
# control), C={5,6}/hooks_on=decode (ABSENT at step 0, newly PRESENT from step 1 onward -- the
# growth case). hooks_on is read per-request from extra_args by the SAME routing builder that
# produces the ReqCaptureRecords build_copy_plans consumes (graph/install_hs.py), so this pattern
# is a property of the mechanism already proven in run_hs_ring_parity.sh's leg A (hooks_on=prefill)
# and the standing hetero oracle, not something new being trusted here for the first time -- what
# IS new is the per-layer _pinned_buf now resizing independently across these transitions.
#
# WALL TIME. run_hs_ring_parity.sh's own history: a 5th leg pushed a 4-leg job over the queue's
# default RUNLIMIT (job 729355, killed mid-boot, twice). That job's actual completed run time for
# 4 legs was ~350s (LSF 773414) -- comfortably fast -- so the historical kill was likely queue/host
# contention, not this workload being slow. Still, 7 legs here is more than that job had even
# counting its would-be 5th, so an explicit generous wall-time budget is cheap insurance rather
# than a repeat of a wasted resubmission.
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/ring/run_hs_ring_selective_parity.sh
# Result:  grep 'hs-selpar\] VERDICT\|CROSSARM.*VERDICT\|SELECTIVE-WITNESS\|LEG .* RESULT\|CROSSARM(.*) RESULT' \
#              tests/cuda_graph/logs/hs_ring_selective.<JOBID>.out
#
#BSUB -J vllm_hook_hs_ring_selective
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -W 02:00
#BSUB -o tests/cuda_graph/logs/hs_ring_selective.%J.out
#BSUB -e tests/cuda_graph/logs/hs_ring_selective.%J.err
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

# Node-local scratch for the Triton/inductor JIT (the kernel specializations compile at boot;
# a full /tmp aborts a JIT build with "No space left on device" and has killed
# ring jobs before -- applied to every leg so compile-cache locality can't land on only one).
export TMPDIR="/opt/nvme/${USER}/hssel_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/proj/dmfexp/fangyunh/scratch_hsring/tmp_hssel_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_CONFIG_FILE="model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"
export VLLM_HOOK_PARITY_HOOKS_ON=both
export VLLM_HOOK_PARITY_HS_MODE=all_tokens
export VLLM_HOOK_PARITY_MAX_TOKENS=12

WORK="/dev/shm/vllm_hook_${USER}/hs_ring_selective_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/ring/hs_ring_selective_parity.py

echo "[hs-selpar] plugin git: $(git rev-parse --short HEAD 2>/dev/null) $(git diff --quiet 2>/dev/null && echo '(clean)' || echo '(dirty)')"
echo "[hs-selpar] host=$(hostname)"

LEG_RESULTS=""

# Blanks every per-leg knob so nothing leaks from the previous leg's exports (same shell process
# for all 7 legs). Always leaves VLLM_HOOK_SELPAR_NCASES=2 as the common default.
reset_selpar_env () {
  # NOTE (Task 19): unsetting VLLM_HOOK_DRAIN_SELECTIVE now leaves it at its DEFAULT, which is ON.
  # Every leg below therefore states its own value explicitly -- leg A pins 0, the rest pin 1.
  unset VLLM_HOOK_DRAIN_SELECTIVE VLLM_HOOK_RING_PER_REQUEST
  unset VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE VLLM_HOOK_SELPAR_REQUIRE_SKIPPED VLLM_HOOK_SELPAR_REQUIRE_REFUSED
  unset VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE
  export VLLM_HOOK_SELPAR_NCASES=2
  for L in A B C; do
    unset "VLLM_HOOK_SELPAR_LAYERS_$L" "VLLM_HOOK_SELPAR_HOOKS_$L" \
          "VLLM_HOOK_SELPAR_HSMODE_$L" "VLLM_HOOK_SELPAR_MAXTOK_$L" "VLLM_HOOK_SELPAR_TEXT_$L"
  done
}

# Standard leg: graph capture + eager capture + compare. Never aborts the job on a single leg's
# failure (set -e is deliberately NOT used, mirroring run_hs_ring_hetero_parity.sh) -- every leg
# gets a chance to run and report, and LEG_RESULTS accumulates a one-line-per-leg summary.
run_leg () {
  local name="$1"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  local ring_dir="$WORK/${name}_ring"
  echo "============================================================"
  echo "[run_hs_ring_selective_parity] LEG=$name  selective=${VLLM_HOOK_DRAIN_SELECTIVE:-1(default)} per_request=${VLLM_HOOK_RING_PER_REQUEST:-0}"
  echo "[run_hs_ring_selective_parity]   ncases=${VLLM_HOOK_SELPAR_NCASES:-2} require_selective=${VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE:-unset} require_skipped=${VLLM_HOOK_SELPAR_REQUIRE_SKIPPED:-unset} require_degenerate=${VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE:-unset}"
  echo "============================================================"
  echo "[run_hs_ring_selective_parity] === ${name} 1/3 graph(ring) capture (FULL, buffer) ==="
  VLLM_HOOK_RING_DIR="$ring_dir" python -u "$PY" capture --mode graph --out "$g"
  local grc=$?
  echo "[run_hs_ring_selective_parity] === ${name} 2/3 eager capture (ground truth) ==="
  python -u "$PY" capture --mode eager --out "$e"
  local erc=$?
  echo "[run_hs_ring_selective_parity] === ${name} 3/3 compare ==="
  local crc=99
  local profile="$WORK/${name}_profile.pkl"
  if [ "$grc" -eq 0 ] && [ "$erc" -eq 0 ]; then
    # --profile-out always written (cheap): legs A/D/E feed the cross-arm gate below, and having
    # every leg's profile on disk costs nothing but lets a future pair be added without a re-run.
    python -u "$PY" compare --graph "$g" --eager "$e" --profile-out "$profile"
    crc=$?
  else
    echo "[run_hs_ring_selective_parity] LEG $name: capture failed (graph_rc=$grc eager_rc=$erc) -- compare skipped"
  fi
  echo "[run_hs_ring_selective_parity] LEG $name RESULT: graph_rc=$grc eager_rc=$erc compare_rc=$crc"
  LEG_RESULTS="${LEG_RESULTS}${name}:graph_rc=$grc,eager_rc=$erc,compare_rc=$crc\n"
}

# Cross-arm error-profile gate (added after the review round -- see the Task 16 fix report).
# Compares two legs' PROFILE files (written by run_leg's compare step above) on their shared
# case::layer cells -- see hs_ring_selective_parity.py's crossarm() docstring for the full
# rationale and the historical calibration of its pass bands (ported verbatim from
# hs_ring_hetero_parity.py's leg 7).
run_crossarm () {
  local label_a="$1" label_b="$2"
  local pa="$WORK/${label_a}_profile.pkl" pb="$WORK/${label_b}_profile.pkl"
  echo "============================================================"
  echo "[run_hs_ring_selective_parity] === CROSSARM: $label_a vs $label_b ==="
  echo "============================================================"
  local rc=99
  if [ -f "$pa" ] && [ -f "$pb" ]; then
    python -u "$PY" crossarm --a "$pa" --b "$pb" --label-a "$label_a" --label-b "$label_b"
    rc=$?
  else
    echo "[run_hs_ring_selective_parity] CROSSARM $label_a vs $label_b: profile(s) missing (a=$pa b=$pb) -- cannot compare"
  fi
  echo "[run_hs_ring_selective_parity] CROSSARM($label_a vs $label_b) RESULT: rc=$rc"
  LEG_RESULTS="${LEG_RESULTS}crossarm_${label_a}_vs_${label_b}:rc=$rc\n"
}

# Negative-control leg (F): graph-only, no eager counterpart, no compare -- the pass criterion is
# entirely "did the in-process refusal assertion fire", which capture() itself enforces via
# VLLM_HOOK_SELPAR_REQUIRE_REFUSED=1. --no-reconstruct skips the flush_ring/ring_reader read (per-
# request delivery never writes the shared sidecar it would read).
run_leg_refusal () {
  local name="$1"
  local g="$WORK/${name}_graph.pkl"
  local ring_dir="$WORK/${name}_ring"
  echo "============================================================"
  echo "[run_hs_ring_selective_parity] LEG=$name (NEGATIVE CONTROL -- refusal must fire, no eager needed)"
  echo "[run_hs_ring_selective_parity]   selective=${VLLM_HOOK_DRAIN_SELECTIVE:-1(default)} per_request=${VLLM_HOOK_RING_PER_REQUEST:-0}"
  echo "============================================================"
  VLLM_HOOK_RING_DIR="$ring_dir" python -u "$PY" capture --mode graph --out "$g" --no-reconstruct
  local grc=$?
  echo "[run_hs_ring_selective_parity] LEG $name RESULT: graph_rc=$grc (0 = refusal fired as expected)"
  LEG_RESULTS="${LEG_RESULTS}${name}:graph_rc=$grc\n"
}

# ---------------------------------------------------------------------------------------------- #
# LEG A -- selective OFF, PINNED. The regression guard for Tasks 14-15: two concurrent requests,
# all 28 layers, all_tokens, both phases. Same shape as run_hs_ring_parity.sh's leg B but via this
# harness's own code path, so a regression introduced HERE (not in the shipped default) would
# still be caught by the standing oracle, and vice versa.
#   THE `=0` IS LOAD-BEARING as of Task 19: the flag now DEFAULTS ON, so leaving it unset would
#   make this leg a second copy of leg B and delete the full-drain control outright (its
#   REQUIRE_SELECTIVE=0 assertion would fail loud rather than silently -- but the coverage would
#   still be gone). Same swap `a45441d` made to run_hs_ring_parity.sh's leg C at the batched-egress
#   flip, and `run_hs_ring_parity.sh` leg C at the CAPTURE_FUSED flip.
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=0
export VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=0
run_leg legA_selective_off

# ---------------------------------------------------------------------------------------------- #
# LEG B -- selective ON, all layers wanted (both concurrent requests want ALL 28 layers). Nothing
# to skip: rows_skipped must be exactly 0. Task 19: this is also the leg where the DEGENERATE FAST
# PATH must fire on every drained step (degenerate_steps > 0) -- rows_skipped==0 alone cannot tell
# "took the fast path" from "rebuilt the identical plan the slow way", and the whole all-layers
# no-cost claim rests on the former.
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=1
export VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=1
export VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=zero
export VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE=positive
run_leg legB_selective_all_layers

# ---------------------------------------------------------------------------------------------- #
# LEG C -- selective ON, UNIFORM subset (both requests want the SAME 4-of-28 layers). 24 installed
# layers are wanted by nobody every step: rows_skipped must be > 0. Task 19: the degenerate fast
# path must NOT fire here (degenerate_steps == 0) -- a detector that misfired on a subset step would
# copy every layer anyway, silently deleting the lever's entire win while still passing the
# byte-compare AND leaving rows_skipped positive from the OTHER steps. Both directions are pinned.
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=1
export VLLM_HOOK_SELPAR_LAYERS_A="1,2,3,4"
export VLLM_HOOK_SELPAR_LAYERS_B="1,2,3,4"
export VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=1
export VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=positive
export VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE=zero
run_leg legC_selective_uniform_subset4

# ---------------------------------------------------------------------------------------------- #
# LEG D -- selective ON, HETEROGENEOUS: two concurrent requests with DISJOINT 4-layer sets
# ({1,2,3,4} vs {5,6,7,8}). compare()'s wanted_layers check (per case, hard-fails on a set
# mismatch) is what proves "each request's own layer set intact"; rows_skipped must be > 0 (20 of
# 28 installed layers wanted by nobody).
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=1
export VLLM_HOOK_SELPAR_LAYERS_A="1,2,3,4"
export VLLM_HOOK_SELPAR_LAYERS_B="5,6,7,8"
export VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=1
export VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=positive
run_leg legD_selective_heterogeneous

# ---------------------------------------------------------------------------------------------- #
# CROSS-ARM GATE -- the review's required addition. A per-leg allclose against eager cannot tell
# "matched because the mechanism is right" from "matched because misaddressed contents happened to
# land within tolerance" (Task 15's own review named this as selective drain's exact failure mode:
# rows copied from the WRONG ring offset). Every leg above shares case A's/B's prompt texts, so two
# legs that both captured the SAME case's SAME layer number captured the SAME underlying hidden
# state through a DIFFERENT mechanism -- their error profiles against eager must agree.
#   legA vs legD: full drain (OFF, case A wants ALL layers) vs selective heterogeneous compaction
#     (case A wants only [1,2,3,4], disjoint from case B's [5,6,7,8]) -- shares case A's layers
#     1-4 AND case B's layers 5-8 (leg A captured those too, wanting all 28) -- 8 cells.
# ---------------------------------------------------------------------------------------------- #
run_crossarm legA_selective_off legD_selective_heterogeneous

# ---------------------------------------------------------------------------------------------- #
# LEG F -- NEGATIVE CONTROL: selective ON + VLLM_HOOK_RING_PER_REQUEST=1. _resolve_selective must
# refuse (per-request delivery demuxes from a dense host image, incompatible with the compacted
# selective copy) and FULL-drain instead. Asserts selective==False WITH a reason string naming
# "per-request" -- distinguishing "the flag was refused" from "the flag was simply never armed".
# Single short request; no eager counterpart needed (this leg does not compare values).
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=1
export VLLM_HOOK_RING_PER_REQUEST=1
export VLLM_HOOK_SELPAR_NCASES=1
export VLLM_HOOK_SELPAR_MAXTOK_A=4
export VLLM_HOOK_SELPAR_REQUIRE_REFUSED=1
run_leg_refusal legF_selective_refused_by_perrequest

# ---------------------------------------------------------------------------------------------- #
# LEG G -- selective ON, per-layer plans are empty/present/newly-present across DIFFERENT steps.
# Three concurrent requests, disjoint 2-layer sets, DIFFERENT hooks_on: A={1,2}/prefill (present
# at step 0 only -- absent at every decode step after), B={3,4}/both (present every step, the
# control), C={5,6}/decode (absent at step 0, newly present from step 1 on -- the growth case).
# _pinned_buf now resizes per LAYER independently every step; a WAR hazard here would be invisible
# on CPU. rows_skipped must be > 0 (22 of 28 installed layers wanted by nobody at any given step).
# ---------------------------------------------------------------------------------------------- #
reset_selpar_env
export VLLM_HOOK_DRAIN_SELECTIVE=1
export VLLM_HOOK_SELPAR_NCASES=3
export VLLM_HOOK_SELPAR_LAYERS_A="1,2"
export VLLM_HOOK_SELPAR_HOOKS_A=prefill
export VLLM_HOOK_SELPAR_LAYERS_B="3,4"
export VLLM_HOOK_SELPAR_HOOKS_B=both
export VLLM_HOOK_SELPAR_LAYERS_C="5,6"
export VLLM_HOOK_SELPAR_HOOKS_C=decode
export VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE=1
export VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=positive
run_leg legG_selective_growshrink

# ---------------------------------------------------------------------------------------------- #
echo "============================================================"
echo "[run_hs_ring_selective_parity] DONE -- per-leg results:"
echo -e "$LEG_RESULTS"
echo "[run_hs_ring_selective_parity] read VERDICT / SELECTIVE-WITNESS lines above for the full picture."
echo "============================================================"
