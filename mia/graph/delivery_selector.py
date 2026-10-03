"""Which per-request delivery path a run gets; makes the hybrid the default."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

MODE_ENV = "MIA_APERTURE_DELIVERY"
STAMP_ENV = "MIA_APERTURE_DELIVERY_RESOLVED"

PER_REQUEST_ENV = "MIA_APERTURE_PER_REQUEST"
GRAPH_ENV = "MIA_ALLOW_CUDAGRAPH"
WORKER_ENV = "MIA_WORKER"
PROFILE_ENV = "MIA_PROFILE_MODE"
GATHER_ENV = "MIA_APERTURE_GATHER"
DELIVER_ENV = "MIA_APERTURE_GATHER_DELIVER"
FLUSH_MS_ENV = "MIA_APERTURE_GATHER_FLUSH_MS"
TRIM_ENV = "MIA_APERTURE_GATHER_TRIM"
MMAP_ENV = "MIA_APERTURE_MMAP"
WRITE_MODE_ENV = "MIA_APERTURE_WRITE_MODE"

# A literal, not an import: mia._plugin is mid-import when apply() runs.
HS_WORKER = "hidden_states"

DEFAULT_FLUSH_MS = 200

MODES = ("auto", "hybrid", "drain")
_RESOLVED_MODES = ("hybrid", "drain", "off")
_SOURCES = ("default", "explicit", "scope")


class DeliveryConfigError(RuntimeError):
    """The delivery configuration cannot be resolved into one path, and will not be guessed."""


@dataclass(frozen=True)
class Selection:
    """The resolved delivery path."""
    mode: str
    source: str
    reason: str
    note: str
    env_set: Tuple[Tuple[str, str], ...] = ()
    env_unset: Tuple[str, ...] = ()

    @property
    def stamp(self) -> str:
        return f"{self.mode}:{self.source}"


def _get(env: Mapping[str, str], key: str) -> Optional[str]:
    v = env.get(key)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def _is_on(env: Mapping[str, str], key: str) -> bool:
    return _get(env, key) == "1"


def _mode_env(env: Mapping[str, str]) -> str:
    raw = _get(env, MODE_ENV)
    if raw is None:
        return "auto"
    m = raw.lower()
    if m not in MODES:
        raise DeliveryConfigError(
            f"{MODE_ENV}={raw!r} is not a delivery mode. The values are {MODES[1]!r} (the shared "
            f"drain plus the streaming gather -- per-request artifacts read with "
            f"aperture_gather.load_delivered), {MODES[2]!r} (the in-drain per-request writer, "
            f"delivered through the response), and {MODES[0]!r} (the default: asking for "
            f"{PER_REQUEST_ENV}=1 selects {MODES[1]!r}). Refused rather than defaulted -- a typo "
            f"that fell back to auto would silently measure something nobody asked for.")
    return m


def _read_stamp(env: Mapping[str, str]) -> Optional[Tuple[str, str]]:
    raw = _get(env, STAMP_ENV)
    if raw is None:
        return None
    mode, sep, source = raw.partition(":")
    if not sep or mode not in _RESOLVED_MODES or source not in _SOURCES:
        raise DeliveryConfigError(
            f"{STAMP_ENV}={raw!r} is not a delivery resolution. This variable is written by MIA's "
            f"own selector so a spawned worker agrees with the process that spawned it; it is not "
            f"a setting. Unset it (and use {MODE_ENV} to choose a mode), or report this -- a "
            f"malformed stamp read as a default would make a worker disagree with its engine about "
            f"where a user's data is being delivered.")
    return mode, source


def _refuse_if_hybrid_impossible(env: Mapping[str, str], source: str) -> None:
    chose = ("the hybrid was chosen BY DEFAULT for this run, because "
             f"{PER_REQUEST_ENV}=1 asks for per-request delivery"
             if source == "default" else
             f"the hybrid was chosen explicitly ({MODE_ENV}=hybrid)")
    out = (f"Either remove the conflicting setting, or set {MODE_ENV}=drain to take the in-drain "
           f"per-request writer instead -- which delivers through the response rather than as a "
           f"file.")
    mmap = _get(env, MMAP_ENV)
    if mmap is not None and mmap != "0":
        raise DeliveryConfigError(
            f"{MMAP_ENV}={mmap!r} selects the legacy mmap sink, which PRE-SIZES each layer file and "
            f"truncates it at close -- so a gather reading those files WHILE they are written would "
            f"hand back every row past the write cursor as ZEROS and deliver them as data. "
            f"{chose}, and the gather is what reads them. {out}")
    wm = _get(env, WRITE_MODE_ENV)
    if wm is not None and wm.lower() == "legacy":
        raise DeliveryConfigError(
            f"{WRITE_MODE_ENV}=legacy keeps the sidecar as LayerEntry objects, but the hybrid "
            f"gather derives its per-request row index from the per-step arrays the other write "
            f"modes keep. {chose}. Use auto (the default), direct or buffered. {out}")


_HYBRID_SET = (
    f"{GATHER_ENV}=1 (the run-encoded per-request row index into the shared layer files), "
    f"{DELIVER_ENV}=1 (the gather child that scatters them), "
    f"{FLUSH_MS_ENV}={DEFAULT_FLUSH_MS} (the publication cadence)"
)
_HYBRID_RETRIEVAL = (
    "RETRIEVAL MOVES: a request's activations are a FILE, read with "
    "aperture_gather.load_delivered(out_dir) -> {req_id: {layer: Tensor}} (its manifest being "
    "present is its readiness signal), a few seconds after the request finishes. They are NOT in "
    "output.probes or the HTTP response body, which is where the in-drain per-request writer puts "
    "them."
)


def _hybrid(env: Mapping[str, str], source: str, reason: str) -> Selection:
    _refuse_if_hybrid_impossible(env, source)
    sets = [(GATHER_ENV, "1"), (DELIVER_ENV, "1")]
    cadence = _get(env, FLUSH_MS_ENV)
    if cadence is None:
        sets.append((FLUSH_MS_ENV, str(DEFAULT_FLUSH_MS)))
    # The trim is not written: setting it would flip trim_explicit() for the same value.
    if source == "default":
        how = (f"chosen BY DEFAULT: {PER_REQUEST_ENV}=1 asks for per-request delivery, and the "
               f"HYBRID is how MIA delivers it ({MODE_ENV}=drain takes the in-drain per-request "
               f"writer instead)")
    else:
        how = f"chosen explicitly ({MODE_ENV}=hybrid)"
    note = (f"per-request delivery -> HYBRID (shared-file drain + streaming gather + trim), {how}. "
            f"Arming {_HYBRID_SET}; the trim keeps its own default (ON). {_HYBRID_RETRIEVAL}")
    return Selection(mode="hybrid", source=source, reason=reason, note=note,
                     env_set=tuple(sets), env_unset=(PER_REQUEST_ENV,))


def _drain(source: str, reason: str) -> Selection:
    note = (f"per-request delivery -> the IN-DRAIN per-request writer ({reason}). Delivered through "
            f"the response (output.probes / the HTTP body), block-until-held within "
            f"MIA_APERTURE_DELIVER_TIMEOUT_S. The hybrid gather is NOT running: "
            f"{MODE_ENV}=hybrid selects it.")
    return Selection(mode="drain", source=source, reason=reason, note=note)


def _off(reason: str) -> Selection:
    return Selection(mode="off", source="default", reason=reason,
                     note=f"no per-request delivery ({reason}); nothing changed")


def resolve(env: Mapping[str, str]) -> Selection:
    """Which delivery path this environment means; pure, mutates nothing."""
    stamp = _read_stamp(env)
    if stamp is not None:
        mode, source = stamp
        reason = f"carried in {STAMP_ENV} (resolution stamp)"
        if mode == "hybrid":
            return _hybrid(env, source, reason)
        if mode == "drain":
            return _drain(source, reason)
        return _off(reason)

    mode_env = _mode_env(env)
    per_request = _is_on(env, PER_REQUEST_ENV)
    gather_on = _is_on(env, GATHER_ENV)
    deliver_on = _is_on(env, DELIVER_ENV)

    if mode_env == "hybrid":
        return _hybrid(env, "explicit", f"{MODE_ENV}=hybrid")
    if mode_env == "drain":
        if deliver_on:
            raise DeliveryConfigError(
                f"{MODE_ENV}=drain selects the in-drain per-request writer, but {DELIVER_ENV}=1 "
                f"starts the hybrid gather, which needs the SHARED-file drain -- a per-request "
                f"drain writes no shared layer file for it to read, and the drain refuses the pair "
                f"at construction. Pick one: unset {DELIVER_ENV}, or set {MODE_ENV}=hybrid.")
        return _drain("explicit", f"{MODE_ENV}=drain")

    if per_request and (gather_on or deliver_on):
        flag = GATHER_ENV if gather_on else DELIVER_ENV
        raise DeliveryConfigError(
            f"{PER_REQUEST_ENV}=1 and {flag}=1 are both set, and they are halves of two different "
            f"delivery paths: the gather reads the SHARED per-layer raw files, which a per-request "
            f"drain never writes (the drain refuses this pair at construction, so the engine would "
            f"die at boot). Asking for {PER_REQUEST_ENV}=1 alone already selects the hybrid, which "
            f"arms {flag} itself. Set {MODE_ENV}=hybrid and drop {PER_REQUEST_ENV}, or set "
            f"{MODE_ENV}=drain and drop {flag}.")
    if deliver_on or gather_on:
        return Selection(
            mode="hybrid" if deliver_on else "off", source="explicit",
            reason=f"{DELIVER_ENV}/{GATHER_ENV} set by hand",
            note=(f"the hybrid gather is armed explicitly ({GATHER_ENV}/{DELIVER_ENV} set by "
                  f"hand); the selector changed nothing. {_HYBRID_RETRIEVAL}"
                  if deliver_on else
                  f"{GATHER_ENV}=1 with no {DELIVER_ENV}: the run-encoded index is published and "
                  f"nothing is delivered; the selector changed nothing"))
    if not per_request:
        return _off(f"{PER_REQUEST_ENV} is not 1")
    if not _is_on(env, GRAPH_ENV):
        return _off(f"{PER_REQUEST_ENV}=1 without {GRAPH_ENV}=1, which is inert -- the capture "
                    f"aperture only exists under graph mode")

    kind = env.get(WORKER_ENV)
    if kind not in (None, "", HS_WORKER):
        return _drain("scope",
                      f"the hybrid gather reads hs_layer_*.raw and is HIDDEN-STATES ONLY; this run "
                      f"is {WORKER_ENV}={kind}, whose per-request delivery is the drain's")
    if _is_on(env, PROFILE_ENV):
        return _drain("scope",
                      f"{PROFILE_ENV}=1 measures the capture pipeline only and disables delivery; "
                      f"the hybrid's gather delivers from another process and would not observe it, "
                      f"so the default does not arm it here ({MODE_ENV}=hybrid overrides)")
    return _hybrid(env, "default", f"{PER_REQUEST_ENV}=1 under {GRAPH_ENV}=1, HS, no override")


_ANNOUNCED: set = set()


def apply(env: Optional[MutableMapping[str, str]] = None, *, announce: bool = True) -> Selection:
    """Resolve, write the answer into ``env`` (default ``os.environ``), stamp it, announce once."""
    if env is None:
        env = os.environ
    sel = resolve(env)
    if sel.mode != "off":
        for k, v in sel.env_set:
            env[k] = v
        for k in sel.env_unset:
            env.pop(k, None)
        env[STAMP_ENV] = sel.stamp
    if announce:
        key = (sel.stamp, id(env))
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            if sel.mode != "off":
                print(f"[mia/delivery] {sel.note}", flush=True)
            logger.info("delivery selector: %s (%s) -- %s", sel.mode, sel.source, sel.reason)
    return sel


__all__ = ["DEFAULT_FLUSH_MS", "DeliveryConfigError", "MODES", "MODE_ENV", "STAMP_ENV",
           "Selection", "apply", "resolve"]
