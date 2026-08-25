"""Marker-based progress reporting.

The whole pipeline is described by a fixed list of *stages* (markers).  Every
stage transitions through the states ``pending -> running -> done`` (or
``error``).  A reporter is a thin, serialisable event emitter: it is fully
framework agnostic so the same core can drive a terminal progress bar, a
WebSocket stream or a file logger.

Event envelope (every message is a JSON-able dict)::

    {"type": "plan",     "stages": [...], "engine": "...", ...}
    {"type": "step",     "marker": "...", "status": "...", "message": "...",
                         "progress": 0.42, "detail": {...}}
    {"type": "log",      "level": "...", "message": "..."}
    {"type": "image",    "marker": "...", "label": "...", "image": "data:...",
                         "caption": "..."}
    {"type": "result",   "image": "data:...", "width": ..., "height": ...,
                         "meta": {...}}
    {"type": "error",    "message": "..."}
"""

from __future__ import annotations

import time

# Ordered pipeline markers.  ``weight`` controls how much each stage contributes
# to the global 0..1 progress bar.
STAGES = [
    {"marker": "prepare", "label": "Load & Prepare", "weight": 0.05},
    {"marker": "features", "label": "Detect Features (AI)", "weight": 0.30},
    {"marker": "match", "label": "Match Features", "weight": 0.25},
    {"marker": "align", "label": "Estimate & Align", "weight": 0.15},
    {"marker": "blend", "label": "Blend & Compose", "weight": 0.20},
    {"marker": "crop", "label": "Auto-Crop", "weight": 0.05},
]

# Marker set used by the incremental "growing map" mode (no auto-crop while
# building, since cropping would invalidate accumulated coordinates).
INCR_STAGES = [
    {"marker": "prepare", "label": "Load image", "weight": 0.05},
    {"marker": "features", "label": "Detect features", "weight": 0.35},
    {"marker": "match", "label": "Match to map", "weight": 0.30},
    {"marker": "align", "label": "Align & expand", "weight": 0.15},
    {"marker": "blend", "label": "Blend into map", "weight": 0.15},
]


class ProgressReporter:
    """Collects pipeline events and forwards them to an ``emit`` callback."""

    def __init__(self, emit=None, stages=None):
        self.emit = emit if emit is not None else (lambda event: None)
        self.stages = stages or STAGES
        self._t0 = time.time()
        self._stage = {s["marker"]: {"status": "pending", "fraction": 0.0}
                       for s in self.stages}
        self._history = []

    # -- low level -----------------------------------------------------------
    def _send(self, event):
        self._history.append(event)
        try:
            self.emit(event)
        except Exception:  # a broken transport must never kill stitching
            pass

    @staticmethod
    def _now_ms():
        return int(time.time() * 1000)

    # -- public API ----------------------------------------------------------
    def plan(self, meta=None):
        """Announce the stage plan so consumers can build a timeline up-front."""
        self._send({
            "type": "plan",
            "stages": self.stages,
            "meta": meta or {},
        })

    def global_progress(self):
        total = 0.0
        for s in self.stages:
            w = s["weight"]
            frac = self._stage[s["marker"]]["fraction"]
            total += w * frac
        return min(1.0, total)

    def _step_event(self, marker, status, message, detail, fraction):
        return {
            "type": "step",
            "marker": marker,
            "status": status,
            "message": message or "",
            "progress": round(self.global_progress(), 4),
            "fraction": round(float(fraction), 4),
            "detail": detail or {},
            "ts": self._now_ms(),
        }

    def stage_start(self, marker, message=""):
        self._stage[marker] = {"status": "running", "fraction": 0.0}
        self._send(self._step_event(marker, "running", message, None, 0.0))

    def stage_progress(self, marker, fraction, message=None, detail=None):
        st = self._stage.setdefault(marker, {"status": "running", "fraction": 0.0})
        st["status"] = "running"
        st["fraction"] = max(st["fraction"], float(fraction))
        self._send(self._step_event(marker, "running", message, detail, st["fraction"]))

    def stage_done(self, marker, message="", detail=None):
        self._stage[marker] = {"status": "done", "fraction": 1.0}
        self._send(self._step_event(marker, "done", message, detail, 1.0))

    def stage_error(self, marker, message=""):
        self._stage[marker] = {"status": "error", "fraction": 1.0}
        self._send(self._step_event(marker, "error", message, None, 1.0))

    def log(self, message, level="info", marker=None):
        self._send({
            "type": "log",
            "level": level,
            "message": message,
            "marker": marker,
            "ts": self._now_ms(),
        })

    def image(self, marker, image_b64, label="", caption="", meta=None):
        self._send({
            "type": "image",
            "marker": marker,
            "label": label,
            "caption": caption,
            "image": image_b64,
            "meta": meta or {},
            "ts": self._now_ms(),
        })

    def result(self, image_b64, width, height, meta=None):
        self._send({
            "type": "result",
            "image": image_b64,
            "width": int(width),
            "height": int(height),
            "meta": meta or {},
            "ts": self._now_ms(),
        })

    def error(self, message):
        self._send({"type": "error", "message": message, "ts": self._now_ms()})

    @property
    def history(self):
        return list(self._history)
