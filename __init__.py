"""Panorama / image-stitching toolkit powered by SuperPoint + SuperGlue.

This package exposes a feature-matching-driven stitching pipeline that emits
fine-grained, marker-based progress events so that both a CLI and a web GUI can
visualise every step in real time.
"""

from .stitcher import (PanoramaStitcher, StitchConfig, StitchResult,
                       StitchError, NoExtensionError)
from .incremental import IncrementalStitcher
from .progress import ProgressReporter, STAGES, INCR_STAGES

__all__ = [
    "PanoramaStitcher",
    "IncrementalStitcher",
    "StitchConfig",
    "StitchResult",
    "StitchError",
    "NoExtensionError",
    "ProgressReporter",
    "STAGES",
    "INCR_STAGES",
]

__version__ = "1.0.0"
