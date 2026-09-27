"""Original-pixel correspondence contract and bridge for legacy adapters."""
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
from .utils import resize_to_max_dim


@dataclass
class PreparedImage:
    image_id: str
    shape: tuple
    payload: object
    small_gray: np.ndarray
    scale: np.ndarray  # independent x/y preview scales
    keypoints: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class PairMatches:
    points0: np.ndarray
    points1: np.ndarray
    scores: np.ndarray | None
    metadata: dict = field(default_factory=dict)

    def validated(self, shape0, shape1):
        x, y = (np.asarray(p, np.float32) for p in (self.points0, self.points1))
        if x.ndim != 2 or x.shape[1:] != (2,) or y.shape != x.shape:
            raise ValueError('Matcher must return equally sized N×2 coordinate arrays')
        scores = None if self.scores is None else np.asarray(self.scores, np.float32)
        if scores is not None and scores.shape != (len(x),):
            raise ValueError('Matcher confidence length does not match coordinates')
        valid = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        for points, shape in ((x, shape0), (y, shape1)):
            valid &= (points >= 0).all(1) & (points[:, 0] < shape[1]) & (points[:, 1] < shape[0])
        if scores is not None:
            valid &= np.isfinite(scores)
        return PairMatches(x[valid], y[valid], None if scores is None else scores[valid],
                           {**self.metadata, 'discarded_invalid': int((~valid).sum())})


def prepare_image(engine, image, image_id=''):
    """Input is BGR (OpenCV convention), or an explicitly grayscale image."""
    started = time.perf_counter()
    from .catalog import CATALOG
    name = getattr(engine, 'config', {}).get('engine', engine.name.split('-outdoor')[0])
    color = CATALOG.get(name, {}).get('capabilities', {}).get('color', 'gray')
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    model_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color == 'rgb' and image.ndim == 3 else gray
    payload = engine.extract(model_input)
    small = getattr(payload, 'small_gray', None)
    if small is None:
        small, _ = resize_to_max_dim(gray, 1024)
    shape = image.shape[:2]
    scale = np.array([small.shape[1] / shape[1], small.shape[0] / shape[0]], np.float32)
    metadata = dict(original_shape=list(shape), prepared_shape=list(small.shape[:2]),
                    preview_scale_xy=scale.tolist(), color=color, elapsed=time.perf_counter()-started,
                    padding='per-image square, top-left content, coarse validity mask' if name=='matchanything-eloftr' else 'none',
                    model_resize_stride=32 if name in ('efficientloftr','matchanything-eloftr') else 8 if name.startswith('loftr-') else None)
    return PreparedImage(str(image_id), shape, payload, small, scale,
                         getattr(payload, 'keypoints', np.empty((0, 2), np.float32)), metadata)


def match_pair(engine, a, b):
    started = time.perf_counter()
    x, y, scores = engine.match(a.payload, b.payload)
    return PairMatches(x, y, scores, dict(elapsed=time.perf_counter()-started,
                       preprocessing=[a.metadata, b.metadata], device=str(getattr(engine, 'device', 'cpu')),
                       score_meaning='adapter-specific confidence; not a calibrated probability' if scores is not None else None)
                       ).validated(a.shape, b.shape)
