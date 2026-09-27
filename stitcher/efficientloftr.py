"""Adapter for the authors' EfficientLoFTR full outdoor model (no fallback)."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PairImage:
    shape: tuple
    small_gray: np.ndarray


class EfficientLoFTREngine:
    name = 'efficientloftr'

    def __init__(self, config, device):
        self.device = device
        self.max_dim = config.get('feature_max_dim', 832)
        self.edge_filter = config.get('edge_filter', False)
        checkpoint = ROOT / 'weights/eloftr_outdoor.ckpt'
        if not checkpoint.is_file() or not (ROOT / 'vendor/EfficientLoFTR/src/loftr').is_dir():
            raise RuntimeError('EfficientLoFTR is not installed. Run .venv/bin/python setup_efficientloftr.py, then restart the server.')
        from vendor.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
        cfg = deepcopy(full_default_cfg)
        # CPU uses full precision; the upstream fine matcher otherwise enables
        # CUDA autocast even when the input is on the CPU.
        self.model = LoFTR(config=cfg)
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
        # The authors' checkpoint includes Lightning callback metadata.
        with torch.serialization.safe_globals([ModelCheckpoint]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['state_dict'])
        self.model = reparameter(self.model).eval().to(device)
        self.model.fine_matching.validate = device == 'cpu'
        self.lock = threading.Lock()  # upstream forward stores per-pair state

    def extract(self, gray):
        h, w = gray.shape
        scale = min(1., self.max_dim / max(h, w))
        hh, ww = max(32, int(h * scale) // 32 * 32), max(32, int(w * scale) // 32 * 32)
        small = cv2.resize(gray, (ww, hh), interpolation=cv2.INTER_AREA)
        if self.edge_filter:
            # Filter both members of each pair at the model's input resolution.
            # Keep the original image untouched for warping and color blending.
            small = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 1.0), 50, 150)
        return PairImage((h, w), small)

    @torch.inference_mode()
    def match(self, a, b):
        batch = {}
        factors = []
        for i, f in enumerate((a, b)):
            h, w = f.small_gray.shape
            batch[f'image{i}'] = torch.from_numpy(f.small_gray).to(self.device, dtype=torch.float32)[None, None] / 255.
            factors.append(np.array([f.shape[1] / w, f.shape[0] / h], dtype=np.float32))
        with self.lock:
            self.model(batch)
        x = batch['mkpts0_f'].cpu().numpy() * factors[0]
        y = batch['mkpts1_f'].cpu().numpy() * factors[1]
        confidence = batch['mconf'].cpu().numpy()
        valid = np.isfinite(x).all(1) & np.isfinite(y).all(1) & np.isfinite(confidence)
        return x[valid], y[valid], confidence[valid]
