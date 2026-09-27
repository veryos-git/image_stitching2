"""Optional learned matchers. Imports and checkpoint loading happen on selection."""
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from .matching import FeatureSet
from .utils import resize_to_max_dim

ROOT = Path(__file__).resolve().parents[1]
# Keep downloaded assets local to the application.
os.environ.setdefault('TORCH_HOME', str(ROOT / '.model_cache'))
os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache' / 'huggingface'))


class AlternativeEngine:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.name = config['engine']
        n = config.get('max_keypoints', 2048)
        if self.name.startswith('xfeat'):
            repo = ROOT / 'vendor' / 'accelerated_features'
            if not repo.exists():
                raise RuntimeError('Install XFeat: git clone https://github.com/verlab/accelerated_features vendor/accelerated_features')
            self.model = torch.hub.load(str(repo), 'XFeat', source='local', pretrained=False, top_k=n)
            self.model.net.load_state_dict(torch.load(repo / 'weights/xfeat.pt', map_location=device, weights_only=True))
        elif self.name.endswith('-lightglue'):
            from lightglue import LightGlue, DISK, ALIKED, SIFT
            kind = self.name.split('-')[0]
            self.model = {'disk': DISK, 'aliked': ALIKED, 'sift': SIFT}[kind](max_num_keypoints=n).eval().to(device)
            self.matcher = LightGlue(features=kind).eval().to(device)
        elif self.name.startswith('loftr'):
            from kornia.feature import LoFTR
            self.model = LoFTR(pretrained=self.name.split('-')[1]).eval().to(device)
        elif self.name in ('roma', 'tiny-roma'):
            from romatch import roma_outdoor, tiny_roma_v1_outdoor
            if self.name == 'roma':
                self.model = roma_outdoor(device=device, use_custom_corr=False)
            else:
                repo = ROOT / 'vendor' / 'accelerated_features'
                xfeat = torch.hub.load(str(repo), 'XFeat', source='local', pretrained=False)
                xfeat.net.load_state_dict(torch.load(repo / 'weights/xfeat.pt', map_location=device, weights_only=True))
                self.model = tiny_roma_v1_outdoor(device=device, xfeat=xfeat.net)
        elif self.name == 'mast3r':
            from mast3r.model import AsymmetricMASt3R
            self.model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').eval().to(device)

    @torch.inference_mode()
    def extract(self, gray):
        small, scale = resize_to_max_dim(gray, self.config['feature_max_dim'])
        fs = FeatureSet(np.empty((0, 2), np.float32), np.empty((0, 0)), np.empty(0), gray.shape, scale, small)
        tensor = torch.from_numpy(small.copy()).float().to(self.device)[None, None] / 255
        fs.tensor = tensor
        fs.payload = None
        if self.name.endswith('-lightglue'):
            fs.payload = self.model.extract(tensor[0].expand(3, -1, -1), resize=None)
            fs.keypoints = fs.payload['keypoints'][0].cpu().numpy() / scale
        elif self.name in ('xfeat', 'xfeat-lighterglue'):
            fs.payload = self.model.detectAndCompute(tensor)[0]
            fs.payload['image_size'] = (small.shape[1], small.shape[0])
            fs.keypoints = fs.payload['keypoints'].cpu().numpy() / scale
        return fs

    @torch.inference_mode()
    def match(self, a, b):
        n = self.config['max_keypoints']
        if self.name.endswith('-lightglue'):
            out = self.matcher({'image0': a.payload, 'image1': b.payload})
            idx = out['matches'][0].cpu().numpy()
            return a.keypoints[idx[:, 0]], b.keypoints[idx[:, 1]], out['scores'][0].cpu().numpy()
        if self.name in ('xfeat', 'xfeat-lighterglue') and (not len(a.keypoints) or not len(b.keypoints)):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        if self.name == 'xfeat':
            i, j = self.model.match(a.payload['descriptors'], b.payload['descriptors'])
            return a.keypoints[i.cpu().numpy()], b.keypoints[j.cpu().numpy()], np.ones(len(i))
        if self.name == 'xfeat-lighterglue':
            x, y, _ = self.model.match_lighterglue(a.payload, b.payload)
            return x / a.scale, y / b.scale, np.ones(len(x))
        if self.name == 'xfeat-star':
            x, y = self.model.match_xfeat_star(a.tensor, b.tensor, top_k=n)
            return x / a.scale, y / b.scale, np.ones(len(x))
        if self.name.startswith('loftr'):
            # LoFTR requires spatial dimensions divisible by eight.
            tensors, factors = [], []
            for f in (a, b):
                h, w = f.small_gray.shape
                hh, ww = max(8, h // 8 * 8), max(8, w // 8 * 8)
                tensors.append(torch.nn.functional.interpolate(f.tensor, (hh, ww), mode='bilinear', align_corners=False))
                factors.append(np.array([f.shape[1] / ww, f.shape[0] / hh]))
            out = self.model({'image0': tensors[0], 'image1': tensors[1]})
            return out['keypoints0'].cpu().numpy() * factors[0], out['keypoints1'].cpu().numpy() * factors[1], out['confidence'].cpu().numpy()
        if self.name in ('roma', 'tiny-roma'):
            from PIL import Image
            kwargs = {'device': self.device} if self.name == 'roma' else {}
            warp, certainty = self.model.match(Image.fromarray(a.small_gray).convert('RGB'), Image.fromarray(b.small_gray).convert('RGB'), **kwargs)
            matches, scores = self.model.sample(warp, certainty, num=n)
            x, y = self.model.to_pixel_coordinates(matches, *a.shape, *b.shape)
            return x.cpu().numpy(), y.cpu().numpy(), scores.cpu().numpy()
        if self.name == 'mast3r':
            import mast3r.utils.path_to_dust3r  # noqa: F401
            from dust3r.inference import inference
            from mast3r.fast_nn import fast_reciprocal_NNs
            views, factors = [], []
            for i, f in enumerate((a, b)):
                h, w = f.shape
                scale = min(512 / max(h, w), 1)
                hh, ww = max(16, round(h * scale / 16) * 16), max(16, round(w * scale / 16) * 16)
                t = torch.nn.functional.interpolate(f.tensor, (hh, ww), mode='bilinear', align_corners=False).expand(-1, 3, -1, -1)
                views.append({'img': t * 2 - 1, 'true_shape': np.array([[hh, ww]], np.int32), 'idx': i, 'instance': str(i)})
                factors.append(np.array([w / ww, h / hh]))
            out = inference([tuple(views)], self.model, self.device, batch_size=1, verbose=False)
            x, y = fast_reciprocal_NNs(out['pred1']['desc'][0], out['pred2']['desc'][0], subsample_or_initxy1=8, device=self.device, dist='dot', block_size=8192)
            return x * factors[0], y * factors[1], np.ones(len(x))
        raise ValueError(self.name)
