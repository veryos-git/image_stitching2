"""MatchAnything ELoFTR using the authors' cross-modality weights and source."""
from pathlib import Path
import threading
import numpy as np
import torch
from PIL import Image
from .matching import FeatureSet

ROOT = Path(__file__).resolve().parents[1]


class MatchAnythingEngine:
    name = 'matchanything-eloftr'

    def __init__(self, config, device):
        self.device = device
        self.max_dim = config.get('feature_max_dim', 832)
        checkpoint = ROOT / 'weights/matchanything_eloftr.ckpt'
        if not checkpoint.is_file() or not (ROOT/'vendor/MatchAnything/src/loftr/loftr.py').is_file():
            raise RuntimeError('Install MatchAnything with .venv/bin/python setup_matchanything.py')
        from vendor.MatchAnything.src.loftr import LoFTR
        from vendor.MatchAnything.configs.models.eloftr_model import cfg
        model_config = cfg.clone()
        model_config.LOFTR.COARSE.NPE = [832, 832, self.max_dim, self.max_dim]
        def lower(node):
            return {k.lower(): lower(v) for k,v in node.items()} if hasattr(node, 'items') else node
        self.model = LoFTR(lower(model_config.LOFTR))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.eval().to(device)
        self.lock = threading.Lock()

    def extract(self, gray):
        h,w = gray.shape
        scale = min(1., self.max_dim / max(h,w))
        hh,ww = max(32,int(h*scale)//32*32),max(32,int(w*scale)//32*32)
        small = np.array(Image.fromarray(gray).resize((ww,hh), Image.Resampling.LANCZOS))
        return FeatureSet(np.empty((0,2),np.float32),np.empty((0,0),np.float32),np.empty(0),gray.shape,scale,small)

    @torch.inference_mode()
    def match(self, a, b):
        # The published wrapper square-pads inputs and masks the padded regions.
        batch = {}
        factors = []
        side = max(*a.small_gray.shape,*b.small_gray.shape)
        for i,fs in enumerate((a,b)):
            h,w = fs.small_gray.shape
            tensor = torch.zeros((1,1,side,side),device=self.device)
            tensor[0,0,:h,:w] = torch.from_numpy(fs.small_gray).to(self.device,dtype=torch.float32)/255.
            mask = torch.zeros((1,side//8,side//8),device=self.device,dtype=torch.bool)
            mask[:,:h//8,:w//8] = True
            batch[f'image{i}'],batch[f'mask{i}'] = tensor,mask
            factors.append(np.array([fs.shape[1]/w,fs.shape[0]/h],np.float32))
        with self.lock:
            self.model(batch)
        x = batch['mkpts0_f'].cpu().numpy()*factors[0]
        y = batch['mkpts1_f'].cpu().numpy()*factors[1]
        confidence = batch['mconf'].cpu().numpy()
        valid = np.isfinite(x).all(1)&np.isfinite(y).all(1)&np.isfinite(confidence)
        for points,fs in ((x,a),(y,b)):
            valid &= (points>=0).all(1)&(points[:,0]<fs.shape[1])&(points[:,1]<fs.shape[0])
        return x[valid],y[valid],confidence[valid]
