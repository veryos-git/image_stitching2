"""SuperPoint + SuperGlue models.

Faithful PyTorch re-implementation of the networks from

    https://github.com/magicleap/SuperGluePretrainedNetwork

(Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural
Networks", CVPR 2020) updated for modern PyTorch 2.x:

* weights are loaded with ``weights_only=False`` (the checkpoints are legacy
  pickle files) and always mapped to CPU;
* the checkpoint path is injected explicitly instead of being hard-coded;
* inference always runs under ``torch.no_grad()``.
"""

from __future__ import annotations

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# SuperPoint
# --------------------------------------------------------------------------- #
def _simple_nms(scores, nms_radius: int):
    """Fast non-maximum suppression to remove nearby points."""
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def _remove_borders(keypoints, scores, border, height, width):
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def _top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def _sample_descriptors(keypoints, descriptors, s=8):
    """Interpolate descriptors at (x, y) keypoint locations."""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)]).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2),
        mode='bilinear', align_corners=True)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
        'remove_borders': 4,
    }

    def __init__(self, config=None, weights_path=None):
        super().__init__()
        self.config = {**self.default_config, **(config or {})}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)

        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)

        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'], 1, 1, 0)

        if weights_path is not None:
            state = torch.load(weights_path, map_location='cpu',
                               weights_only=False)
            self.load_state_dict(state)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or -1')

    @torch.no_grad()
    def forward(self, image):
        """``image``: [1, 1, H, W] float tensor in [0, 1]."""
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Dense keypoint scores.
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = _simple_nms(scores, self.config['nms_radius'])

        keypoints = [torch.nonzero(s > self.config['keypoint_threshold'])
                     for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        keypoints, scores = list(zip(*[
            _remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                _top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # (h, w) -> (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        descriptors = [_sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints[0],
            'scores': scores[0],
            'descriptors': descriptors[0],
        }


# --------------------------------------------------------------------------- #
# SuperGlue
# --------------------------------------------------------------------------- #
def _mlp(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], 1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _normalize_keypoints(kpts, image_shape):
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class _KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = _mlp([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def _attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class _MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, 1)
        self.proj = nn.ModuleList([self._copy(self.merge) for _ in range(3)])

    @staticmethod
    def _copy(m):
        import copy
        return copy.deepcopy(m)

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = _attention(query, key, value)
        return self.merge(x.contiguous().view(
            batch_dim, self.dim * self.num_heads, -1))


class _AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = _MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = _mlp([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class _AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList([
            _AttentionalPropagation(feature_dim, 4) for _ in layer_names])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = desc0 + delta0, desc1 + delta1
        return desc0, desc1


def _log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def _log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = _log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm
    return Z


def _arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


class SuperGlue(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config=None, weights_path=None):
        super().__init__()
        self.config = {**self.default_config, **(config or {})}

        self.kenc = _KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = _AttentionalGNN(self.config['descriptor_dim'],
                                   self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        self.register_parameter('bin_score', nn.Parameter(torch.tensor(1.)))

        if weights_path is not None:
            state = torch.load(weights_path, map_location='cpu',
                               weights_only=False)
            self.load_state_dict(state)

    @torch.no_grad()
    def forward(self, kpts0, kpts1, desc0, desc1, scores0, scores1,
                shape0, shape1):
        """Match two sets of SuperPoint features.

        ``kpts*``: [N, 2] (x, y); ``desc*``: [N, D]; ``scores*``: [N];
        ``shape*``: ``(1, 1, H, W)`` tensors-like tuple of the source image.

        Returns ``(matches0, mscores0)`` where ``matches0[i]`` is the index of
        the matching keypoint in image 1 (or ``-1``) and ``mscores0`` the
        confidence of that assignment.
        """
        if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
            return (kpts0.new_full((kpts0.shape[0],), -1, dtype=torch.long),
                    kpts0.new_zeros((kpts0.shape[0],)))

        # Normalize keypoint locations (encoder expects [B, N, 2]).
        kpts0n = _normalize_keypoints(kpts0[None], shape0)
        kpts1n = _normalize_keypoints(kpts1[None], shape1)

        desc0 = desc0[None].transpose(1, 2) + self.kenc(kpts0n, scores0[None])
        desc1 = desc1[None].transpose(1, 2) + self.kenc(kpts1n, scores1[None])

        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        scores = _log_optimal_transport(
            scores, self.bin_score, self.config['sinkhorn_iterations'])

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = _arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = _arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0[0], mscores0[0]
