import torch
import torch.nn as nn
from .base_network import MultiHeadAttention
import torch.nn.functional as F
import numpy as np


class CondNeRF(nn.Module):
    """Conditional NeRF; take (position, image_features) as input, output rgb images."""

    def __init__(self, args, posenc_dim, viewenc_dim):
        super(CondNeRF, self).__init__()
        self.define_network(args, posenc_dim, viewenc_dim)

    def define_network(self, args, posenc_dim=3, viewenc_dim=3):
        W = args.nerf_width
        D = args.nerf_depth
        input_ch_feat = posenc_dim
        input_3D_dim = viewenc_dim

        self.skip = args.nerf_skip
        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_3D_dim, W, bias=True)] +
            [torch.nn.Linear(W, W, bias=True) if i not in self.skip else
             torch.nn.Linear(W + input_3D_dim, W) for i in range(D - 1)])
        self.pts_bias = torch.nn.Linear(args.netwidth, W)

        raytrans_act = torch.nn.ReLU(inplace=True)
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(viewenc_dim + W, W // 2)])
        self.alpha_linear = torch.nn.Sequential(torch.nn.Linear(W, 16), raytrans_act)
        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_alpha_linear = torch.nn.Sequential(torch.nn.Linear(16, 16),
                                                    raytrans_act,
                                                    torch.nn.Linear(16, 1),
                                                    torch.nn.ReLU(inplace=True))

        self.feature_linear = torch.nn.Linear(W, W)
        self.rgb_linear = torch.nn.Linear(W // 2, 3)


        self.pts_linears.apply(self.weights_init)
        self.views_linears.apply(self.weights_init)
        self.feature_linear.apply(self.weights_init)
        self.alpha_linear.apply(self.weights_init)
        self.rgb_linear.apply(self.weights_init)

    def forward(self, input_pts, input_views, cond, mask):
        h = input_pts
        bias = self.pts_bias(cond)
        for i, l in enumerate(self.pts_linears):
            h = l(h) * bias
            h = F.relu(h)
            if i in self.skip:
                h = torch.cat([input_pts, h], -1)

        raw_alpha = self.alpha_linear(h)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=-2)  # [n_rays, n_samples, 1]
        alpha, _, _ = self.ray_attention(raw_alpha, raw_alpha, raw_alpha,
                                        mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        alpha = self.out_alpha_linear(alpha)

        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = l(h)
            h = F.relu(h)
        rgb = torch.sigmoid(self.rgb_linear(h))  # [n_rays, n_samples, 3]

        return rgb, alpha

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

    def composite(self, ray, rgb_samples, density_samples, depth_samples, setbg_opaque, wo_render_interval=False):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0]-depth_samples[..., :-1, 0]  # [HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)], dim=1)  # [HW,N]
        dist_samples = depth_intv_samples*ray_length  # [HW,N]
        if wo_render_interval:
            # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
            # very different scales, and using interval can affect the model's generalization ability.
            # Therefore we don't use the intervals for both training and evaluation. [IBRNet]
            sigma_delta = density_samples.squeeze(-1)
        else:
            sigma_delta = density_samples.squeeze(-1)*dist_samples  # [HW,N]

        alpha = 1-(-sigma_delta).exp_()  # [HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=1).cumsum(dim=1)).exp_()  # [HW,N]
        prob = (T*alpha)[..., None]  # [HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=1)  # [HW,1]
        rgb = (rgb_samples*prob).sum(dim=1)  # [HW,3]
        opacity = prob.sum(dim=1)  # [HW,1]
        if setbg_opaque:
            rgb = rgb + 1 * (1 - opacity)
        return rgb, depth, opacity, prob  # [HW,K]
