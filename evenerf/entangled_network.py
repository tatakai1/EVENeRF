import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .condition_nerf import CondNeRF
from .base_network import MultiHeadAttention, ConvAutoEncoder

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

class View_Cross_Transformer(nn.Module):
    def __init__(self, n_head, d_model,  dropout=0.1, D=2):
        super(View_Cross_Transformer, self).__init__()

        self.posediff_linear = nn.Sequential(nn.Linear(12, d_model), nn.ReLU())
        self.cross_transformers = nn.ModuleList([MultiHeadAttention(n_head, d_model, 
                                                             d_model//n_head, 
                                                             d_model//n_head, dropout) for _ in range(D)])

    def forward(self, ray_query, pose_diff):
        pos = self.posediff_linear(pose_diff)
        q = ray_query + pos 
        for cross_trans in self.cross_transformers:
            q, _, _ = cross_trans(q, q, q)
        out = torch.sigmoid(q)
        return out
    
 

class View_Dual_Transformer(nn.Module):
    def __init__(self, args, posenc_dim=3, viewenc_dim=3, dropout=0.1, d_pose=4):
        super(View_Dual_Transformer, self).__init__()
        n_head = args.head
        d_model = args.netwidth
        n_samples = args.N_samples
        self.view_transformer = MultiHeadAttention(n_head, d_model, 
                                                   d_model//n_head, d_model//n_head, 
                                                   dropout, d_pose)
        self.epipolar_interaction_net = ConvAutoEncoder(d_model*2+posenc_dim+viewenc_dim, d_model, n_samples)

    def forward(self, feats, mask, ray_diff, pts):
        B, N, V, _ = feats.shape
        feats, mask, ray_diff = feats.flatten(0, 1), mask.flatten(0, 1), ray_diff.flatten(0, 1)
        view_agg_feats, view_attn, view_V =  self.view_transformer(feats, feats, feats, mask, ray_diff) # (B*N, V, C)
        view_agg_feats = view_agg_feats.reshape(B, N, *view_agg_feats.shape[1:])
        view_attn = view_attn.reshape(B, N, *view_attn.shape[1:])
        view_V = view_V.reshape(B, N, *view_V.shape[1:])

        V = torch.cat(torch.var_mean(view_V, dim=-2), dim=-1) # (B, N, 2C)
        V = V.transpose(1, 2).contiguous()  # (B, 2C, N)
        pts = pts.transpose(1, 2).contiguous()  # (B, C, N)
        
        epipolar_interaction_map = self.epipolar_interaction_net(V, pts).transpose(1, 2).contiguous().unsqueeze(-2)  # (B, N, 1, C)

        interaction_feats = view_agg_feats*epipolar_interaction_map
        return interaction_feats, view_attn, epipolar_interaction_map
    
class Epipolar_Dual_Transformer(nn.Module):
    def __init__(self, args, dropout=0.1, d_pose=4):
        super(Epipolar_Dual_Transformer, self).__init__()
        n_head = args.head
        d_model = args.netwidth
        self.epipolar_transformer = MultiHeadAttention(n_head, d_model, 
                                                   d_model//n_head, d_model//n_head, 
                                                   dropout, d_pose)
        self.view_interaction_net = View_Cross_Transformer(n_head, d_model, 
                                                            dropout, args.cross_depth)

    def forward(self, feats, mask, ray_diff, pose_diff):
        B, V, N, _ = feats.shape
        feats, mask, ray_diff = feats.flatten(0, 1), mask.flatten(0, 1), ray_diff.flatten(0, 1)
        epipolar_agg_feats, epipolar_attn, epipolar_V =  self.epipolar_transformer(feats, feats, feats, mask, ray_diff) # (B*V, N, C)
        epipolar_agg_feats = epipolar_agg_feats.reshape(B, V, *epipolar_agg_feats.shape[1:])
        epipolar_attn = epipolar_attn.reshape(B, V, *epipolar_attn.shape[1:])
        
        Q = epipolar_V.max(dim=-2, keepdim=True)[0].reshape(B, V, -1) # B*V, C
        view_interaction_map = self.view_interaction_net(Q, pose_diff)
        view_interaction_map = view_interaction_map.reshape(B, V, 1, -1)    

        interaction_feats =  epipolar_agg_feats*view_interaction_map
        return interaction_feats, epipolar_attn, view_interaction_map



class Entangled_Net(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3):
        super(Entangled_Net, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        self.view_dual_trans = nn.ModuleList([])
        self.epipolar_dual_trans = nn.ModuleList([])
        for i in range(args.trans_depth):
            # view transformer
            view_dual_trans = View_Dual_Transformer(args, posenc_dim, viewenc_dim)
            self.view_dual_trans.append(view_dual_trans)
            # ray transformer
            epipolar_dual_trans = Epipolar_Dual_Transformer(args)
            self.epipolar_dual_trans.append(epipolar_dual_trans)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.cond_nerf = CondNeRF(args, posenc_dim, viewenc_dim)

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d, pose_diff):
        B, N, V, _ = rgb_feat.shape
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)

        # project rgb features to netwidth
        feats = self.rgbfeat_fc(rgb_feat) # [B, N, V, C]
        ray_diff_ = ray_diff.transpose(1, 2)
        mask_ = mask.transpose(1, 2)

        # transformer modules
        for i, (view_trans, epipolar_trans) in enumerate(
            zip(self.view_dual_trans, self.epipolar_dual_trans)
        ):  
            feats_raw = feats
            # view dual transformer to update q
            feats, _, _ = view_trans(feats, mask, ray_diff, embed)
            feats_ = feats.transpose(1, 2).contiguous()
            # epipolar dual transformer to update q
            feats_, _, _ = epipolar_trans(feats_, mask_, ray_diff_, pose_diff)
            feats = feats_.transpose(1, 2).contiguous() + feats_raw

        agg_feats = feats.mean(dim=2) # (B, N, C)

        rgb, alpha = self.cond_nerf(input_pts, input_views, agg_feats, mask)

        return rgb, alpha



