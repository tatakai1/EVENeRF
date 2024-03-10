import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention, modified from IBRNet. '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, pose=None):
        if pose is not None:
            attn = torch.matmul((q + pose) / self.temperature, k.transpose(2, 3))
        else:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9 if attn.dtype == torch.float32 else -1e4)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module, modified from IBRNet. '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, d_pose=None):
        super().__init__()
        assert n_head*d_v == d_model

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        if d_pose is not None:
            self.w_pose = nn.Linear(d_pose, n_head * d_k, bias=False)
        self.fc = FeedForward(d_model*4, d_model, dropout)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, pose=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.attn_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v)
        out_v = v
        v = v.view(sz_b, len_v, n_head, d_v)
        if pose is not None:
            pose = self.w_pose(pose).view(sz_b, len_q, n_head, d_k)
            pose = pose.transpose(1, 2)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask, pose=pose)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = q + residual

        # q = self.dropout(self.fc(q))
        residual = q
        q = self.ff_norm(q)
        q = self.fc(q)
        q += residual

        return q, attn, out_v


class FeedForward(nn.Module):
    def __init__(self, hid_dim, dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x
    
## Auto-encoder network
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_ch, num_ch, S):
        super(ConvAutoEncoder, self).__init__()
        # condition input mlp
        self.input_mlp = nn.Sequential(nn.Conv1d(input_ch, num_ch, 1),
                                       nn.LayerNorm(S, elementwise_affine=False),
                                       nn.ELU(alpha=1.0, inplace=True),
                                       )
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Conv1d(num_ch+input_ch, num_ch, 3, stride=1, padding=1)
        

    def forward(self, x, cond):
        input = torch.cat([x, cond], dim=-2)
        x = self.input_mlp(input)
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        out = torch.sigmoid(x)

        return out