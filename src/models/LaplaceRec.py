# -*- coding: UTF-8 -*-
# @Author  : ReChorus User (Laplace Transform Variant)

import torch
import torch.nn as nn
import numpy as np
import math

from models.BaseModel import SequentialModel

class LaplaceRec(SequentialModel):
    """
    Variant of BSARec using Discrete Cosine Transform (DCT) 
    which corresponds to the spectrum of the Graph Laplacian for sequences.
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    # 更新 log 参数，保持与 BSARec 一致
    extra_log_args = ['alpha', 'c', 'beta_init', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--alpha', type=float, default=0.7, help='Weight for Inductive Bias (0<=alpha<=1).')
        parser.add_argument('--c', type=int, default=5, help='Low-frequency cutoff.')
        parser.add_argument('--beta_init', type=float, default=0.0, help='Init value for beta.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        
        # 自动适配 history_len
        if hasattr(args, 'history_len'):
            self.max_len = args.history_len
        elif hasattr(args, 'max_his'):
            self.max_len = args.max_his
        else:
            self.max_len = 20

        self.alpha = args.alpha
        self.c = args.c
        
        # 1. Embeddings (padding_idx=0 极其重要)
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_len, self.emb_size)
        
        # 2. Initial Norm & Dropout
        self.emb_layer_norm = nn.LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        # 3. Encoder Layers (使用 LaplaceLayer)
        self.layers = nn.ModuleList([
            LaplaceLayer(self.emb_size, args.num_heads, self.emb_size * 4, args.dropout, 
                     self.alpha, self.c, args.beta_init, self.max_len)
            for _ in range(args.num_layers)
        ])
        
        # 4. Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.emb_size, eps=1e-12)

        self.apply(self.init_weights)

    def forward(self, feed_dict):
        history = feed_dict['history_items']  # [batch, seq_len]
        
        # Padding Mask
        mask = (history != 0).float().unsqueeze(-1)
        
        # Embedding
        seq_emb = self.item_embeddings(history)
        positions = torch.arange(history.shape[1], dtype=torch.long, device=history.device)
        positions = positions.unsqueeze(0).expand_as(history)
        if positions.shape[1] > self.max_len:
             positions = positions[:, :self.max_len]
        pos_emb = self.position_embeddings(positions)
        
        x = seq_emb + pos_emb
        x = self.emb_layer_norm(x)
        x = self.dropout(x)
        x = x * mask 

        # Attention Masks
        seq_len = history.shape[1]
        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=history.device))
        key_padding_mask = (history == 0)

        # Encoder Loop
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = x * mask

        # Final LayerNorm
        x = self.final_layer_norm(x)

        # 预测逻辑 (同 BSARec)
        valid_len = (history != 0).sum(dim=1)
        last_index = (valid_len - 1).clamp(min=0)
        batch_size = x.shape[0]
        gather_index = last_index.view(batch_size, 1, 1).expand(-1, -1, self.emb_size)
        user_emb = x.gather(1, gather_index).squeeze(1) 

        if 'item_id' in feed_dict:
            item_ids = feed_dict['item_id']
            item_embs = self.item_embeddings(item_ids) 
            prediction = (user_emb.unsqueeze(1) * item_embs).sum(dim=-1)
        else:
            prediction = user_emb
        
        return {'prediction': prediction}


class LaplaceLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, alpha, c, beta_init, max_len):
        super(LaplaceLayer, self).__init__()
        self.alpha = alpha
        
        # Branch 1: Self-Attention (保持不变)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(d_model)

        # Branch 2: Inductive Bias (改为使用 LaplaceRescaler)
        self.inductive_bias = LaplaceRescaler(d_model, c, beta_init, max_len)
        self.norm_ib = nn.LayerNorm(d_model)

        # FFN (保持不变)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        # 1. SA Branch
        sa_output, _ = self.self_attn(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        sa_part = self.norm_sa(src + self.dropout_sa(sa_output))
        
        # 2. IB Branch (Laplace/DCT)
        ib_output = self.inductive_bias(src)
        ib_part = self.norm_ib(src + ib_output)
        
        # 3. Mix
        mixed_output = self.alpha * ib_part + (1 - self.alpha) * sa_part

        # 4. FFN
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(mixed_output))))
        output = self.norm_ffn(mixed_output + self.dropout_ffn(ff_output))
        
        return output


class LaplaceRescaler(nn.Module):
    """
    Replaces FFT with Discrete Cosine Transform (DCT).
    DCT is the spectral basis of the Graph Laplacian for a path graph (sequence).
    """
    def __init__(self, d_model, c, beta_init, max_len):
        super(LaplaceRescaler, self).__init__()
        self.c = c
        self.max_len = max_len
        # DCT 是实数变换，所以 beta 也是实数缩放
        self.beta = nn.Parameter(torch.tensor(beta_init))
        
        # 预计算 DCT 变换矩阵 (N, N)
        self.register_buffer('dct_matrix', self._get_dct_matrix(max_len))
        self.register_buffer('idct_matrix', self._get_dct_matrix(max_len).t()) # DCT 是正交的，逆变换即转置

    def _get_dct_matrix(self, N):
        # 实现 DCT-II 类型
        dct_m = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0:
                    coef = np.sqrt(1.0 / N)
                else:
                    coef = np.sqrt(2.0 / N)
                dct_m[k, n] = coef * np.cos(np.pi / N * (n + 0.5) * k)
        return torch.from_numpy(dct_m).float()

    def forward(self, x):
        # x: [batch, seq_len, emb_size]
        batch, seq_len, emb = x.shape
        
        # 如果当前 batch 的 seq_len 小于 max_len (例如测试时的动态长度)，
        # 我们需要切片使用 DCT 矩阵，或者重新生成。
        # 考虑到效率和通常 padding 到固定长度，这里假设 seq_len <= max_len
        # 如果 seq_len 变动，建议动态生成或 padding。
        # 为了稳健性，这里根据当前 seq_len 动态生成/切片处理稍微复杂，
        # 我们假设输入已经被 padding 到了 max_len 或者我们只处理前 seq_len 部分。
        
        if seq_len != self.max_len:
             # 动态生成以适应变长序列 (Fallback)
             dct_m = self._get_dct_matrix(seq_len).to(x.device)
             idct_m = dct_m.t()
        else:
             dct_m = self.dct_matrix
             idct_m = self.idct_matrix

        # 1. DCT 变换 (Time -> Frequency/Laplace Domain)
        # x: [B, L, E] -> permute -> [B, E, L] 以便与 matrix [L, L] 相乘
        # 或者直接 x [B, L, E], 变换矩阵在左侧乘: D @ x_vector
        # 这里的矩阵乘法：DCT [L, L] @ x [B, L, E] -> 结果维度 [B, L, E]
        # 需要将 L 维度放在前面或使用 matmul 广播
        
        # 变换公式: X_k = sum_n (D_kn * x_n)
        # 使用 einsum: 'lk, bke -> ble' (l: freq, k: time)
        # dct_matrix: [L_freq, L_time]
        dct_x = torch.einsum('ij, bje -> bie', dct_m, x)
        
        # 2. 滤波 (Filtering)
        # dct_x 维度 [batch, seq_len, emb]
        # 维度 1 是频率维度，0 是低频 (DC)，L-1 是高频
        
        cutoff = min(self.c, seq_len)
        lfc = dct_x[:, :cutoff, :]
        hfc = dct_x[:, cutoff:, :]
        
        # 3. 缩放高频 (Scale High Frequencies)
        hfc_scaled = hfc * self.beta
        
        # 4. 拼接 (Combine)
        dct_combined = torch.cat([lfc, hfc_scaled], dim=1)
        
        # 5. IDCT 逆变换 (Frequency -> Time)
        # idct_matrix: [L_time, L_freq]
        output = torch.einsum('ij, bje -> bie', idct_m, dct_combined)
        
        return output