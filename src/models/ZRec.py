import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel

class ZRec(SequentialModel):
    """
    Variant of BSARec using Z-Transform (Generalized DFT).
    It introduces a radius 'gamma' to evaluate the spectrum off the unit circle.
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['alpha', 'c', 'beta_init', 'gamma_init', 'fix_gamma', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--alpha', type=float, default=0.7, help='Weight for Inductive Bias (0<=alpha<=1).')
        parser.add_argument('--c', type=int, default=5, help='Low-frequency cutoff.')
        parser.add_argument('--beta_init', type=float, default=0.0, help='Init value for beta.')
        
        parser.add_argument('--gamma_init', type=float, default=1.0, help='Init value for gamma (Z-plane radius).')
        parser.add_argument('--fix_gamma', type=int, default=0, help='Whether to fix gamma (1) or make it learnable (0).')
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        
        if hasattr(args, 'history_len'):
            self.max_len = args.history_len
        elif hasattr(args, 'max_his'):
            self.max_len = args.max_his
        else:
            self.max_len = 20

        self.alpha = args.alpha
        self.c = args.c
        
        # Embeddings
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_len, self.emb_size)
        
        # Norm & Dropout
        self.emb_layer_norm = nn.LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        # Encoder Layers
        self.layers = nn.ModuleList([
            ZLayer(self.emb_size, args.num_heads, self.emb_size * 4, args.dropout, 
                   self.alpha, self.c, args.beta_init, args.gamma_init, args.fix_gamma, self.max_len)
            for _ in range(args.num_layers)
        ])
        
        # Final Norm
        self.final_layer_norm = nn.LayerNorm(self.emb_size, eps=1e-12)

        self.apply(self.init_weights)

    def forward(self, feed_dict):
        history = feed_dict['history_items']
        
        mask = (history != 0).float().unsqueeze(-1)
        
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

        seq_len = history.shape[1]
        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=history.device))
        key_padding_mask = (history == 0)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = x * mask

        x = self.final_layer_norm(x)

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


class ZLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, alpha, c, beta_init, gamma_init, fix_gamma, max_len):
        super(ZLayer, self).__init__()
        self.alpha = alpha
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(d_model)

        self.inductive_bias = ZRescaler(d_model, c, beta_init, gamma_init, fix_gamma, max_len)
        self.norm_ib = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        sa_output, _ = self.self_attn(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        sa_part = self.norm_sa(src + self.dropout_sa(sa_output))
        
        ib_output = self.inductive_bias(src)
        ib_part = self.norm_ib(src + ib_output)
        
        mixed_output = self.alpha * ib_part + (1 - self.alpha) * sa_part

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(mixed_output))))
        output = self.norm_ffn(mixed_output + self.dropout_ffn(ff_output))
        
        return output


class ZRescaler(nn.Module):
    def __init__(self, d_model, c, beta_init, gamma_init, fix_gamma, max_len):
        super(ZRescaler, self).__init__()
        self.c = c
        self.beta = nn.Parameter(torch.tensor(beta_init))
        
        gamma_tensor = torch.tensor(gamma_init, dtype=torch.float32)
        
        if fix_gamma == 1:
            # buffer
            self.register_buffer('gamma', gamma_tensor)
        else:
            # Parameter (into optimization)
            self.gamma = nn.Parameter(gamma_tensor)
        
    def forward(self, x):
        seq_len = x.shape[1]
        device = x.device

        n = torch.arange(seq_len, device=device, dtype=x.dtype)
        
        
        gamma_val = self.gamma
        if gamma_val <= 1e-6:
             gamma_val = gamma_val + 1e-6

        # weights = gamma^(-n), gamma_val (Scalar) -> pow -> weights (1, L, 1)
        weights = torch.pow(gamma_val, -n).view(1, -1, 1)
        inverse_weights = torch.pow(gamma_val, n).view(1, -1, 1)

        x_weighted = x * weights


        fft_x = torch.fft.rfft(x_weighted, dim=1, norm='ortho') 
        
        cutoff = min(self.c, fft_x.size(1))
        lfc = fft_x[:, :cutoff, :]
        hfc = fft_x[:, cutoff:, :]
        hfc_scaled = hfc * self.beta
        fft_combined = torch.cat([lfc, hfc_scaled], dim=1)
        
        
        output_weighted = torch.fft.irfft(fft_combined, n=seq_len, dim=1, norm='ortho')
        output = output_weighted * inverse_weights
        
        return output
    
    