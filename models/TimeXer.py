import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MultiScaleEnEmbedding(nn.Module):
    """Multi-scale patch embedding for M1 module"""
    def __init__(self, n_vars, d_model, patch_sizes, seq_len, dropout):
        super(MultiScaleEnEmbedding, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Create embeddings for each patch size
        self.patch_embeddings = nn.ModuleDict()
        self.patch_nums = {}
        
        for patch_size in patch_sizes:
            if patch_size > 0:
                patch_num = seq_len // patch_size
                self.patch_nums[str(patch_size)] = patch_num
                self.patch_embeddings[str(patch_size)] = nn.Linear(patch_size, d_model, bias=False)
        
        # Single shared global token (more parameter efficient)
        self.shared_glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        
        # Scale-specific adaptation layers for the global token
        self.scale_adapters = nn.ModuleDict()
        for patch_size in patch_sizes:
            self.scale_adapters[str(patch_size)] = nn.Linear(d_model, d_model, bias=True)
        
        # Position embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Fusion mechanism - learnable weights for combining different scales
        self.scale_fusion = nn.Linear(len(patch_sizes) * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total patch numbers for head computation
        self.total_patch_num = sum(self.patch_nums.values()) + len(patch_sizes)  # +len for global tokens
        
    def forward(self, x):
        # x shape: [B, C, L]
        n_vars = x.shape[1]
        batch_size = x.shape[0]
        
        scale_embeddings = []
        
        for patch_size in self.patch_sizes:
            if patch_size > 0:
                # Patch the input with current scale
                x_patched = x.unfold(dimension=-1, size=patch_size, step=patch_size)
                # x_patched: [B, C, patch_num, patch_size]
                
                # Reshape for embedding
                x_reshaped = torch.reshape(x_patched, (batch_size * n_vars, x_patched.shape[2], x_patched.shape[3]))
                
                # Apply embedding
                embedded = self.patch_embeddings[str(patch_size)](x_reshaped) + self.position_embedding(x_reshaped)
                
                # Reshape back
                embedded = torch.reshape(embedded, (batch_size, n_vars, embedded.shape[-2], embedded.shape[-1]))
                
                # Adapt shared global token for this scale
                adapted_glb = self.scale_adapters[str(patch_size)](self.shared_glb_token)
                adapted_glb = adapted_glb.repeat((batch_size, 1, 1, 1))
                
                embedded_with_glb = torch.cat([embedded, adapted_glb], dim=2)
                
                scale_embeddings.append(embedded_with_glb)
        
        # Concatenate all scales along the patch dimension
        if len(scale_embeddings) > 1:
            # For fusion, we need to handle different patch numbers
            # Let's use a simple concatenation approach first
            combined_embedding = torch.cat(scale_embeddings, dim=2)  # Concat along patch dimension
        else:
            combined_embedding = scale_embeddings[0]
        
        # Reshape for encoder input
        final_embedding = torch.reshape(combined_embedding, 
                                      (combined_embedding.shape[0] * combined_embedding.shape[1], 
                                       combined_embedding.shape[2], combined_embedding.shape[3]))
        
        return self.dropout(final_embedding), n_vars


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        if self.patch_len > 0:
            # do patching
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        else:
            # 不做 patch，保持 shape: (B, C, L)
            x = x.unsqueeze(2)  # 轉成 (B, C, 1, L) 來模擬 patch 結果

        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        
        # Multi-scale patch support
        self.use_multi_scale = getattr(configs, 'use_multi_scale', False)
        if self.use_multi_scale:
            self.patch_sizes = getattr(configs, 'patch_sizes', [8, 16, 24])
        
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # Embedding - choose between single scale or multi-scale
        if self.use_multi_scale:
            self.en_embedding = MultiScaleEnEmbedding(
                self.n_vars, configs.d_model, self.patch_sizes, configs.seq_len, configs.dropout
            )
            # Update patch_num to use the total from multi-scale embedding
            self.patch_num = self.en_embedding.total_patch_num
        else:
            self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
            # Fix patch_num calculation for single scale
            if self.patch_len != 0:
                self.patch_num = int(configs.seq_len // self.patch_len) + 1  # +1 for global token
            else:
                self.patch_num = 1  # Only global token

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None