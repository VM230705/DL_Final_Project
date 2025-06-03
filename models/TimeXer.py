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


class ScaleAwareAttentionFusion(nn.Module):
    """Scale-aware attention fusion with per-scale attention patterns"""
    def __init__(self, d_model, patch_sizes):
        super().__init__()
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        
        # Per-scale self-attention modules
        self.scale_attentions = nn.ModuleDict()
        for patch_size in patch_sizes:
            self.scale_attentions[str(patch_size)] = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=0.1, batch_first=True
            )
        
        # Cross-scale interaction attention
        self.cross_scale_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Learnable scale importance weights
        self.scale_importance = nn.Parameter(torch.ones(len(patch_sizes)))
        
        # Layer normalization for each scale
        self.scale_norms = nn.ModuleDict()
        for patch_size in patch_sizes:
            self.scale_norms[str(patch_size)] = nn.LayerNorm(d_model)
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, scale_embeddings, scale_patch_nums):
        batch_size, n_vars = scale_embeddings[0].shape[:2]
        
        # Step 1: Per-scale self-attention
        refined_scales = []
        for i, (patch_size, embedding) in enumerate(zip(self.patch_sizes, scale_embeddings)):
            # Reshape for attention: [B*n_vars, patch_num, d_model]
            embedding_flat = embedding.view(batch_size * n_vars, embedding.shape[2], self.d_model)
            
            # Apply scale-specific attention
            refined, _ = self.scale_attentions[str(patch_size)](
                embedding_flat, embedding_flat, embedding_flat
            )
            
            # Residual connection and normalization
            refined = self.scale_norms[str(patch_size)](embedding_flat + refined)
            
            # Reshape back: [B, n_vars, patch_num, d_model]
            refined = refined.view(batch_size, n_vars, embedding.shape[2], self.d_model)
            refined_scales.append(refined)
        
        # Step 2: Cross-scale interaction
        all_scales = torch.cat(refined_scales, dim=2)  # [B, n_vars, total_patches, d_model]
        all_scales_flat = all_scales.view(batch_size * n_vars, -1, self.d_model)
        
        cross_attended, attention_weights = self.cross_scale_attention(
            all_scales_flat, all_scales_flat, all_scales_flat
        )
        
        # Step 3: Apply learnable scale importance weights
        importance_weights = F.softmax(self.scale_importance, dim=0)
        
        # Split back to individual scales and apply importance weights
        start_idx = 0
        weighted_scales = []
        for i, (embedding, patch_num) in enumerate(zip(refined_scales, scale_patch_nums)):
            end_idx = start_idx + patch_num
            scale_output = cross_attended[:, start_idx:end_idx, :]
            
            # Apply importance weight
            weighted_scale = scale_output * importance_weights[i]
            weighted_scales.append(weighted_scale)
            start_idx = end_idx
        
        # Concatenate weighted scales
        final_output = torch.cat(weighted_scales, dim=1)
        
        # Final projection
        final_output = self.output_projection(final_output)
        
        # Reshape back to [B, n_vars, total_patches, d_model]
        total_patches = final_output.shape[1]
        final_output = final_output.view(batch_size, n_vars, total_patches, self.d_model)
        
        return final_output


class ProgressiveMultiResFusion(nn.Module):
    """Progressive multi-resolution fusion inspired by FPN"""
    def __init__(self, d_model, patch_sizes):
        super().__init__()
        self.d_model = d_model
        self.patch_sizes = sorted(patch_sizes)  # Sort from fine to coarse
        
        # Pyramid fusion layers
        self.pyramid_layers = nn.ModuleList()
        for i in range(len(patch_sizes) - 1):
            self.pyramid_layers.append(
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Cross-resolution connections with adaptive pooling
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1) for _ in range(len(patch_sizes) - 1)
        ])
        
    def forward(self, scale_embeddings, scale_patch_nums):
        batch_size, n_vars = scale_embeddings[0].shape[:2]
        
        # Sort embeddings by patch size (fine to coarse)
        patch_size_indices = np.argsort(self.patch_sizes)
        sorted_embeddings = [scale_embeddings[i] for i in patch_size_indices]
        sorted_patch_nums = [scale_patch_nums[i] for i in patch_size_indices]
        
        # Start with finest scale
        features = sorted_embeddings[0]  # [B, n_vars, patch_num, d_model]
        
        # Progressive fusion - but keep concatenating all scales
        all_fused_scales = [features]
        
        for i, pyramid_layer in enumerate(self.pyramid_layers):
            next_scale = sorted_embeddings[i + 1]
            
            # Simple approach: just combine features with next scale through concatenation
            # and apply pyramid layer for refinement
            combined = torch.cat([features, next_scale], dim=-1)  # [B, n_vars, patch_num, 2*d_model]
            
            # Reshape for pyramid layer
            combined_flat = combined.view(batch_size * n_vars, combined.shape[2], combined.shape[3])
            
            # Apply pyramid layer
            refined = pyramid_layer(combined_flat)
            refined = refined.view(batch_size, n_vars, refined.shape[1], self.d_model)
            
            # Update features for next iteration
            features = refined
            all_fused_scales.append(next_scale)  # Keep all original scales
        
        # Final output: concatenate all original scales to maintain expected dimensions
        final_output = torch.cat(all_fused_scales, dim=2)
        
        return final_output


class ScaleFusionModule(nn.Module):
    """Advanced fusion module for multi-scale patches"""
    def __init__(self, d_model, num_scales, fusion_type="attention"):
        super(ScaleFusionModule, self).__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            # Cross-scale attention mechanism
            self.scale_attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            )
            
        elif fusion_type == "scale_aware_attention":
            # Use the new scale-aware attention fusion
            self.scale_aware_fusion = ScaleAwareAttentionFusion(d_model, [8, 16, 24])  # Default patch sizes
            
        elif fusion_type == "progressive_multires":
            # Use progressive multi-resolution fusion
            self.progressive_fusion = ProgressiveMultiResFusion(d_model, [8, 16, 24])
            
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.gate_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.gate_projection = nn.Linear(d_model * num_scales, d_model)
            self.gate_activation = nn.Sigmoid()
            
        elif fusion_type == "hierarchical":
            # Hierarchical fusion - combine scales progressively
            self.scale_combiners = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ) for _ in range(num_scales - 1)
            ])
    
    def forward(self, scale_embeddings, scale_patch_nums):
        """
        Args:
            scale_embeddings: List of embeddings for each scale [B, n_vars, patch_num_i, d_model]
            scale_patch_nums: List of patch numbers for each scale
        Returns:
            fused_embedding: [B, n_vars, total_patches, d_model]
        """
        batch_size, n_vars = scale_embeddings[0].shape[:2]
        
        if self.fusion_type == "attention":
            return self._attention_fusion(scale_embeddings, batch_size, n_vars)
        elif self.fusion_type == "scale_aware_attention":
            return self.scale_aware_fusion(scale_embeddings, scale_patch_nums)
        elif self.fusion_type == "progressive_multires":
            return self.progressive_fusion(scale_embeddings, scale_patch_nums)
        elif self.fusion_type == "gated":
            return self._gated_fusion(scale_embeddings, batch_size, n_vars)
        elif self.fusion_type == "hierarchical":
            return self._hierarchical_fusion(scale_embeddings, batch_size, n_vars)
        else:
            # Fallback to simple concatenation
            return torch.cat(scale_embeddings, dim=2)
    
    def _attention_fusion(self, scale_embeddings, batch_size, n_vars):
        """Cross-scale attention fusion"""
        # Concatenate all scales for attention
        all_patches = torch.cat(scale_embeddings, dim=2)  # [B, n_vars, total_patches, d_model]
        
        # Reshape for attention: [B*n_vars, total_patches, d_model]
        all_patches_flat = all_patches.view(batch_size * n_vars, -1, self.d_model)
        
        # Self-attention across all patches from different scales
        attn_out, _ = self.scale_attention(all_patches_flat, all_patches_flat, all_patches_flat)
        attn_out = self.norm1(all_patches_flat + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(attn_out)
        fused = self.norm2(attn_out + ffn_out)
        
        # Reshape back: [B, n_vars, total_patches, d_model]
        total_patches = fused.shape[1]
        fused = fused.view(batch_size, n_vars, total_patches, self.d_model)
        
        return fused
    
    def _gated_fusion(self, scale_embeddings, batch_size, n_vars):
        """Gated fusion with learnable scale weights"""
        # Apply learned weights to each scale
        weighted_scales = []
        for i, embedding in enumerate(scale_embeddings):
            weight = torch.softmax(self.gate_weights, dim=0)[i]
            weighted_scales.append(embedding * weight)
        
        # Concatenate weighted scales
        concatenated = torch.cat(weighted_scales, dim=2)
        
        return concatenated
    
    def _hierarchical_fusion(self, scale_embeddings, batch_size, n_vars):
        """Hierarchical fusion - combine scales progressively"""
        if len(scale_embeddings) < 2:
            return scale_embeddings[0]
        
        # Start with the first two scales
        fused = scale_embeddings[0]
        
        for i in range(1, len(scale_embeddings)):
            # Combine current fused result with next scale
            next_scale = scale_embeddings[i]
            
            # Pad to same length if needed for concatenation
            if fused.shape[2] != next_scale.shape[2]:
                # Simply concatenate along patch dimension
                combined_input = torch.cat([fused, next_scale], dim=2)
            else:
                # If same patch count, concatenate along feature dimension
                combined_input = torch.cat([fused, next_scale], dim=3)
                # Project back to d_model
                combined_input = combined_input.view(batch_size * n_vars, -1, self.d_model * 2)
                combined_input = self.scale_combiners[i-1](combined_input)
                combined_input = combined_input.view(batch_size, n_vars, -1, self.d_model)
                fused = combined_input
                continue
            
            # For different patch counts, we concatenate and continue
            fused = combined_input
        
        return fused


class MultiScaleEnEmbedding(nn.Module):
    """Enhanced Multi-scale patch embedding with advanced fusion"""
    def __init__(self, n_vars, d_model, patch_sizes, seq_len, dropout, fusion_type="attention"):
        super(MultiScaleEnEmbedding, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # Create embeddings for each patch size
        self.patch_embeddings = nn.ModuleDict()
        self.patch_nums = {}
        
        for patch_size in patch_sizes:
            if patch_size > 0:
                patch_num = seq_len // patch_size
                self.patch_nums[str(patch_size)] = patch_num
                self.patch_embeddings[str(patch_size)] = nn.Linear(patch_size, d_model, bias=False)
        
        # Scale-specific global tokens (one per scale)
        self.global_tokens = nn.ParameterDict()
        for patch_size in patch_sizes:
            self.global_tokens[str(patch_size)] = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        
        # Position embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Advanced fusion module with new fusion types
        self.scale_fusion = ScaleFusionModule(d_model, len(patch_sizes), fusion_type)
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total patch numbers for head computation
        self.total_patch_num = sum(self.patch_nums.values()) + len(patch_sizes)  # +len for global tokens
        
    def forward(self, x):
        # x shape: [B, C, L]
        n_vars = x.shape[1]
        batch_size = x.shape[0]
        
        scale_embeddings = []
        scale_patch_nums = []
        
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
                
                # Add scale-specific global token
                scale_global = self.global_tokens[str(patch_size)].repeat((batch_size, 1, 1, 1))
                embedded_with_glb = torch.cat([embedded, scale_global], dim=2)
                
                scale_embeddings.append(embedded_with_glb)
                scale_patch_nums.append(embedded_with_glb.shape[2])
        
        # Apply advanced fusion
        if len(scale_embeddings) > 1:
            fused_embedding = self.scale_fusion(scale_embeddings, scale_patch_nums)
        else:
            fused_embedding = scale_embeddings[0]
        
        # Reshape for encoder input
        final_embedding = torch.reshape(fused_embedding, 
                                      (fused_embedding.shape[0] * fused_embedding.shape[1], 
                                       fused_embedding.shape[2], fused_embedding.shape[3]))
        
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
            self.fusion_type = getattr(configs, 'fusion_type', 'attention')
        
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # Embedding - choose between single scale or multi-scale
        if self.use_multi_scale:
            self.en_embedding = MultiScaleEnEmbedding(
                self.n_vars, configs.d_model, self.patch_sizes, configs.seq_len, configs.dropout, self.fusion_type
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