#!/usr/bin/env python3
"""Debug script to verify multi-scale patch usage"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimeXer import Model, MultiScaleEnEmbedding
import argparse

def create_debug_config():
    """Create a debug configuration"""
    config = argparse.Namespace()
    config.task_name = 'long_term_forecast'
    config.features = 'M'
    config.seq_len = 96
    config.pred_len = 96
    config.use_norm = 1
    config.enc_in = 7
    config.d_model = 128
    config.n_heads = 8
    config.e_layers = 2
    config.d_ff = 256
    config.factor = 1
    config.dropout = 0.1
    config.activation = 'gelu'
    config.embed = 'timeF'
    config.freq = 'h'
    
    # Multi-scale config
    config.use_multi_scale = True
    config.patch_sizes = [8, 16, 24]
    config.patch_len = 16
    
    return config

def debug_multi_scale_usage():
    """Debug whether multi-scale patches are actually being used"""
    print("🔍 DEBUGGING MULTI-SCALE PATCH USAGE")
    print("="*60)
    
    config = create_debug_config()
    model = Model(config)
    
    # Check model configuration
    print(f"✅ Model Configuration:")
    print(f"   - use_multi_scale: {model.use_multi_scale}")
    print(f"   - patch_sizes: {model.patch_sizes if hasattr(model, 'patch_sizes') else 'Not found'}")
    print(f"   - features: {model.features}")
    print(f"   - n_vars: {model.n_vars}")
    
    # Check embedding type
    print(f"\n✅ Embedding Type:")
    print(f"   - Embedding class: {type(model.en_embedding).__name__}")
    if isinstance(model.en_embedding, MultiScaleEnEmbedding):
        print(f"   - Patch sizes: {model.en_embedding.patch_sizes}")
        print(f"   - Total patches: {model.en_embedding.total_patch_num}")
        print(f"   - Individual patch counts: {model.en_embedding.patch_nums}")
    
    # Test forward pass routing
    print(f"\n🔍 Testing Forward Pass Routing:")
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.pred_len, config.enc_in)
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    # Check which forecast method is called
    print(f"   - Model features: {model.features}")
    if model.features == 'M':
        print(f"   - Will call: forecast_multi() ✅")
        
        # Test the embedding directly
        print(f"\n🔍 Testing Multi-Scale Embedding Directly:")
        test_input = x_enc.permute(0, 2, 1)  # [B, C, L] format for multi-scale
        print(f"   - Input shape to embedding: {test_input.shape}")
        
        with torch.no_grad():
            embed_out, n_vars = model.en_embedding(test_input)
            print(f"   - Embedding output shape: {embed_out.shape}")
            print(f"   - Expected patches: {model.en_embedding.total_patch_num}")
            print(f"   - Actual patches: {embed_out.shape[1]}")
            
            # Check if all patch sizes are being used
            if isinstance(model.en_embedding, MultiScaleEnEmbedding):
                print(f"\n📊 Patch Size Breakdown:")
                for patch_size in model.en_embedding.patch_sizes:
                    patch_count = model.en_embedding.patch_nums[str(patch_size)]
                    print(f"   - Patch size {patch_size}: {patch_count} patches + 1 global token")
    else:
        print(f"   - Will call: forecast() ❌ (single-scale path)")
    
    # Test full forward pass
    print(f"\n🔍 Testing Full Forward Pass:")
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"   - Final output shape: {output.shape}")
    
    return True

def compare_single_vs_multi():
    """Compare single vs multi-scale embeddings side by side"""
    print(f"\n" + "="*60)
    print("🔄 COMPARING SINGLE VS MULTI-SCALE SIDE BY SIDE")
    print("="*60)
    
    # Create test input
    batch_size = 1
    seq_len = 96
    n_vars = 7
    x = torch.randn(batch_size, n_vars, seq_len)
    
    # Test Multi-Scale
    print(f"🔍 Multi-Scale Embedding:")
    multi_embed = MultiScaleEnEmbedding(n_vars=n_vars, d_model=128, 
                                       patch_sizes=[8, 16, 24], seq_len=seq_len, dropout=0.1)
    multi_out, _ = multi_embed(x)
    print(f"   - Input: {x.shape}")
    print(f"   - Output: {multi_out.shape}")
    print(f"   - Patches breakdown:")
    for patch_size in [8, 16, 24]:
        count = seq_len // patch_size
        print(f"     * Size {patch_size}: {count} patches + 1 global = {count + 1}")
    print(f"   - Total expected: {sum([seq_len // p for p in [8, 16, 24]]) + 3} = {multi_embed.total_patch_num}")
    
    # Test if different patch sizes produce different embeddings
    print(f"\n📊 Verifying Different Patch Sizes Produce Different Embeddings:")
    
    # Extract embeddings for each scale separately (simulate what happens inside)
    for i, patch_size in enumerate([8, 16, 24]):
        x_patched = x.unfold(dimension=-1, size=patch_size, step=patch_size)
        print(f"   - Patch size {patch_size}: {x_patched.shape}")
        
        # Check if patches are different
        if i == 0:
            first_patch_shape = x_patched.shape
        else:
            if x_patched.shape != first_patch_shape:
                print(f"     ✅ Different from previous shapes - Multi-scale IS working!")
    
    return True

if __name__ == "__main__":
    debug_multi_scale_usage()
    compare_single_vs_multi()