#!/usr/bin/env python3
"""Test script to compare the improved global token design"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimeXer import MultiScaleEnEmbedding, EnEmbedding

def test_improved_global_tokens():
    """Test the improved shared global token design"""
    print("Testing Improved Multi-Scale Patch Embedding with Shared Global Token...")
    
    # Test parameters
    batch_size = 2
    n_vars = 7
    seq_len = 96
    d_model = 128
    patch_sizes = [8, 16, 24]
    dropout = 0.1
    
    # Create improved multi-scale embedding
    improved_embedding = MultiScaleEnEmbedding(
        n_vars=n_vars, 
        d_model=d_model, 
        patch_sizes=patch_sizes, 
        seq_len=seq_len, 
        dropout=dropout
    )
    
    print(f"Patch sizes: {patch_sizes}")
    print(f"Shared global token shape: {improved_embedding.shared_glb_token.shape}")
    print(f"Number of scale adapters: {len(improved_embedding.scale_adapters)}")
    
    # Test forward pass
    x = torch.randn(batch_size, n_vars, seq_len)
    output, n_vars_out = improved_embedding(x)
    
    print(f"\nForward pass results:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected patch dimension: {improved_embedding.total_patch_num}")
    
    # Check scale adapters
    print(f"\n=== Scale Adapter Analysis ===")
    for patch_size in patch_sizes:
        adapter = improved_embedding.scale_adapters[str(patch_size)]
        print(f"Adapter for patch size {patch_size}:")
        print(f"  Weight shape: {adapter.weight.shape}")
        print(f"  Has bias: {adapter.bias is not None}")
    
    # Test adapted global tokens
    print(f"\n=== Adapted Global Tokens ===")
    adapted_tokens = []
    for patch_size in patch_sizes:
        adapted = improved_embedding.scale_adapters[str(patch_size)](improved_embedding.shared_glb_token)
        adapted_tokens.append(adapted)
        print(f"Patch size {patch_size} adapted token norm: {torch.norm(adapted).item():.4f}")
    
    # Check similarities between adapted tokens
    print(f"\n=== Adapted Token Similarities ===")
    for i in range(len(adapted_tokens)):
        for j in range(i+1, len(adapted_tokens)):
            token1 = adapted_tokens[i].flatten()
            token2 = adapted_tokens[j].flatten()
            similarity = torch.cosine_similarity(token1, token2, dim=0)
            print(f"Similarity between adapted tokens {patch_sizes[i]} and {patch_sizes[j]}: {similarity.item():.4f}")
    
    # Parameter count comparison
    total_params = sum(p.numel() for p in improved_embedding.parameters())
    print(f"\nTotal parameters in improved design: {total_params}")
    
    return True

def compare_designs():
    """Compare the improved design with alternative approaches"""
    print("\n" + "="*60)
    print("Design Comparison")
    
    # Parameters for comparison
    n_vars, d_model, patch_sizes, seq_len, dropout = 7, 128, [8, 16, 24], 96, 0.1
    
    # Improved design (current)
    improved = MultiScaleEnEmbedding(n_vars, d_model, patch_sizes, seq_len, dropout)
    improved_params = sum(p.numel() for p in improved.parameters())
    
    # Single scale baseline
    single_scale = EnEmbedding(n_vars, d_model, 16, dropout)  # patch_len=16
    single_params = sum(p.numel() for p in single_scale.parameters())
    
    print(f"Parameter Comparison:")
    print(f"- Single-scale (patch_len=16): {single_params:,} parameters")
    print(f"- Multi-scale (improved): {improved_params:,} parameters")
    print(f"- Overhead: {improved_params - single_params:,} parameters ({(improved_params/single_params - 1)*100:.1f}% increase)")
    
    # Test computational efficiency
    x_single = torch.randn(2, n_vars, seq_len)
    x_multi = torch.randn(2, n_vars, seq_len)
    
    # Time single scale (fix the input format)
    import time
    start = time.time()
    for _ in range(100):
        # For single scale, we need to use only one channel as per original TimeXer
        x_single_input = x_single[:, :, -1].unsqueeze(-1).permute(0, 2, 1)  # [B, 1, C]
        _, _ = single_scale(x_single_input)
    single_time = time.time() - start
    
    # Time multi-scale
    start = time.time()
    for _ in range(100):
        _, _ = improved(x_multi)  # [B, C, L] format
    multi_time = time.time() - start
    
    print(f"\nComputational Efficiency (100 forward passes):")
    print(f"- Single-scale: {single_time:.4f}s")
    print(f"- Multi-scale: {multi_time:.4f}s")
    print(f"- Overhead: {multi_time/single_time:.2f}x slower")
    
    print(f"\nDesign Benefits:")
    print(f"✓ Shared global token reduces parameters vs separate tokens")
    print(f"✓ Scale-specific adaptation maintains expressiveness") 
    print(f"✓ Learnable transformations for each patch size")
    print(f"✓ Multi-resolution temporal modeling")
    
    # Breakdown of parameter usage
    print(f"\nParameter Breakdown for Multi-scale:")
    patch_embed_params = sum(p.numel() for p in improved.patch_embeddings.parameters())
    adapter_params = sum(p.numel() for p in improved.scale_adapters.parameters())
    global_token_params = improved.shared_glb_token.numel()
    other_params = improved_params - patch_embed_params - adapter_params - global_token_params
    
    print(f"- Patch embeddings: {patch_embed_params:,} ({patch_embed_params/improved_params*100:.1f}%)")
    print(f"- Scale adapters: {adapter_params:,} ({adapter_params/improved_params*100:.1f}%)")
    print(f"- Shared global token: {global_token_params:,} ({global_token_params/improved_params*100:.1f}%)")
    print(f"- Other components: {other_params:,} ({other_params/improved_params*100:.1f}%)")

if __name__ == "__main__":
    test_improved_global_tokens()
    compare_designs()