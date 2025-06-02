#!/usr/bin/env python3
"""Test different fusion mechanisms for multi-scale patches"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimeXer import Model
import argparse

def create_test_config(fusion_type="attention"):
    """Create a test configuration for different fusion types"""
    config = argparse.Namespace()
    
    # Basic config
    config.task_name = 'long_term_forecast'
    config.features = 'M'
    config.seq_len = 96
    config.pred_len = 96
    config.use_norm = 1
    config.enc_in = 21  # Weather dataset
    config.d_model = 128
    config.n_heads = 8
    config.e_layers = 2
    config.d_ff = 256
    config.factor = 1
    config.dropout = 0.1
    config.activation = 'gelu'
    config.embed = 'timeF'
    config.freq = 'h'
    
    # Multi-scale patch config
    config.use_multi_scale = True
    config.patch_sizes = [8, 16, 24]
    config.patch_len = 16
    config.fusion_type = fusion_type
    
    return config

def test_fusion_mechanisms():
    """Test different fusion mechanisms"""
    print("🔬 TESTING DIFFERENT MULTI-SCALE FUSION MECHANISMS")
    print("="*70)
    
    fusion_types = ["attention", "gated", "hierarchical", "concat"]
    batch_size = 2
    
    results = {}
    
    for fusion_type in fusion_types:
        print(f"\n📊 Testing {fusion_type.upper()} Fusion:")
        print("-" * 40)
        
        try:
            # Create model with specific fusion type
            config = create_test_config(fusion_type)
            model = Model(config)
            
            print(f"✅ Model created with {fusion_type} fusion")
            print(f"   - Patch sizes: {model.patch_sizes}")
            print(f"   - Total patches: {model.patch_num}")
            print(f"   - Fusion type: {model.fusion_type}")
            print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create test data
            x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
            x_dec = torch.randn(batch_size, config.pred_len, config.enc_in)
            x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
            
            # Test forward pass with timing
            model.eval()
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            else:
                elapsed_time = 0  # CPU timing would need different approach
            
            print(f"   - Output shape: {output.shape}")
            print(f"   - Forward pass time: {elapsed_time:.2f}ms" if torch.cuda.is_available() else "   - CPU mode")
            print(f"   - Output mean: {output.mean().item():.6f}")
            print(f"   - Output std: {output.std().item():.6f}")
            
            results[fusion_type] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'output_shape': output.shape,
                'time_ms': elapsed_time if torch.cuda.is_available() else 0,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error with {fusion_type} fusion: {str(e)}")
            results[fusion_type] = {'success': False, 'error': str(e)}
    
    # Summary comparison
    print(f"\n📈 FUSION MECHANISM COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Fusion Type':<15} {'Parameters':<12} {'Time (ms)':<10} {'Status':<10}")
    print("-" * 70)
    
    for fusion_type, result in results.items():
        if result['success']:
            params = f"{result['parameters']:,}"
            time_ms = f"{result['time_ms']:.2f}" if result['time_ms'] > 0 else "N/A"
            status = "✅ Success"
        else:
            params = "N/A"
            time_ms = "N/A"
            status = "❌ Failed"
        
        print(f"{fusion_type:<15} {params:<12} {time_ms:<10} {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    successful = [k for k, v in results.items() if v['success']]
    
    if 'attention' in successful:
        print("✅ ATTENTION fusion: Best for learning complex cross-scale relationships")
    if 'gated' in successful:
        print("✅ GATED fusion: Good balance of performance and efficiency")
    if 'hierarchical' in successful:
        print("✅ HIERARCHICAL fusion: Progressive combination, memory efficient")
    if 'concat' in successful:
        print("✅ CONCAT fusion: Simple baseline, fastest")
    
    return results

if __name__ == "__main__":
    test_fusion_mechanisms()