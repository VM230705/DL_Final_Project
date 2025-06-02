#!/usr/bin/env python3
"""Test script for complete TimeXer model with multi-scale patches"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.TimeXer import Model
import argparse

def create_test_config():
    """Create a test configuration for the model"""
    config = argparse.Namespace()
    
    # Basic config
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
    
    # Multi-scale patch config
    config.use_multi_scale = True
    config.patch_sizes = [8, 16, 24]
    config.patch_len = 16  # Fallback for single scale
    
    return config

def test_complete_model():
    """Test the complete TimeXer model with multi-scale patches"""
    print("Testing Complete TimeXer Model with Multi-Scale Patches...")
    
    # Create test config
    config = create_test_config()
    
    # Create model
    model = Model(config)
    print(f"Model created successfully!")
    print(f"Using multi-scale: {model.use_multi_scale}")
    print(f"Patch sizes: {model.patch_sizes if model.use_multi_scale else [model.patch_len]}")
    print(f"Total patch number: {model.patch_num}")
    print(f"Head input features: {model.head_nf}")
    
    # Create test data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # Typical time features
    x_dec = torch.randn(batch_size, config.pred_len, config.enc_in)
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    print(f"\nInput shapes:")
    print(f"x_enc: {x_enc.shape}")
    print(f"x_mark_enc: {x_mark_enc.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected output shape: [batch_size, pred_len, enc_in] = [{batch_size}, {config.pred_len}, {config.enc_in}]")
    
    # Test single scale for comparison
    print("\n" + "="*50)
    print("Testing Single-Scale Model for comparison...")
    
    config_single = create_test_config()
    config_single.use_multi_scale = False
    
    model_single = Model(config_single)
    print(f"Single-scale patch number: {model_single.patch_num}")
    print(f"Single-scale head input features: {model_single.head_nf}")
    
    with torch.no_grad():
        output_single = model_single(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"Single-scale output shape: {output_single.shape}")
    
    print("\nTest completed successfully!")
    print(f"Multi-scale model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Single-scale model has {sum(p.numel() for p in model_single.parameters())} parameters")
    
    return True

if __name__ == "__main__":
    test_complete_model()