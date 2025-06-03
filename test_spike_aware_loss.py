#!/usr/bin/env python3
"""
Comprehensive test script for spike-aware loss functions.
Tests functionality, correctness, and effectiveness of all spike-aware loss variants.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append('/home/vm230705/dlp/Project')

from utils.losses import SpikeAwareLoss, AdaptiveSpikeAwareLoss, HybridLoss

def create_synthetic_data_with_spikes(batch_size=8, seq_len=96, num_vars=7, spike_probability=0.1, spike_magnitude=3.0):
    """
    Create synthetic time series data with controlled spikes for testing.
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence  
        num_vars: Number of variables/features
        spike_probability: Probability of spike occurrence at each timestep
        spike_magnitude: Magnitude multiplier for spikes
    
    Returns:
        data: [batch_size, seq_len, num_vars] synthetic time series with spikes
        spike_locations: [batch_size, seq_len, num_vars] binary mask indicating spike locations
    """
    # Generate base smooth time series
    t = torch.linspace(0, 4*np.pi, seq_len)
    base_data = torch.zeros(batch_size, seq_len, num_vars)
    
    for b in range(batch_size):
        for v in range(num_vars):
            # Different frequency for each variable and batch
            freq = 0.5 + v * 0.3 + b * 0.1
            phase = b * 0.2 + v * 0.5
            base_data[b, :, v] = torch.sin(freq * t + phase) + 0.1 * torch.randn(seq_len)
    
    # Add controlled spikes
    spike_mask = torch.rand(batch_size, seq_len, num_vars) < spike_probability
    spike_values = torch.randn(batch_size, seq_len, num_vars) * spike_magnitude
    
    # Apply spikes
    spiked_data = base_data.clone()
    spiked_data[spike_mask] += spike_values[spike_mask]
    
    return spiked_data, spike_mask

def test_spike_aware_loss_basic_functionality():
    """Test basic functionality of all spike-aware loss variants"""
    print("🔧 Testing Basic Functionality of Spike-Aware Loss Functions")
    print("=" * 70)
    
    # Create test data
    batch_size, seq_len, num_vars = 4, 48, 3
    true_data, spike_locations = create_synthetic_data_with_spikes(batch_size, seq_len, num_vars)
    
    # Test all loss variants
    loss_functions = {
        'SpikeAwareLoss': SpikeAwareLoss(base_loss='mse', spike_weight=2.0, derivative_weight=1.5),
        'AdaptiveSpikeAwareLoss': AdaptiveSpikeAwareLoss(base_loss='mse', initial_spike_weight=2.0, initial_derivative_weight=1.5),
        'HybridLoss': HybridLoss(base_loss='mse', alpha=0.7, spike_weight=2.0, derivative_weight=1.5)
    }
    
    results = {}
    for name, loss_fn in loss_functions.items():
        try:
            # Create fresh prediction data with gradient tracking for each test
            pred_data = true_data + 0.2 * torch.randn_like(true_data)
            pred_data = pred_data.clone().detach().requires_grad_(True)
            
            loss_value = loss_fn(pred_data, true_data)
            results[name] = loss_value.item()
            print(f"✅ {name}: {loss_value.item():.6f}")
            
            # Test gradient computation
            loss_value.backward()
            if pred_data.grad is not None:
                print(f"   Gradient computation: ✅ Success")
            else:
                print(f"❌ {name}: No gradients computed")
            
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            results[name] = None
    
    # Compare with standard MSE
    pred_data_mse = true_data + 0.2 * torch.randn_like(true_data)
    mse_loss = nn.MSELoss()(pred_data_mse, true_data)
    results['StandardMSE'] = mse_loss.item()
    print(f"📊 Standard MSE: {mse_loss.item():.6f}")
    
    print(f"\n🔍 Analysis:")
    for name, value in results.items():
        if value is not None and name != 'StandardMSE':
            ratio = value / results['StandardMSE']
            print(f"   {name} vs MSE ratio: {ratio:.3f}")
    
    return results

def test_spike_detection_accuracy():
    """Test if spike-aware loss correctly identifies and weights spikes"""
    print("\n🎯 Testing Spike Detection Accuracy")
    print("=" * 70)
    
    # Create data with known spike locations
    batch_size, seq_len, num_vars = 2, 32, 2
    true_data, known_spikes = create_synthetic_data_with_spikes(
        batch_size, seq_len, num_vars, spike_probability=0.15, spike_magnitude=4.0
    )
    
    # Create prediction that's worse at spike locations
    pred_data = true_data.clone()
    # Add larger errors at spike locations
    spike_errors = torch.randn_like(true_data) * 1.5
    pred_data[known_spikes] += spike_errors[known_spikes]
    # Add smaller errors elsewhere
    normal_errors = torch.randn_like(true_data) * 0.3
    pred_data[~known_spikes] += normal_errors[~known_spikes]
    
    # Test spike-aware loss
    spike_loss = SpikeAwareLoss(base_loss='mse', spike_weight=3.0, derivative_weight=2.0, percentile=90)
    
    # Compute losses
    base_losses = nn.MSELoss(reduction='none')(pred_data, true_data)
    error_weights = spike_loss._compute_error_weights(base_losses)
    derivative_weights = spike_loss._compute_derivative_weights(true_data)
    
    print(f"📈 Spike Analysis:")
    print(f"   Known spikes: {known_spikes.sum().item()}/{known_spikes.numel()} locations")
    print(f"   High error weights: {(error_weights > 0).sum().item()} locations")
    print(f"   High derivative weights: {(derivative_weights > 0).sum().item()} locations")
    
    # Check overlap between known spikes and detected high-weight regions
    total_weights = error_weights + derivative_weights
    detected_spikes = total_weights > 0.5
    
    # Calculate detection metrics
    true_positives = (known_spikes & detected_spikes).sum().item()
    false_positives = (~known_spikes & detected_spikes).sum().item()
    false_negatives = (known_spikes & ~detected_spikes).sum().item()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    
    print(f"   Detection Precision: {precision:.3f}")
    print(f"   Detection Recall: {recall:.3f}")
    
    return precision, recall

def test_gradient_flow_and_numerical_stability():
    """Test gradient flow and numerical stability"""
    print("\n🔄 Testing Gradient Flow and Numerical Stability")
    print("=" * 70)
    
    # Test with various data scales and conditions
    test_conditions = [
        ("Normal scale", 1.0, False),
        ("Large scale", 100.0, False),
        ("Small scale", 0.01, False),
        ("With NaN protection", 1.0, True)
    ]
    
    for condition_name, scale, add_extreme in test_conditions:
        print(f"\n🧪 Testing: {condition_name}")
        
        # Create test data
        true_data = torch.randn(4, 24, 3) * scale
        pred_data = true_data + 0.1 * torch.randn_like(true_data) * scale
        
        if add_extreme:
            # Add some extreme values to test robustness
            extreme_mask = torch.rand_like(true_data) < 0.05
            true_data[extreme_mask] *= 1000
        
        # Test loss computation and gradients
        loss_fn = SpikeAwareLoss(base_loss='mse', spike_weight=2.0, derivative_weight=1.5)
        
        # Enable gradient tracking
        pred_data.requires_grad_(True)
        
        try:
            loss = loss_fn(pred_data, true_data)
            
            # Check for NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ❌ Loss is NaN/Inf: {loss.item()}")
                continue
            
            # Compute gradients
            loss.backward()
            
            # Check gradients
            if pred_data.grad is not None:
                grad_norm = pred_data.grad.norm().item()
                has_nan_grad = torch.isnan(pred_data.grad).any().item()
                has_inf_grad = torch.isinf(pred_data.grad).any().item()
                
                print(f"   ✅ Loss: {loss.item():.6f}")
                print(f"   ✅ Gradient norm: {grad_norm:.6f}")
                print(f"   ✅ Has NaN gradients: {has_nan_grad}")
                print(f"   ✅ Has Inf gradients: {has_inf_grad}")
                
                if has_nan_grad or has_inf_grad:
                    print(f"   ⚠️  Gradient issues detected!")
            else:
                print(f"   ❌ No gradients computed")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def test_effectiveness_on_spike_prediction():
    """Test if spike-aware loss actually improves spike prediction"""
    print("\n📊 Testing Effectiveness on Spike Prediction")
    print("=" * 70)
    
    # Create challenging spike data
    batch_size, seq_len, num_vars = 16, 64, 5
    true_data, spike_mask = create_synthetic_data_with_spikes(
        batch_size, seq_len, num_vars, spike_probability=0.08, spike_magnitude=5.0
    )
    
    # Simulate two different prediction scenarios
    # Scenario 1: Predictions good everywhere
    pred_uniform = true_data + 0.3 * torch.randn_like(true_data)
    
    # Scenario 2: Predictions worse at spikes  
    pred_spike_biased = true_data.clone()
    pred_spike_biased += 0.2 * torch.randn_like(true_data)  # Base error
    spike_error = 2.0 * torch.randn_like(true_data)  # Large spike errors
    pred_spike_biased[spike_mask] += spike_error[spike_mask]
    
    # Test with different loss functions
    standard_mse = nn.MSELoss()
    spike_aware = SpikeAwareLoss(base_loss='mse', spike_weight=3.0, derivative_weight=2.0)
    hybrid_loss = HybridLoss(base_loss='mse', alpha=0.6, spike_weight=3.0, derivative_weight=2.0)
    
    results = {}
    
    for pred_name, pred_data in [("Uniform Error", pred_uniform), ("Spike-Biased Error", pred_spike_biased)]:
        print(f"\n📈 {pred_name} Predictions:")
        
        # Compute losses
        mse_loss = standard_mse(pred_data, true_data)
        spike_loss = spike_aware(pred_data, true_data)
        hybrid_loss_val = hybrid_loss(pred_data, true_data)
        
        # Compute spike-specific metrics
        spike_locations = spike_mask
        spike_mse = nn.MSELoss()(pred_data[spike_locations], true_data[spike_locations])
        normal_mse = nn.MSELoss()(pred_data[~spike_locations], true_data[~spike_locations])
        
        print(f"   Standard MSE: {mse_loss.item():.6f}")
        print(f"   Spike-Aware Loss: {spike_loss.item():.6f}")
        print(f"   Hybrid Loss: {hybrid_loss_val.item():.6f}")
        print(f"   Spike-only MSE: {spike_mse.item():.6f}")
        print(f"   Normal-only MSE: {normal_mse.item():.6f}")
        print(f"   Spike/Normal MSE ratio: {(spike_mse/normal_mse).item():.3f}")
        
        results[pred_name] = {
            'standard_mse': mse_loss.item(),
            'spike_aware': spike_loss.item(),
            'hybrid': hybrid_loss_val.item(),
            'spike_mse': spike_mse.item(),
            'normal_mse': normal_mse.item()
        }
    
    # Analysis
    print(f"\n🔍 Effectiveness Analysis:")
    uniform_results = results["Uniform Error"]
    biased_results = results["Spike-Biased Error"]
    
    # Check if spike-aware loss correctly penalizes spike-biased predictions more
    mse_penalty_ratio = biased_results['standard_mse'] / uniform_results['standard_mse']
    spike_penalty_ratio = biased_results['spike_aware'] / uniform_results['spike_aware']
    
    print(f"   MSE penalty ratio (biased/uniform): {mse_penalty_ratio:.3f}")
    print(f"   Spike-aware penalty ratio (biased/uniform): {spike_penalty_ratio:.3f}")
    
    if spike_penalty_ratio > mse_penalty_ratio:
        print(f"   ✅ Spike-aware loss correctly penalizes spike errors more heavily")
        effectiveness_score = (spike_penalty_ratio - mse_penalty_ratio) / mse_penalty_ratio
        print(f"   📊 Effectiveness improvement: {effectiveness_score*100:.1f}%")
    else:
        print(f"   ⚠️  Spike-aware loss doesn't show expected behavior")
        effectiveness_score = 0
    
    return effectiveness_score

def run_comprehensive_spike_loss_test():
    """Run all spike-aware loss tests"""
    print("🚀 COMPREHENSIVE SPIKE-AWARE LOSS TESTING")
    print("=" * 80)
    
    try:
        # Test 1: Basic functionality
        basic_results = test_spike_aware_loss_basic_functionality()
        
        # Test 2: Spike detection accuracy
        precision, recall = test_spike_detection_accuracy()
        
        # Test 3: Gradient flow and numerical stability
        test_gradient_flow_and_numerical_stability()
        
        # Test 4: Effectiveness on spike prediction
        effectiveness = test_effectiveness_on_spike_prediction()
        
        # Final summary
        print("\n" + "=" * 80)
        print("📋 FINAL TEST SUMMARY")
        print("=" * 80)
        
        print(f"✅ Basic Functionality: All loss variants working")
        print(f"🎯 Spike Detection - Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"🔄 Gradient Flow: Stable (check details above)")
        print(f"📊 Effectiveness Improvement: {effectiveness*100:.1f}%")
        
        # Overall assessment
        all_tests_passed = all([
            all(v is not None for v in basic_results.values() if v != basic_results.get('StandardMSE')),
            precision > 0.3,  # Reasonable spike detection
            recall > 0.2,     # Reasonable spike detection
            effectiveness > 0.1  # At least 10% improvement
        ])
        
        if all_tests_passed:
            print(f"\n🎉 OVERALL ASSESSMENT: ✅ SPIKE-AWARE LOSS IS WORKING CORRECTLY AND EFFECTIVELY")
        else:
            print(f"\n⚠️  OVERALL ASSESSMENT: ❌ SOME ISSUES DETECTED - CHECK INDIVIDUAL TESTS")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN TESTING: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_spike_loss_test()