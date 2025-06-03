# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


class SpikeAwareLoss(nn.Module):
    """
    Spike-aware loss function that gives higher weight to high-error and high-derivative regions.
    This helps the model pay more attention to sudden changes and spikes in time series.
    """
    def __init__(self, base_loss='mse', spike_weight=2.0, derivative_weight=1.5, percentile=95):
        super(SpikeAwareLoss, self).__init__()
        self.base_loss = base_loss
        self.spike_weight = spike_weight
        self.derivative_weight = derivative_weight
        self.percentile = percentile
        
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
    
    def forward(self, pred, true):
        """
        Args:
            pred: [B, L, C] - predicted values
            true: [B, L, C] - ground truth values
        Returns:
            weighted_loss: scalar loss value
        """
        # Base loss computation
        base_losses = self.base_criterion(pred, true)  # [B, L, C]
        
        # 1. High-error region weighting
        error_weights = self._compute_error_weights(base_losses)
        
        # 2. High-derivative region weighting (spike detection)
        derivative_weights = self._compute_derivative_weights(true)
        
        # 3. Combine weights
        total_weights = 1.0 + error_weights + derivative_weights
        
        # 4. Apply weights and compute final loss
        weighted_losses = base_losses * total_weights
        
        return weighted_losses.mean()
    
    def _compute_error_weights(self, losses):
        """
        Compute weights based on error magnitude.
        High-error regions get higher weights.
        """
        batch_size, seq_len, num_vars = losses.shape
        
        # Compute percentile threshold for each variable across batch and time
        weights = torch.zeros_like(losses)
        
        for c in range(num_vars):
            var_losses = losses[:, :, c].flatten()
            threshold = torch.quantile(var_losses, self.percentile / 100.0)
            
            # Assign higher weights to losses above threshold
            high_error_mask = losses[:, :, c] > threshold
            weights[:, :, c] = torch.where(
                high_error_mask,
                torch.tensor(self.spike_weight - 1.0, device=losses.device),
                torch.tensor(0.0, device=losses.device)
            )
        
        return weights
    
    def _compute_derivative_weights(self, true):
        """
        Compute weights based on derivative magnitude (rate of change).
        Regions with high derivatives (spikes) get higher weights.
        """
        batch_size, seq_len, num_vars = true.shape
        
        # Compute first-order differences (derivatives)
        if seq_len > 1:
            derivatives = torch.abs(true[:, 1:, :] - true[:, :-1, :])  # [B, L-1, C]
            # Pad to match original length
            derivatives = F.pad(derivatives, (0, 0, 0, 1), mode='constant', value=0)
        else:
            derivatives = torch.zeros_like(true)
        
        # Compute weights based on derivative magnitude
        weights = torch.zeros_like(true)
        
        for c in range(num_vars):
            var_derivatives = derivatives[:, :, c].flatten()
            threshold = torch.quantile(var_derivatives, self.percentile / 100.0)
            
            # Assign higher weights to high-derivative regions
            high_derivative_mask = derivatives[:, :, c] > threshold
            weights[:, :, c] = torch.where(
                high_derivative_mask,
                torch.tensor(self.derivative_weight - 1.0, device=true.device),
                torch.tensor(0.0, device=true.device)
            )
        
        return weights


class AdaptiveSpikeAwareLoss(nn.Module):
    """
    Adaptive version that learns the optimal weighting during training.
    """
    def __init__(self, base_loss='mse', initial_spike_weight=2.0, initial_derivative_weight=1.5):
        super(AdaptiveSpikeAwareLoss, self).__init__()
        self.base_loss = base_loss
        
        # Learnable parameters
        self.spike_weight = nn.Parameter(torch.tensor(initial_spike_weight))
        self.derivative_weight = nn.Parameter(torch.tensor(initial_derivative_weight))
        
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
    
    def forward(self, pred, true):
        """Similar to SpikeAwareLoss but with learnable weights"""
        base_losses = self.base_criterion(pred, true)
        
        # Use learnable parameters
        error_weights = self._compute_error_weights(base_losses)
        derivative_weights = self._compute_derivative_weights(true)
        
        total_weights = 1.0 + error_weights + derivative_weights
        weighted_losses = base_losses * total_weights
        
        return weighted_losses.mean()
    
    def _compute_error_weights(self, losses):
        """Similar to SpikeAwareLoss but uses learnable spike_weight"""
        batch_size, seq_len, num_vars = losses.shape
        weights = torch.zeros_like(losses)
        
        for c in range(num_vars):
            var_losses = losses[:, :, c].flatten()
            threshold = torch.quantile(var_losses, 0.95)
            
            high_error_mask = losses[:, :, c] > threshold
            weights[:, :, c] = torch.where(
                high_error_mask,
                F.softplus(self.spike_weight) - 1.0,  # Ensure positive
                torch.tensor(0.0, device=losses.device)
            )
        
        return weights
    
    def _compute_derivative_weights(self, true):
        """Similar to SpikeAwareLoss but uses learnable derivative_weight"""
        batch_size, seq_len, num_vars = true.shape
        
        if seq_len > 1:
            derivatives = torch.abs(true[:, 1:, :] - true[:, :-1, :])
            derivatives = F.pad(derivatives, (0, 0, 0, 1), mode='constant', value=0)
        else:
            derivatives = torch.zeros_like(true)
        
        weights = torch.zeros_like(true)
        
        for c in range(num_vars):
            var_derivatives = derivatives[:, :, c].flatten()
            threshold = torch.quantile(var_derivatives, 0.95)
            
            high_derivative_mask = derivatives[:, :, c] > threshold
            weights[:, :, c] = torch.where(
                high_derivative_mask,
                F.softplus(self.derivative_weight) - 1.0,  # Ensure positive
                torch.tensor(0.0, device=true.device)
            )
        
        return weights


class HybridLoss(nn.Module):
    """
    Combines standard loss with spike-aware loss for balanced training.
    """
    def __init__(self, base_loss='mse', alpha=0.7, spike_weight=2.0, derivative_weight=1.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # Weight for standard loss
        self.beta = 1.0 - alpha  # Weight for spike-aware loss
        
        if base_loss == 'mse':
            self.standard_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.standard_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
        
        self.spike_aware_loss = SpikeAwareLoss(
            base_loss=base_loss, 
            spike_weight=spike_weight, 
            derivative_weight=derivative_weight
        )
    
    def forward(self, pred, true):
        standard = self.standard_loss(pred, true)
        spike_aware = self.spike_aware_loss(pred, true)
        
        return self.alpha * standard + self.beta * spike_aware
