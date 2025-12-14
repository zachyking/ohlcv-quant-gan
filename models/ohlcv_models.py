"""
Multi-channel GAN models for OHLCV data generation.

These models extend the original QuantGAN architecture to handle multiple
features simultaneously (log_return, body_ratio, wicks, range, volume).

Key improvements:
1. Multi-channel input/output for all OHLCV features
2. Spectral normalization for training stability
3. Self-attention layers to capture long-range dependencies (S/R levels)
4. Wasserstein loss with gradient penalty (WGAN-GP) for better convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Self-attention module to capture long-range dependencies."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, seq_len = x.size()
        
        # Compute attention
        query = self.query(x).view(batch_size, -1, seq_len).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, seq_len)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(batch_size, -1, seq_len)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        return self.gamma * out + x


class ResidualBlock1D(nn.Module):
    """Residual block with dilated convolutions for temporal modeling."""
    
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = spectral_norm(nn.Conv1d(channels, channels, kernel_size, 
                                              padding=padding, dilation=dilation))
        self.conv2 = spectral_norm(nn.Conv1d(channels, channels, kernel_size,
                                              padding=padding, dilation=dilation))
        self.norm1 = nn.InstanceNorm1d(channels)
        self.norm2 = nn.InstanceNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return F.leaky_relu(out + residual, 0.2)


class OHLCVGenerator(nn.Module):
    """
    Generator for multi-channel OHLCV features.
    
    Takes random noise and produces sequences of OHLCV features
    that should look like real market data.
    
    Args:
        noise_dim: Dimensionality of input noise per timestep
        hidden_dim: Hidden layer dimension
        output_dim: Number of output features (default 6: log_return, body_ratio, etc.)
        n_layers: Number of residual blocks
        seq_length: Output sequence length
    """
    
    def __init__(self, noise_dim=32, hidden_dim=128, output_dim=6, 
                 n_layers=4, seq_length=64, use_attention=True):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        
        # Initial projection
        self.input_proj = nn.Linear(noise_dim, hidden_dim * seq_length)
        
        # Residual blocks with increasing dilation
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(ResidualBlock1D(hidden_dim, dilation=dilation))
            
            # Add attention in the middle
            if use_attention and i == n_layers // 2:
                layers.append(SelfAttention(hidden_dim))
        
        self.main = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)
        
    def forward(self, z):
        """
        Args:
            z: Noise tensor of shape (batch_size, noise_dim)
        Returns:
            Generated features of shape (batch_size, output_dim, seq_length)
        """
        batch_size = z.size(0)
        
        # Project noise to sequence
        x = self.input_proj(z)
        x = x.view(batch_size, self.hidden_dim, self.seq_length)
        
        # Process through residual blocks
        x = self.main(x)
        
        # Output projection with tanh (features are normalized to [-1, 1])
        x = torch.tanh(self.output_proj(x))
        
        return x


class OHLCVDiscriminator(nn.Module):
    """
    Discriminator for multi-channel OHLCV features.
    
    Takes sequences of OHLCV features and outputs a realism score.
    Uses spectral normalization for WGAN-GP training stability.
    
    Args:
        input_dim: Number of input features (default 6)
        hidden_dim: Hidden layer dimension
        n_layers: Number of residual blocks
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, n_layers=4, use_attention=True):
        super().__init__()
        
        # Input projection
        self.input_proj = spectral_norm(nn.Conv1d(input_dim, hidden_dim, 1))
        
        # Residual blocks
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(ResidualBlock1D(hidden_dim, dilation=dilation))
            
            # Add attention
            if use_attention and i == n_layers // 2:
                layers.append(SelfAttention(hidden_dim))
        
        self.main = nn.Sequential(*layers)
        
        # Global pooling + output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: Feature tensor of shape (batch_size, input_dim, seq_length)
        Returns:
            Realism score of shape (batch_size, 1)
        """
        x = F.leaky_relu(self.input_proj(x), 0.2)
        x = self.main(x)
        x = self.pool(x).squeeze(-1)
        return self.output(x)


class OHLCVGeneratorLSTM(nn.Module):
    """
    LSTM-based generator for comparison/ablation.
    Better at capturing long-term dependencies but slower to train.
    """
    
    def __init__(self, noise_dim=32, hidden_dim=256, output_dim=6, 
                 n_layers=2, seq_length=64):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.n_layers = n_layers
        
        # Project noise to initial hidden state
        self.h0_proj = nn.Linear(noise_dim, n_layers * hidden_dim)
        self.c0_proj = nn.Linear(noise_dim, n_layers * hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(noise_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Create initial hidden states from noise
        h0 = self.h0_proj(z).view(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.c0_proj(z).view(self.n_layers, batch_size, self.hidden_dim)
        
        # Create sequence input from noise
        z_seq = z.unsqueeze(1).expand(-1, self.seq_length, -1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(z_seq, (h0.contiguous(), c0.contiguous()))
        
        # Project to output
        output = torch.tanh(self.output_proj(lstm_out))
        
        # Transpose to (batch, features, seq_len)
        return output.transpose(1, 2)


class OHLCVDiscriminatorLSTM(nn.Module):
    """LSTM-based discriminator for comparison."""
    
    def __init__(self, input_dim=6, hidden_dim=256, n_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Transpose to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        return self.output(lstm_out[:, -1])


def gradient_penalty(discriminator, real, fake, device='cpu'):
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces Lipschitz constraint by penalizing gradients
    of discriminator on interpolated samples.
    """
    batch_size = real.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated = interpolated.requires_grad_(True)
    
    # Get discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients - use cpu for MPS compatibility if needed
    try:
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
    except RuntimeError:
        # MPS fallback - compute on CPU
        d_interp_cpu = discriminator(interpolated.cpu())
        gradients = torch.autograd.grad(
            outputs=d_interp_cpu,
            inputs=interpolated.cpu(),
            grad_outputs=torch.ones_like(d_interp_cpu),
            create_graph=True,
            retain_graph=True
        )[0].to(device)
    
    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


# Feature constraint losses for OHLCV validity
def ohlcv_constraint_loss(generated):
    """
    Soft constraints to ensure generated features are valid OHLCV relationships.
    
    For the normalized features:
    - body_ratio should be in [-1, 1]
    - upper_wick, lower_wick should be in [0, 1]
    - range_pct should be positive
    """
    batch_size, n_features, seq_len = generated.size()
    
    loss = 0.0
    
    # Indices (assuming standard feature order)
    # 0: log_return, 1: body_ratio, 2: upper_wick, 3: lower_wick, 4: range_pct, 5: volume_norm
    
    if n_features >= 5:
        # Upper and lower wicks should sum to <= 1 (normalized)
        # In normalized space, this constraint is implicit
        
        # Range should be positive (in denormalized space)
        # For normalized, we just ensure it's not too negative
        range_pct = generated[:, 4, :]
        loss += F.relu(-range_pct - 0.5).mean()  # Penalize very negative values
    
    return loss
