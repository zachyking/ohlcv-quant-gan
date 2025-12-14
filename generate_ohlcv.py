#!/usr/bin/env python3
"""
Generate synthetic OHLCV data using trained QuantGAN models.

This script takes a trained model and generates synthetic market data
that preserves the statistical properties of the training data while
being entirely synthetic - useful for testing trading strategies
without overfitting to historical data.

Usage:
    python generate_ohlcv.py --checkpoint_dir checkpoints/BTCUSDT_... \
                            --n_sequences 100 \
                            --output_dir ../testdata/synthetic/
    
Output format matches your testdata format:
    time,open,high,low,close,volume
"""

import os
import argparse
import json
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from models.ohlcv_models import OHLCVGenerator, OHLCVGeneratorLSTM


def load_model(checkpoint_dir: str, checkpoint_name: str = 'netG_best.pth'):
    """Load trained generator model and configuration."""
    
    # Load training info
    with open(os.path.join(checkpoint_dir, 'train_info.json'), 'r') as f:
        train_info = json.load(f)
    
    # Create model
    if train_info['model_type'] == 'tcn':
        netG = OHLCVGenerator(
            noise_dim=train_info['noise_dim'],
            hidden_dim=train_info['hidden_dim'],
            output_dim=train_info['n_features'],
            n_layers=train_info['n_layers'],
            seq_length=train_info['seq_length'],
            use_attention=train_info.get('use_attention', True)
        )
    else:
        netG = OHLCVGeneratorLSTM(
            noise_dim=train_info['noise_dim'],
            hidden_dim=train_info['hidden_dim'],
            output_dim=train_info['n_features'],
            n_layers=train_info['n_layers'],
            seq_length=train_info['seq_length']
        )
    
    # Load weights
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    netG.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    netG.eval()
    
    return netG, train_info


def inverse_gaussianize(z_norm: np.ndarray, params: dict) -> np.ndarray:
    """Reverse the Gaussianization transform."""
    if not params.get('gaussianize', True):
        # Simple denormalization
        min_val, max_val = params.get('min', -1), params.get('max', 1)
        return (z_norm + 1) / 2 * (max_val - min_val) + min_val
    
    # Un-normalize z
    z = z_norm * params['std_z'] + params['mean_z']
    
    # Inverse Lambert W: x = z * exp(delta/2 * z^2)
    delta = params['delta']
    if delta > 0.01:
        x_norm = z * np.exp(delta / 2 * z**2)
    else:
        x_norm = z
    
    # Un-normalize x  
    x = x_norm * params['std_x'] + params['mean_x']
    
    return x


def features_to_ohlcv(features: np.ndarray, 
                      params: dict,
                      initial_price: float = 50000.0,
                      timeframe_seconds: int = 300,
                      start_timestamp: int = None) -> pd.DataFrame:
    """
    Convert generated features back to OHLCV data.
    
    Features order: [log_return, body_ratio, upper_wick, lower_wick, range_pct, volume_norm]
    
    Args:
        features: Array of shape (seq_length, n_features)
        params: Transform parameters from preprocessing
        initial_price: Starting close price
        timeframe_seconds: Time between candles (300 = 5 minutes)
        start_timestamp: Unix timestamp for first candle
    
    Returns:
        DataFrame with columns [time, open, high, low, close, volume]
    """
    n_candles = features.shape[0]
    
    # Extract and denormalize features
    feature_names = ['log_return', 'body_ratio', 'upper_wick', 'lower_wick', 'range_pct', 'volume_norm']
    
    denorm_features = {}
    for i, name in enumerate(feature_names):
        if name in params:
            denorm_features[name] = inverse_gaussianize(features[:, i], params[name])
        else:
            denorm_features[name] = features[:, i]
    
    # Reconstruct OHLCV
    closes = np.zeros(n_candles)
    opens = np.zeros(n_candles)
    highs = np.zeros(n_candles)
    lows = np.zeros(n_candles)
    volumes = np.zeros(n_candles)
    
    # First candle
    closes[0] = initial_price
    range_0 = abs(denorm_features['range_pct'][0]) * closes[0]
    range_0 = max(range_0, 0.01 * closes[0])  # Minimum range
    
    body_ratio = np.clip(denorm_features['body_ratio'][0], -1, 1)
    body = body_ratio * range_0
    
    if body >= 0:  # Bullish
        opens[0] = closes[0] - body
        upper_wick = np.clip(denorm_features['upper_wick'][0], 0, 1) * range_0
        lower_wick = np.clip(denorm_features['lower_wick'][0], 0, 1) * range_0
        highs[0] = closes[0] + upper_wick
        lows[0] = opens[0] - lower_wick
    else:  # Bearish
        opens[0] = closes[0] - body  # body is negative, so open > close
        upper_wick = np.clip(denorm_features['upper_wick'][0], 0, 1) * range_0
        lower_wick = np.clip(denorm_features['lower_wick'][0], 0, 1) * range_0
        highs[0] = opens[0] + upper_wick
        lows[0] = closes[0] - lower_wick
    
    # Volume - use absolute value and scale
    volumes[0] = abs(denorm_features['volume_norm'][0]) * 100 + 10
    
    # Rest of candles
    for i in range(1, n_candles):
        # Close from log return
        log_ret = denorm_features['log_return'][i]
        # Clip extreme returns
        log_ret = np.clip(log_ret, -0.1, 0.1)
        closes[i] = closes[i-1] * np.exp(log_ret)
        
        # Range
        range_i = abs(denorm_features['range_pct'][i]) * closes[i]
        range_i = max(range_i, 0.001 * closes[i])  # Minimum range
        
        # Body
        body_ratio = np.clip(denorm_features['body_ratio'][i], -1, 1)
        body = body_ratio * range_i
        
        if body >= 0:  # Bullish
            opens[i] = closes[i] - body
            upper_wick = np.clip(denorm_features['upper_wick'][i], 0, 1) * (range_i - abs(body))
            lower_wick = np.clip(denorm_features['lower_wick'][i], 0, 1) * (range_i - abs(body))
        else:  # Bearish
            opens[i] = closes[i] - body
            upper_wick = np.clip(denorm_features['upper_wick'][i], 0, 1) * (range_i - abs(body))
            lower_wick = np.clip(denorm_features['lower_wick'][i], 0, 1) * (range_i - abs(body))
        
        highs[i] = max(opens[i], closes[i]) + upper_wick
        lows[i] = min(opens[i], closes[i]) - lower_wick
        
        # Ensure OHLC validity
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
        
        # Volume
        volumes[i] = abs(denorm_features['volume_norm'][i]) * 100 + 10
    
    # Generate timestamps
    if start_timestamp is None:
        start_timestamp = int(datetime.now().timestamp())
    timestamps = np.array([start_timestamp + i * timeframe_seconds for i in range(n_candles)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps.astype(int),
        'open': np.round(opens, 2),
        'high': np.round(highs, 2),
        'low': np.round(lows, 2),
        'close': np.round(closes, 2),
        'volume': np.round(volumes, 5)
    })
    
    return df


def stitch_sequences(sequences: list, overlap: int = 0) -> np.ndarray:
    """
    Stitch multiple generated sequences together for longer time series.
    
    Args:
        sequences: List of arrays, each of shape (seq_length, n_features)
        overlap: Number of timesteps to overlap and blend
    
    Returns:
        Single array of shape (total_length, n_features)
    """
    if len(sequences) == 1:
        return sequences[0]
    
    result = sequences[0]
    
    for seq in sequences[1:]:
        if overlap > 0:
            # Blend overlapping region
            blend_weights = np.linspace(1, 0, overlap).reshape(-1, 1)
            blended = (result[-overlap:] * blend_weights + 
                      seq[:overlap] * (1 - blend_weights))
            result = np.vstack([result[:-overlap], blended, seq[overlap:]])
        else:
            result = np.vstack([result, seq])
    
    return result


def generate_synthetic_dataset(
    netG: torch.nn.Module,
    train_info: dict,
    n_sequences: int = 10,
    initial_price: float = 50000.0,
    timeframe_seconds: int = 300,
    start_timestamp: int = None,
    stitch: bool = True,
    overlap: int = 8,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Generate a complete synthetic OHLCV dataset.
    
    Args:
        netG: Trained generator model
        train_info: Training configuration and parameters
        n_sequences: Number of sequences to generate
        initial_price: Starting price
        timeframe_seconds: Candle timeframe
        start_timestamp: Start time
        stitch: Whether to stitch sequences into one continuous series
        overlap: Overlap for stitching
        device: Device to use
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    netG.to(device)
    netG.eval()
    
    noise_dim = train_info['noise_dim']
    seq_length = train_info['seq_length']
    params = train_info['data_params']
    
    # Generate sequences
    generated_sequences = []
    
    with torch.no_grad():
        for i in range(n_sequences):
            z = torch.randn(1, noise_dim, device=device)
            fake = netG(z)
            
            # Shape: (1, n_features, seq_length) -> (seq_length, n_features)
            fake_np = fake.squeeze(0).cpu().numpy().T
            generated_sequences.append(fake_np)
    
    # Stitch or keep separate
    if stitch:
        features = stitch_sequences(generated_sequences, overlap=overlap)
    else:
        features = np.vstack(generated_sequences)
    
    # Convert to OHLCV
    df = features_to_ohlcv(
        features,
        params,
        initial_price=initial_price,
        timeframe_seconds=timeframe_seconds,
        start_timestamp=start_timestamp
    )
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic OHLCV data')
    
    # Model
    parser.add_argument('--checkpoint_dir', required=True, 
                       help='Path to checkpoint directory')
    parser.add_argument('--checkpoint_name', default='netG_best.pth',
                       help='Checkpoint file name')
    
    # Generation
    parser.add_argument('--n_sequences', type=int, default=0,
                       help='Number of sequences to generate (0=use --duration instead)')
    parser.add_argument('--duration', type=str, default='3m',
                       help='Duration of data to generate: 1w, 2w, 1m, 3m, 6m, 1y (e.g., 3m = 3 months)')
    parser.add_argument('--n_datasets', type=int, default=1,
                       help='Number of separate datasets to generate')
    parser.add_argument('--initial_price', type=float, default=50000.0,
                       help='Starting price (BTC ~50000, ETH ~3000)')
    parser.add_argument('--timeframe', type=int, default=300,
                       help='Candle timeframe in seconds (300=5min)')
    parser.add_argument('--no_stitch', action='store_true',
                       help='Do not stitch sequences (keep separate)')
    parser.add_argument('--overlap', type=int, default=8,
                       help='Overlap for stitching sequences')
    
    # Output
    parser.add_argument('--output_dir', default='../testdata/synthetic',
                       help='Output directory for synthetic data')
    parser.add_argument('--output_prefix', default='synthetic',
                       help='Output file prefix')
    
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Setup device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint_dir}...")
    netG, train_info = load_model(args.checkpoint_dir, args.checkpoint_name)
    netG.to(device)
    
    seq_length = train_info['seq_length']
    
    # Calculate n_sequences from duration if not specified
    if args.n_sequences <= 0:
        # Parse duration string (1w, 2w, 1m, 3m, 6m, 1y)
        duration_map = {
            '1w': 7, '2w': 14, '1m': 30, '2m': 60, '3m': 90,
            '4m': 120, '6m': 180, '9m': 270, '1y': 365
        }
        days = duration_map.get(args.duration.lower(), 90)
        
        # Calculate candles needed (5-min = 288 candles/day)
        candles_per_day = 24 * 60 * 60 // args.timeframe
        total_candles = days * candles_per_day
        
        # Calculate sequences needed (accounting for overlap)
        effective_seq_len = seq_length - args.overlap
        args.n_sequences = (total_candles // effective_seq_len) + 1
        
        print(f"Duration: {args.duration} = {days} days = {total_candles:,} candles")
    
    expected_candles = args.n_sequences * seq_length - (args.n_sequences - 1) * args.overlap
    print(f"Will generate ~{expected_candles:,} candles per dataset ({args.n_sequences} sequences)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate datasets
    for i in range(args.n_datasets):
        print(f"\nGenerating dataset {i+1}/{args.n_datasets}...")
        
        # Use different random seed for each dataset
        if args.seed is not None:
            seed = args.seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Random start time within last year
        start_ts = int(datetime.now().timestamp()) - np.random.randint(0, 365 * 24 * 3600)
        
        df = generate_synthetic_dataset(
            netG, train_info,
            n_sequences=args.n_sequences,
            initial_price=args.initial_price,
            timeframe_seconds=args.timeframe,
            start_timestamp=start_ts,
            stitch=not args.no_stitch,
            overlap=args.overlap,
            device=device
        )
        
        # Save
        output_file = os.path.join(
            args.output_dir, 
            f'{args.output_prefix}_{i:04d}.csv'
        )
        df.to_csv(output_file, index=False)
        print(f"  Saved {len(df)} candles to {output_file}")
        
        # Print some statistics
        returns = np.log(df['close'].values[1:] / df['close'].values[:-1])
        print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"  Mean return: {returns.mean():.6f}, Std: {returns.std():.6f}")
        print(f"  Volume range: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
    
    print(f"\nGeneration complete! Files saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
