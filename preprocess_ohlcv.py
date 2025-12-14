#!/usr/bin/env python3
"""
OHLCV Preprocessor for QuantGAN

Converts raw OHLCV data into normalized features suitable for GAN training.
The key insight: instead of just log returns, we model multiple aspects
of the candle structure to preserve S/R and breakout patterns.

Features generated:
1. log_return: log(close_t / close_{t-1}) - main price movement
2. body_ratio: (close - open) / (high - low) - candle body position [-1, 1]
3. upper_wick: (high - max(open,close)) / (high - low) - upper wick ratio [0, 1]
4. lower_wick: (min(open,close) - low) / (high - low) - lower wick ratio [0, 1]
5. range_pct: (high - low) / close - volatility measure
6. volume_norm: normalized volume (z-score with rolling window)

These features preserve the structure needed for S/R and breakout detection.
"""

import pandas as pd
import numpy as np
from scipy.special import lambertw
import os
from typing import Tuple, Optional


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """Load OHLCV CSV file."""
    df = pd.read_csv(filepath)
    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def compute_ohlcv_features(df: pd.DataFrame, volume_window: int = 100) -> pd.DataFrame:
    """
    Convert OHLCV to normalized features that preserve candle structure.
    
    Args:
        df: DataFrame with columns [time, open, high, low, close, volume]
        volume_window: Rolling window for volume normalization
    
    Returns:
        DataFrame with normalized features
    """
    features = pd.DataFrame()
    features['time'] = df['time']
    
    # 1. Log returns (primary price movement)
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Candle range (for volatility)
    candle_range = df['high'] - df['low']
    # Avoid division by zero
    candle_range = candle_range.replace(0, np.nan)
    
    # 3. Body ratio: where is the body positioned in the candle? [-1, 1]
    # 1 = fully bullish (close at high, open at low)
    # -1 = fully bearish (close at low, open at high)
    body = df['close'] - df['open']
    features['body_ratio'] = body / candle_range
    
    # 4. Upper wick ratio [0, 1]
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    features['upper_wick'] = upper_wick / candle_range
    
    # 5. Lower wick ratio [0, 1]
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    features['lower_wick'] = lower_wick / candle_range
    
    # 6. Range percentage (volatility measure)
    features['range_pct'] = candle_range / df['close']
    
    # 7. Volume (z-score normalized with rolling window)
    vol_mean = df['volume'].rolling(window=volume_window, min_periods=1).mean()
    vol_std = df['volume'].rolling(window=volume_window, min_periods=1).std()
    vol_std = vol_std.replace(0, 1)  # Avoid division by zero
    features['volume_norm'] = (df['volume'] - vol_mean) / vol_std
    
    # Drop first row (NaN from log return) and any rows with NaN
    features = features.dropna()
    
    return features


def gaussianize_series(x: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Apply inverse Lambert W transform to make distribution more Gaussian.
    This helps with heavy-tailed financial data.
    
    Returns:
        Tuple of (transformed data, parameters for inverse transform)
    """
    # Normalize first
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x == 0:
        std_x = 1
    x_norm = (x - mean_x) / std_x
    
    # Estimate delta parameter (heavy-tailedness)
    # Using simple moment-based estimation
    kurtosis = np.mean(x_norm**4) - 3
    delta = max(0, min(1, kurtosis / 10))  # Clamp to [0, 1]
    
    # Apply inverse Lambert W transform (Gaussianize)
    # For small delta, this is approximately: z = x * exp(-delta/2 * x^2)
    if delta > 0.01:
        z = x_norm * np.exp(-delta / 2 * x_norm**2)
    else:
        z = x_norm
    
    # Normalize again
    mean_z = np.mean(z)
    std_z = np.std(z)
    if std_z == 0:
        std_z = 1
    z_norm = (z - mean_z) / std_z
    
    params = {
        'mean_x': mean_x,
        'std_x': std_x,
        'delta': delta,
        'mean_z': mean_z,
        'std_z': std_z
    }
    
    return z_norm, params


def inverse_gaussianize(z_norm: np.ndarray, params: dict) -> np.ndarray:
    """
    Reverse the Gaussianization transform.
    """
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


def preprocess_for_gan(df: pd.DataFrame, 
                       gaussianize: bool = True,
                       feature_cols: Optional[list] = None) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Full preprocessing pipeline for GAN training.
    
    Args:
        df: Raw OHLCV DataFrame
        gaussianize: Whether to apply Lambert W transform
        feature_cols: Which feature columns to use (default: all)
    
    Returns:
        Tuple of (normalized array, original features df, transform parameters)
    """
    # Compute features
    features = compute_ohlcv_features(df)
    
    if feature_cols is None:
        feature_cols = ['log_return', 'body_ratio', 'upper_wick', 
                       'lower_wick', 'range_pct', 'volume_norm']
    
    # Store parameters for each feature
    all_params = {}
    normalized_data = []
    
    for col in feature_cols:
        data = features[col].values
        
        if gaussianize:
            norm_data, params = gaussianize_series(data)
        else:
            # Just normalize to [-1, 1]
            min_val, max_val = data.min(), data.max()
            if max_val == min_val:
                norm_data = np.zeros_like(data)
            else:
                norm_data = 2 * (data - min_val) / (max_val - min_val) - 1
            params = {'min': min_val, 'max': max_val, 'gaussianize': False}
        
        params['gaussianize'] = gaussianize
        all_params[col] = params
        normalized_data.append(norm_data)
    
    # Stack into array: shape (n_samples, n_features)
    normalized_array = np.stack(normalized_data, axis=1)
    
    return normalized_array, features, all_params


def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Create overlapping sequences for GAN training.
    
    Args:
        data: Shape (n_samples, n_features)
        seq_length: Length of each sequence
    
    Returns:
        Array of shape (n_sequences, n_features, seq_length)
    """
    n_samples, n_features = data.shape
    n_sequences = n_samples - seq_length + 1
    
    sequences = np.zeros((n_sequences, n_features, seq_length))
    for i in range(n_sequences):
        sequences[i] = data[i:i+seq_length].T
    
    return sequences


def save_preprocessed(output_dir: str, 
                     name: str,
                     sequences: np.ndarray, 
                     features: pd.DataFrame,
                     params: dict):
    """Save preprocessed data and parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sequences
    np.save(os.path.join(output_dir, f'{name}_sequences.npy'), sequences)
    
    # Save original features
    features.to_csv(os.path.join(output_dir, f'{name}_features.csv'), index=False)
    
    # Save parameters as JSON
    import json
    # Convert numpy types to Python types
    params_serializable = {}
    for k, v in params.items():
        params_serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                   for kk, vv in v.items()}
    
    with open(os.path.join(output_dir, f'{name}_params.json'), 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Saved preprocessed data to {output_dir}/")
    print(f"  - {name}_sequences.npy: shape {sequences.shape}")
    print(f"  - {name}_features.csv: {len(features)} rows")
    print(f"  - {name}_params.json")


def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess OHLCV data for QuantGAN')
    parser.add_argument('input_file', help='Path to input OHLCV CSV')
    parser.add_argument('--output_dir', default='preprocessed', help='Output directory')
    parser.add_argument('--name', default=None, help='Output name prefix (default: from filename)')
    parser.add_argument('--seq_length', type=int, default=64, help='Sequence length (power of 2 recommended)')
    parser.add_argument('--no_gaussianize', action='store_true', help='Skip Lambert W transform')
    
    args = parser.parse_args()
    
    # Determine output name
    if args.name is None:
        args.name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    print(f"Loading {args.input_file}...")
    df = load_ohlcv(args.input_file)
    print(f"Loaded {len(df)} rows")
    
    print("Computing features...")
    normalized, features, params = preprocess_for_gan(
        df, 
        gaussianize=not args.no_gaussianize
    )
    print(f"Computed {normalized.shape[1]} features for {normalized.shape[0]} samples")
    
    print(f"Creating sequences of length {args.seq_length}...")
    sequences = create_sequences(normalized, args.seq_length)
    print(f"Created {sequences.shape[0]} sequences")
    
    save_preprocessed(args.output_dir, args.name, sequences, features, params)


if __name__ == '__main__':
    main()
