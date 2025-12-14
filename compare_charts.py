#!/usr/bin/env python3
"""
Compare real vs synthetic OHLCV data with candlestick charts.
Creates a side-by-side visualization to assess synthetic data quality.
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """Load OHLCV data and convert to mplfinance format."""
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    
    # Convert timestamp to datetime index
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Date', inplace=True)
    
    # Rename columns for mplfinance
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute data to daily for cleaner visualization."""
    daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return daily


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute data to 4-hour candles."""
    h4 = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return h4


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute key statistics for comparison."""
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    return {
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'max_drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
        'price_range': (df['Close'].max() - df['Close'].min()) / df['Close'].mean(),
        'avg_range': ((df['High'] - df['Low']) / df['Close']).mean(),
    }


def plot_comparison(real_df: pd.DataFrame, synth_df: pd.DataFrame, 
                   output_path: str = 'comparison.png',
                   n_candles: int = 200,
                   timeframe: str = '4h'):
    """Create side-by-side candlestick comparison."""
    
    # Resample if needed
    if timeframe == 'daily':
        real_plot = resample_to_daily(real_df).tail(n_candles)
        synth_plot = resample_to_daily(synth_df).tail(n_candles)
        tf_label = 'Daily'
    elif timeframe == '4h':
        real_plot = resample_to_4h(real_df).tail(n_candles)
        synth_plot = resample_to_4h(synth_df).tail(n_candles)
        tf_label = '4H'
    else:
        real_plot = real_df.tail(n_candles)
        synth_plot = synth_df.tail(n_candles)
        tf_label = '5min'
    
    # Compute stats
    real_stats = compute_stats(real_plot)
    synth_stats = compute_stats(synth_plot)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Custom style
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350',
        edge='inherit',
        wick='inherit',
        volume={'up': '#26a69a', 'down': '#ef5350'}
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    
    # Create subplots manually for more control
    ax1 = fig.add_subplot(2, 2, 1)
    ax1v = ax1.twinx()
    ax2 = fig.add_subplot(2, 2, 2)
    ax2v = ax2.twinx()
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Plot candlesticks
    mpf.plot(real_plot, type='candle', ax=ax1, volume=ax1v, style=style)
    ax1.set_title(f'REAL BTC Data ({tf_label})', fontsize=14, fontweight='bold')
    # Make volume bars smaller by limiting y-axis
    ax1v.set_ylim(0, ax1v.get_ylim()[1] * 4)
    ax1v.set_alpha(0.3)
    
    mpf.plot(synth_plot, type='candle', ax=ax2, volume=ax2v, style=style)
    ax2.set_title(f'SYNTHETIC BTC Data ({tf_label})', fontsize=14, fontweight='bold')
    # Make volume bars smaller
    ax2v.set_ylim(0, ax2v.get_ylim()[1] * 4)
    ax2v.set_alpha(0.3)
    
    # Plot returns distribution
    real_returns = np.log(real_plot['Close'] / real_plot['Close'].shift(1)).dropna()
    synth_returns = np.log(synth_plot['Close'] / synth_plot['Close'].shift(1)).dropna()
    
    ax3.hist(real_returns, bins=50, alpha=0.7, label='Real', color='#2196f3', density=True)
    ax3.hist(synth_returns, bins=50, alpha=0.7, label='Synthetic', color='#ff9800', density=True)
    ax3.set_title('Log Returns Distribution', fontsize=12)
    ax3.set_xlabel('Log Return')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Stats comparison table
    stats_labels = ['Mean Return', 'Volatility (Std)', 'Skewness', 'Kurtosis', 
                   'Max Drawdown', 'Price Range %', 'Avg Candle Range']
    real_values = [real_stats['mean_return'], real_stats['std_return'], 
                  real_stats['skewness'], real_stats['kurtosis'],
                  real_stats['max_drawdown'], real_stats['price_range'],
                  real_stats['avg_range']]
    synth_values = [synth_stats['mean_return'], synth_stats['std_return'],
                   synth_stats['skewness'], synth_stats['kurtosis'],
                   synth_stats['max_drawdown'], synth_stats['price_range'],
                   synth_stats['avg_range']]
    
    ax4.axis('off')
    table_data = []
    for label, rv, sv in zip(stats_labels, real_values, synth_values):
        if 'Return' in label or 'Range' in label or 'Drawdown' in label:
            table_data.append([label, f'{rv:.6f}', f'{sv:.6f}'])
        else:
            table_data.append([label, f'{rv:.4f}', f'{sv:.4f}'])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Metric', 'Real', 'Synthetic'],
        cellLoc='center',
        loc='center',
        colColours=['#e3f2fd', '#e3f2fd', '#fff3e0']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Statistical Comparison', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comparison chart saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Compare real vs synthetic OHLCV data')
    parser.add_argument('--real', default='../testdata/BTCUSDT_5m_365d.csv',
                       help='Path to real OHLCV data')
    parser.add_argument('--synthetic', default='../testdata/synthetic/BTCUSDT_2month_0000.csv',
                       help='Path to synthetic OHLCV data')
    parser.add_argument('--output', default='comparison.png',
                       help='Output image path')
    parser.add_argument('--candles', type=int, default=200,
                       help='Number of candles to show')
    parser.add_argument('--timeframe', default='4h', choices=['5min', '4h', 'daily'],
                       help='Timeframe for visualization')
    
    args = parser.parse_args()
    
    print(f"Loading real data from {args.real}...")
    real_df = load_ohlcv(args.real)
    print(f"  Loaded {len(real_df)} candles")
    
    print(f"Loading synthetic data from {args.synthetic}...")
    synth_df = load_ohlcv(args.synthetic)
    print(f"  Loaded {len(synth_df)} candles")
    
    print(f"\nCreating comparison chart ({args.timeframe} timeframe)...")
    plot_comparison(real_df, synth_df, args.output, args.candles, args.timeframe)


if __name__ == '__main__':
    main()
