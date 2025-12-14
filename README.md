# OHLCV QuantGAN - Synthetic Market Data Generator

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Generate realistic synthetic OHLCV (Open, High, Low, Close, Volume) data for backtesting trading strategies without overfitting to historical data.

## Why This Matters for S/R and Breakout Strategies

Traditional synthetic data generators (like the original QuantGAN) only generate **log returns** - a single column. But for support/resistance and breakout strategies, you need:

- **Full OHLC structure** to detect S/R levels (highs and lows matter!)
- **Candle patterns** (wicks, body ratios) for rejection signals
- **Volume** for breakout confirmation
- **Volatility clustering** (periods of calm followed by explosions)

This modified QuantGAN generates **6 correlated features** that preserve these relationships:

| Feature | What It Captures |
|---------|------------------|
| `log_return` | Price movement direction and magnitude |
| `body_ratio` | Bullish/bearish sentiment (-1 to 1) |
| `upper_wick` | Rejection from highs (resistance) |
| `lower_wick` | Rejection from lows (support) |
| `range_pct` | Volatility / candle size |
| `volume_norm` | Volume patterns (spikes on breakouts) |

## Quick Start

### 1. Install Dependencies

```bash
cd QuantGAN
pip install -r requirements.txt
```

For GPU training (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# or
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

### 2. Preprocess Your Data

```bash
# Preprocess BTC data
python preprocess_ohlcv.py ../testdata/BTCUSDT_5m_365d.csv \
    --output_dir preprocessed \
    --seq_length 64

# Preprocess ETH data
python preprocess_ohlcv.py ../testdata/ETHUSDT_5m_365d.csv \
    --output_dir preprocessed \
    --seq_length 64
```

This creates:
- `preprocessed/BTCUSDT_5m_365d_sequences.npy` - Training sequences
- `preprocessed/BTCUSDT_5m_365d_features.csv` - Feature data
- `preprocessed/BTCUSDT_5m_365d_params.json` - Transform parameters

### 3. Train the GAN

```bash
# Train on BTC (GPU recommended)
python train_ohlcv.py --name BTCUSDT_5m_365d \
    --epochs 200 \
    --batch_size 64 \
    --cuda

# Train on ETH
python train_ohlcv.py --name ETHUSDT_5m_365d \
    --epochs 200 \
    --batch_size 64 \
    --cuda
```

Training creates checkpoints in `checkpoints/<name>_<timestamp>/`:
- `netG_best.pth` - Best generator (lowest loss)
- `netG_final.pth` - Final generator
- `train_info.json` - Model configuration

**Training time estimates:**
- CPU: ~2-4 hours for 200 epochs
- GPU (RTX 3080+): ~15-30 minutes

### 4. Generate Synthetic Data

```bash
# Generate 10 synthetic BTC datasets (each ~6400 candles ≈ 22 days of 5m data)
python generate_ohlcv.py \
    --checkpoint_dir checkpoints/BTCUSDT_5m_365d_tcn_<timestamp> \
    --n_sequences 100 \
    --n_datasets 10 \
    --initial_price 50000 \
    --output_dir ../testdata/synthetic \
    --output_prefix BTCUSDT_synthetic

# Generate synthetic ETH
python generate_ohlcv.py \
    --checkpoint_dir checkpoints/ETHUSDT_5m_365d_tcn_<timestamp> \
    --n_sequences 100 \
    --n_datasets 10 \
    --initial_price 3500 \
    --output_dir ../testdata/synthetic \
    --output_prefix ETHUSDT_synthetic
```

## Output Format

Generated files match your existing testdata format:

```csv
time,open,high,low,close,volume
1702000000,50000.00,50125.50,49875.25,50100.00,123.45678
1702000300,50100.00,50200.00,50050.00,50175.50,145.67890
...
```

## Architecture

The GAN uses **Temporal Convolutional Networks (TCN)** with:

- **Dilated convolutions** for capturing long-range patterns (S/R levels that persist)
- **Self-attention** for learning global dependencies (market regime awareness)
- **Spectral normalization** for stable WGAN-GP training
- **OHLCV constraint loss** to ensure valid candle relationships

```
Generator:
  Noise (64-dim) → Project → [ResBlock + Attention] × 4 → OHLCV Features (6-dim)

Discriminator:  
  OHLCV Features (6-dim) → [ResBlock + Attention] × 4 → Pool → Real/Fake Score
```

## Advanced Usage

### Hyperparameter Tuning

```bash
python train_ohlcv.py --name BTCUSDT_5m_365d \
    --model_type tcn \       # or 'lstm' for slower but potentially better long-term patterns
    --hidden_dim 256 \       # larger = more capacity, slower training
    --n_layers 6 \           # more layers = longer temporal receptive field
    --noise_dim 128 \        # more noise = more diversity in generations
    --lr_g 0.00005 \         # lower LR if training is unstable
    --n_critic 5 \           # D updates per G update (WGAN-GP)
    --epochs 500
```

### Resume Training

```bash
python train_ohlcv.py --name BTCUSDT_5m_365d \
    --resume_G checkpoints/.../netG_epoch_100.pth \
    --resume_D checkpoints/.../netD_epoch_100.pth \
    --epochs 200
```

### Generate with Specific Seeds (Reproducibility)

```bash
python generate_ohlcv.py \
    --checkpoint_dir checkpoints/... \
    --n_datasets 5 \
    --seed 42  # Same seed = same synthetic data
```

## How to Use with Your Backtest

Update your Go backtest to load synthetic data:

```go
// In pkg/data/loader.go or similar
func LoadSyntheticData(path string) ([]types.Candle, error) {
    // Same format as real data - just point to synthetic files
    return LoadCSV(path)
}

// Test strategy robustness
syntheticFiles := []string{
    "testdata/synthetic/BTCUSDT_synthetic_0000.csv",
    "testdata/synthetic/BTCUSDT_synthetic_0001.csv",
    // ... more synthetic datasets
}

for _, file := range syntheticFiles {
    candles, _ := LoadSyntheticData(file)
    result := backtest.Run(strategy, candles)
    // Strategy should perform similarly on synthetic vs real
    // Large divergence = possible overfitting
}
```

## Tips for Better Synthetic Data

1. **Train on more data**: The more real data, the better the GAN learns patterns
2. **Longer sequences**: Try `--seq_length 128` for longer temporal dependencies
3. **Multiple assets**: Train separate models for BTC and ETH, or combine them
4. **Validate visually**: Plot synthetic vs real data to check for obvious issues
5. **Statistical tests**: Compare return distributions, autocorrelations, etc.

## Troubleshooting

**Training is unstable (loss oscillating wildly)**
- Reduce learning rates: `--lr_g 0.00005 --lr_d 0.00005`
- Increase critic steps: `--n_critic 10`

**Generated data looks too smooth/boring**
- Train longer
- Increase noise dimension: `--noise_dim 128`
- Check if your preprocessing lost volatility

**Generated data has invalid OHLC (high < low)**
- This should be handled by `features_to_ohlcv()` but if it persists, increase `--lambda_constraint`

**Out of memory on GPU**
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Reduce hidden dimension: `--hidden_dim 64`

## License

MIT License - see [LICENSE](LICENSE) for details. Use freely, but keep the attribution!
