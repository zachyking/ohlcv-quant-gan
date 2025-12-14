#!/usr/bin/env python3
"""
Quick test script to verify the OHLCV GAN pipeline works.
Uses a tiny subset of data for fast iteration.
"""

import os
import numpy as np
import torch
from models.ohlcv_models import OHLCVGenerator, OHLCVDiscriminator

def main():
    print("=" * 50)
    print("Quick OHLCV GAN Test")
    print("=" * 50)
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    
    # Load a tiny subset of data
    data_path = 'preprocessed/BTCUSDT_5m_sequences.npy'
    if not os.path.exists(data_path):
        print(f"✗ Data not found at {data_path}")
        print("  Run: python preprocess_ohlcv.py ../testdata/BTCUSDT_5m_365d.csv --output_dir preprocessed --name BTCUSDT_5m")
        return
    
    print(f"\nLoading data from {data_path}...")
    sequences = np.load(data_path)
    print(f"  Full dataset: {sequences.shape}")
    
    # Use only 500 sequences for quick test
    n_samples = min(500, len(sequences))
    sequences = sequences[:n_samples]
    print(f"  Using subset: {sequences.shape}")
    
    n_samples, n_features, seq_length = sequences.shape
    
    # Create small models
    print("\nCreating models...")
    netG = OHLCVGenerator(
        noise_dim=32,
        hidden_dim=64,  # Smaller
        output_dim=n_features,
        n_layers=3,     # Fewer layers
        seq_length=seq_length,
        use_attention=False  # Faster without attention
    ).to(device)
    
    netD = OHLCVDiscriminator(
        input_dim=n_features,
        hidden_dim=64,
        n_layers=3,
        use_attention=False
    ).to(device)
    
    print(f"  Generator: {sum(p.numel() for p in netG.parameters()):,} params")
    print(f"  Discriminator: {sum(p.numel() for p in netD.parameters()):,} params")
    
    # Convert data to tensor
    data_tensor = torch.FloatTensor(sequences).to(device)
    
    # Quick training loop (3 epochs, no fancy stuff)
    print("\nTraining (3 quick epochs)...")
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.001)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.001)
    
    batch_size = 32
    n_batches = n_samples // batch_size
    
    for epoch in range(3):
        epoch_loss_D = 0
        epoch_loss_G = 0
        
        for i in range(n_batches):
            # Get batch
            idx = i * batch_size
            real = data_tensor[idx:idx+batch_size]
            
            # Train Discriminator
            netD.zero_grad()
            real_out = netD(real)
            
            z = torch.randn(batch_size, 32, device=device)
            fake = netG(z).detach()
            fake_out = netD(fake)
            
            # Simple WGAN loss (no GP for speed)
            loss_D = fake_out.mean() - real_out.mean()
            loss_D.backward()
            optimizerD.step()
            
            # Clip weights (WGAN)
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # Train Generator
            netG.zero_grad()
            z = torch.randn(batch_size, 32, device=device)
            fake = netG(z)
            fake_out = netD(fake)
            loss_G = -fake_out.mean()
            loss_G.backward()
            optimizerG.step()
            
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
        
        print(f"  Epoch {epoch+1}/3 | D: {epoch_loss_D/n_batches:.4f} | G: {epoch_loss_G/n_batches:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    netG.eval()
    with torch.no_grad():
        z = torch.randn(1, 32, device=device)
        fake = netG(z)
        print(f"  Generated shape: {fake.shape}")
        print(f"  Generated range: [{fake.min().item():.3f}, {fake.max().item():.3f}]")
    
    # Quick OHLCV reconstruction test
    print("\nTesting OHLCV reconstruction...")
    from generate_ohlcv import features_to_ohlcv
    import json
    
    with open('preprocessed/BTCUSDT_5m_params.json', 'r') as f:
        params = json.load(f)
    
    fake_np = fake.squeeze(0).cpu().numpy().T  # (seq_length, n_features)
    df = features_to_ohlcv(fake_np, params, initial_price=50000)
    
    print(f"  Generated {len(df)} candles")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Sample candle:")
    print(df.head(3).to_string(index=False))
    
    print("\n" + "=" * 50)
    print("✓ Quick test PASSED! Pipeline works.")
    print("=" * 50)
    print("\nFor full training, run:")
    print("  python train_ohlcv.py --name BTCUSDT_5m --epochs 50 --batch_size 128")


if __name__ == '__main__':
    main()
