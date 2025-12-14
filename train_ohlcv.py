#!/usr/bin/env python3
"""
Training script for OHLCV QuantGAN.

Trains a GAN to generate realistic OHLCV features that preserve
the patterns needed for support/resistance and breakout detection.

Usage:
    python train_ohlcv.py --data_dir preprocessed --name BTCUSDT_5m_365d --epochs 200
    
The model will be saved to checkpoints/ and can be used with generate_ohlcv.py
"""

import os
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.ohlcv_models import (
    OHLCVGenerator, 
    OHLCVDiscriminator,
    OHLCVGeneratorLSTM,
    OHLCVDiscriminatorLSTM,
    gradient_penalty,
    ohlcv_constraint_loss
)


def load_preprocessed(data_dir: str, name: str):
    """Load preprocessed data."""
    sequences = np.load(os.path.join(data_dir, f'{name}_sequences.npy'))
    with open(os.path.join(data_dir, f'{name}_params.json'), 'r') as f:
        params = json.load(f)
    return sequences, params


def get_device(prefer_gpu=True):
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)."""
    if not prefer_gpu:
        return torch.device('cpu')
    
    # Check for Apple Silicon MPS
    if torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    return torch.device('cpu')


def train(args):
    """Main training loop."""
    
    # Setup device
    device = get_device(prefer_gpu=args.gpu)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading preprocessed data from {args.data_dir}/{args.name}...")
    sequences, params = load_preprocessed(args.data_dir, args.name)
    
    # Subsample if requested (for faster training/testing)
    if args.max_samples > 0 and args.max_samples < len(sequences):
        # Random subsample for diversity
        indices = np.random.choice(len(sequences), args.max_samples, replace=False)
        sequences = sequences[indices]
        print(f"Subsampled to {len(sequences)} sequences")
    
    n_samples, n_features, seq_length = sequences.shape
    print(f"Using {n_samples} sequences with {n_features} features, length {seq_length}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(torch.FloatTensor(sequences))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=args.workers, drop_last=True)
    
    # Create models
    if args.model_type == 'tcn':
        netG = OHLCVGenerator(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            output_dim=n_features,
            n_layers=args.n_layers,
            seq_length=seq_length,
            use_attention=args.use_attention
        ).to(device)
        
        netD = OHLCVDiscriminator(
            input_dim=n_features,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            use_attention=args.use_attention
        ).to(device)
    else:  # lstm
        netG = OHLCVGeneratorLSTM(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            output_dim=n_features,
            n_layers=args.n_layers,
            seq_length=seq_length
        ).to(device)
        
        netD = OHLCVDiscriminatorLSTM(
            input_dim=n_features,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers
        ).to(device)
    
    print(f"\nGenerator: {sum(p.numel() for p in netG.parameters())} parameters")
    print(f"Discriminator: {sum(p.numel() for p in netD.parameters())} parameters")
    
    # Load checkpoints if resuming
    start_epoch = 0
    if args.resume_G and os.path.exists(args.resume_G):
        netG.load_state_dict(torch.load(args.resume_G, map_location=device))
        print(f"Loaded generator from {args.resume_G}")
    if args.resume_D and os.path.exists(args.resume_D):
        netD.load_state_dict(torch.load(args.resume_D, map_location=device))
        print(f"Loaded discriminator from {args.resume_D}")
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args.epochs)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=args.epochs)
    
    # Create output directory
    run_name = f"{args.name}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training log
    log_file = open(os.path.join(checkpoint_dir, 'training.log'), 'w')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to {checkpoint_dir}/")
    
    best_loss_G = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        loss_D_sum = 0
        loss_G_sum = 0
        gp_sum = 0
        n_batches = 0
        
        for i, (real_data,) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            for _ in range(args.n_critic):
                netD.zero_grad()
                
                # Real data
                real_validity = netD(real_data)
                
                # Fake data
                z = torch.randn(batch_size, args.noise_dim, device=device)
                fake_data = netG(z).detach()
                fake_validity = netD(fake_data)
                
                # WGAN-GP loss
                loss_D = fake_validity.mean() - real_validity.mean()
                
                # Gradient penalty
                gp = gradient_penalty(netD, real_data, fake_data, device)
                loss_D_total = loss_D + args.lambda_gp * gp
                
                loss_D_total.backward()
                optimizerD.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            netG.zero_grad()
            
            z = torch.randn(batch_size, args.noise_dim, device=device)
            fake_data = netG(z)
            fake_validity = netD(fake_data)
            
            # Generator loss: want discriminator to think fake is real
            loss_G = -fake_validity.mean()
            
            # Add OHLCV constraint loss
            constraint_loss = ohlcv_constraint_loss(fake_data)
            loss_G_total = loss_G + args.lambda_constraint * constraint_loss
            
            loss_G_total.backward()
            optimizerG.step()
            
            # Accumulate for logging
            loss_D_sum += loss_D.item()
            loss_G_sum += loss_G.item()
            gp_sum += gp.item()
            n_batches += 1
            
            # Print progress
            if (i + 1) % args.log_interval == 0:
                print(f"\r  Batch {i+1}/{len(dataloader)} | "
                      f"D: {loss_D.item():.4f} | G: {loss_G.item():.4f} | "
                      f"GP: {gp.item():.4f}", end='')
        
        # Epoch done
        epoch_time = time.time() - epoch_start
        avg_loss_D = loss_D_sum / n_batches
        avg_loss_G = loss_G_sum / n_batches
        avg_gp = gp_sum / n_batches
        
        log_msg = (f"Epoch [{epoch+1}/{args.epochs}] | "
                   f"Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | "
                   f"GP: {avg_gp:.4f} | Time: {epoch_time:.1f}s")
        print(f"\n{log_msg}")
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        # Update learning rates
        schedulerG.step()
        schedulerD.step()
        
        # Save checkpoints
        if (epoch + 1) % args.save_interval == 0:
            torch.save(netG.state_dict(), 
                      os.path.join(checkpoint_dir, f'netG_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(),
                      os.path.join(checkpoint_dir, f'netD_epoch_{epoch+1}.pth'))
            print(f"  Saved checkpoint at epoch {epoch+1}")
        
        # Save best model
        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            torch.save(netG.state_dict(),
                      os.path.join(checkpoint_dir, 'netG_best.pth'))
            torch.save(netD.state_dict(),
                      os.path.join(checkpoint_dir, 'netD_best.pth'))
    
    # Save final models
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'netD_final.pth'))
    
    # Save training params for generation
    train_info = {
        'n_features': n_features,
        'seq_length': seq_length,
        'noise_dim': args.noise_dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'model_type': args.model_type,
        'use_attention': args.use_attention,
        'data_params': params
    }
    with open(os.path.join(checkpoint_dir, 'train_info.json'), 'w') as f:
        json.dump(train_info, f, indent=2)
    
    log_file.close()
    print(f"\nTraining complete! Models saved to {checkpoint_dir}/")
    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description='Train OHLCV QuantGAN')
    
    # Data
    parser.add_argument('--data_dir', default='preprocessed', help='Directory with preprocessed data')
    parser.add_argument('--name', required=True, help='Dataset name (e.g., BTCUSDT_5m_365d)')
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples to use (0=all, useful for quick testing)')
    
    # Model architecture
    parser.add_argument('--model_type', default='tcn', choices=['tcn', 'lstm'],
                       help='Model architecture type')
    parser.add_argument('--noise_dim', type=int, default=64, help='Noise vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use self-attention (TCN only)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='Discriminator learning rate')
    parser.add_argument('--n_critic', type=int, default=5, help='Discriminator updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Gradient penalty coefficient')
    parser.add_argument('--lambda_constraint', type=float, default=0.1, help='OHLCV constraint loss weight')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume_G', default='', help='Path to generator checkpoint to resume')
    parser.add_argument('--resume_D', default='', help='Path to discriminator checkpoint to resume')
    
    # Misc
    parser.add_argument('--gpu', action='store_true', help='Use GPU (MPS on Apple Silicon, CUDA on NVIDIA)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if GPU available')
    parser.add_argument('--workers', type=int, default=0, help='Data loader workers (0 for MPS compatibility)')
    parser.add_argument('--log_interval', type=int, default=50, help='Log every N batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Handle CPU override
    if args.cpu:
        args.gpu = False
    else:
        args.gpu = True  # Default to GPU
    
    train(args)


if __name__ == '__main__':
    main()
