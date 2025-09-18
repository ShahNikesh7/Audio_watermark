#!/usr/bin/env python3
"""
Distributed training script for SoundSafeAI watermarking model.
Supports multi-GPU, multi-node training with PyTorch DistributedDataParallel.
"""

import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from client.phm.training.watermark_trainer import WatermarkTrainer, create_training_config


def setup_distributed(rank: int, world_size: int, master_addr: str = 'localhost', master_port: str = '12345'):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, config: dict, output_dir: str):
    """Worker function for distributed training"""
    try:
        setup_distributed(rank, world_size, config['master_addr'], config['master_port'])

        # Adjust batch size for distributed training
        config['data']['batch_size'] = config['data']['batch_size'] // world_size

        # Create rank-specific output directory
        rank_output_dir = f"{output_dir}/rank_{rank}"
        os.makedirs(rank_output_dir, exist_ok=True)

        # Initialize trainer
        trainer = WatermarkTrainer(
            config=config,
            device=f'cuda:{rank}',
            use_wandb=False  # Disable wandb for distributed training
        )

        if rank == 0:
            print(f"Starting distributed training on {world_size} GPUs...")
            print(f"Output directory: {output_dir}")
            print(f"Batch size per GPU: {config['data']['batch_size']}")
            print(f"Total epochs: {config['training']['epochs']}")

        # Train the model
        trainer.train(
            num_epochs=config['training']['epochs'],
            save_dir=rank_output_dir
        )

        if rank == 0:
            print("Distributed training completed!")

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    finally:
        cleanup_distributed()


def create_distributed_config(args):
    """Create configuration for distributed training"""
    config = create_training_config(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        experiment_name=args.experiment_name
    )

    # Add distributed training parameters
    config['distributed'] = {
        'world_size': args.world_size,
        'master_addr': args.master_addr,
        'master_port': args.master_port,
        'backend': 'nccl'
    }

    return config


def main():
    parser = argparse.ArgumentParser(description='Distributed training for SoundSafeAI watermarking model')

    # Required arguments
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Directory containing training audio files')

    # Optional arguments
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Directory containing validation audio files')
    parser.add_argument('--output-dir', type=str, default='experiments/distributed_run',
                       help='Base output directory for checkpoints and logs')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Audio sample rate for processing')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Total batch size (will be divided among GPUs)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--experiment-name', type=str, default='distributed_watermark_training',
                       help='Experiment name')

    # Distributed training arguments
    parser.add_argument('--world-size', type=int, default=8,
                       help='Total number of GPUs across all nodes')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='IP address of master node')
    parser.add_argument('--master-port', type=str, default='12345',
                       help='Port for master node')

    args = parser.parse_args()

    # Create unique output directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Create distributed configuration
    print("📋 Creating distributed training configuration...")
    config = create_distributed_config(args)

    # Save configuration
    config_path = f"{output_dir}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("🔧 Distributed Training Configuration:")
    print(f"  • World size: {args.world_size} GPUs")
    print(f"  • Master: {args.master_addr}:{args.master_port}")
    print(f"  • Train dir: {args.train_dir}")
    print(f"  • Val dir: {args.val_dir or 'None'}")
    print(f"  • Output dir: {output_dir}")
    print(f"  • Batch size per GPU: {args.batch_size // args.world_size}")
    print(f"  • Learning rate: {args.learning_rate}")
    print(f"  • Epochs: {args.epochs}")

    # Check training data
    if not os.path.exists(args.train_dir):
        print(f"Training directory not found: {args.train_dir}")
        return 1

    # Count audio files
    import glob
    audio_patterns = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    audio_files = []
    for pattern in audio_patterns:
        audio_files.extend(glob.glob(os.path.join(args.train_dir, '**', pattern), recursive=True))

    print(f"Found {len(audio_files)} audio files in training directory")

    if len(audio_files) == 0:
        print("No audio files found in training directory")
        return 1

    # Launch distributed training
    print("Launching distributed training...")

    try:
        # Use torch.multiprocessing to spawn processes
        mp.spawn(
            train_worker,
            args=(args.world_size, config, output_dir),
            nprocs=args.world_size,
            join=True
        )

        print("Distributed training completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("Check checkpoints in rank-specific subdirectories")

    except Exception as e:
        print(f"Distributed training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
