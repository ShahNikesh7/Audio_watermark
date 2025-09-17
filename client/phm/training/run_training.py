"""
Simple training script for SoundSafeAI watermarking model.
Run this script to train the watermarking model on your audio data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from client.phm.training.watermark_trainer import WatermarkTrainer, create_training_config


def main():
    parser = argparse.ArgumentParser(description='Train SoundSafeAI watermarking model')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Directory containing validation audio files (optional)')
    parser.add_argument('--output-dir', type=str, default='experiments/run1',
                       help='Directory to save model checkpoints and logs')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Audio sample rate for processing')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device (cuda or cpu)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')

    args = parser.parse_args()

    # Check if training directory exists
    if not os.path.exists(args.train_dir):
        print(f"Training directory not found: {args.train_dir}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create training configuration
    print("📋 Creating training configuration...")
    config = create_training_config(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        experiment_name=f"watermark_training_{os.path.basename(args.output_dir)}"
    )

    print("🔧 Training Configuration:")
    print(f"  • Train dir: {config['data']['train_dir']}")
    print(f"  • Val dir: {config['data']['val_dir'] or 'None'}")
    print(f"  • Sample rate: {config['data']['sample_rate']} Hz")
    print(f"  • Batch size: {config['data']['batch_size']}")
    print(f"  • Learning rate: {config['optimizer']['learning_rate']}")
    print(f"  • Epochs: {config['training']['epochs']}")
    print(f"  • Device: {args.device}")
    print(f"  • Output dir: {args.output_dir}")

    # Initialize trainer
    print("Initializing WatermarkTrainer...")
    try:
        trainer = WatermarkTrainer(
            config=config,
            device=args.device,
            use_wandb=args.use_wandb
        )
        print("Trainer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")
        return 1

    # Start training
    print("🏃 Starting training...")
    try:
        trainer.train(
            num_epochs=config['training']['epochs'],
            save_dir=args.output_dir
        )
        print("Training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
