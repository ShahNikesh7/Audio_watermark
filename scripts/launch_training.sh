#!/bin/bash

# Training launch script for SoundSafeAI

set -e

echo "Starting SoundSafeAI training pipeline..."

# Check if data is available
if [ ! -d "data/processed" ]; then
    echo "Processed data not found. Running preprocessing..."
    python scripts/preprocess.py
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="soundsafeai"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Training configuration
EXPERIMENT_NAME="watermark_training_$(date +%Y%m%d_%H%M%S)"
echo "Experiment name: $EXPERIMENT_NAME"

# Create experiment directory
mkdir -p "experiments/runs/$EXPERIMENT_NAME"
cd "experiments/runs/$EXPERIMENT_NAME"

# Copy configuration
cp ../../../config/training_config.yaml .

echo "Training configuration:"
cat training_config.yaml

# Launch training
echo "Starting training..."

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    ../../../training/train.py \
    --config training_config.yaml \
    --experiment-name "$EXPERIMENT_NAME" \
    --output-dir . \
    --data-dir ../../../data/processed \
    --log-level INFO \
    --save-checkpoints \
    --validate-every 1000 \
    --early-stopping-patience 10

echo "Training completed!"
echo "Results saved in: experiments/runs/$EXPERIMENT_NAME"
echo "Logs available in: experiments/runs/$EXPERIMENT_NAME/logs"
echo "Models saved in: experiments/runs/$EXPERIMENT_NAME/models"

# Run evaluation
echo "Running evaluation..."
python ../../../evaluation/evaluate.py \
    --model-dir models \
    --data-dir ../../../data/processed \
    --output-dir evaluation_results

echo "Evaluation completed!"
echo "All results saved in: experiments/runs/$EXPERIMENT_NAME"
