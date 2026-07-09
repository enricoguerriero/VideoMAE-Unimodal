#!/usr/bin/env bash
#SBATCH --job-name=videomae-unimodal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.log

set -euo pipefail
mkdir -p logs

# module load cuda   # if your cluster uses environment modules
source activate videomae-unimodal   # or: source .venv/bin/activate

MODEL="${MODEL:-VideoMAE}"
python -m src.training --model "${MODEL}"
