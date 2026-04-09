#!/bin/bash
#SBATCH -J ece570-vit
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00

# Submit from the Gilbreth login node with something like:
#   sbatch -A <your_account> -p <gilbreth_a100_partition> \
#     --export=ALL,ROOT_DIR=$PWD,CONFIG=$PWD/configs/paper_core.json,VENV_DIR=$HOME/.venvs/ece570-vit \
#     scripts/slurm/gilbreth_gpu.sbatch
#
# Check your valid account and A100 partition names before submitting:
#   slist
#   sinfo -o "%P %G"

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?Set ROOT_DIR to the repo root when submitting.}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/paper_core.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/gilbreth_$(date +%Y%m%d_%H%M%S)}"
VENV_DIR="${VENV_DIR:?Set VENV_DIR to the virtualenv created by scripts/setup_env.sh.}"
HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"

mkdir -p "$HF_HOME" "$OUTPUT_ROOT"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

python scripts/run_experiments.py \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --continue-on-error

python scripts/summarize_results.py --results-root "$OUTPUT_ROOT"
