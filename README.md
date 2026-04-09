# ECE 5700 Project: ViT Adapter Tuning on CIFAR-10

This repo now supports reproducible experiments for comparing four tuning strategies on a pre-trained Vision Transformer:

- `linear_probe`
- `full_ft`
- `lora`
- `pissa_lora`

The original notebooks are still in the repo for checkpoint history, but the recommended workflow is the script-based pipeline under `src/`, `scripts/`, and `configs/`.

## Why This Setup Is Better

- Multiple seeds instead of single-run anecdotes
- Clean experiment plans in JSON
- Structured outputs per run (`config.json`, `history.json`, `result.json`)
- Automatic CSV summaries and plots
- Cluster-ready batch scripts for Purdue RCAC systems

## Repo Layout

- `configs/`: experiment plans
- `scripts/run_experiments.py`: launch one full sweep
- `scripts/summarize_results.py`: aggregate metrics and plots
- `scripts/prefetch_assets.py`: cache the ViT checkpoint and CIFAR-10 ahead of time
- `scripts/setup_env.sh`: create a virtualenv and install dependencies
- `scripts/slurm/`: Gautschi and Gilbreth batch scripts
- `src/ece570_vit_adapters/`: training and reporting code
- `paper/report_draft.md`: report scaffold aligned to the rubric

## Local Quick Start

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/run_experiments.py --config configs/smoke.json --dry-run
python scripts/run_experiments.py --config configs/smoke.json --device cuda
python scripts/summarize_results.py --results-root outputs/smoke
```

If you want to warm the model and dataset caches first:

```bash
source .venv/bin/activate
python scripts/prefetch_assets.py --cache-dir "$HOME/.cache/huggingface"
```

## Recommended Paper Experiments

Start with the core sweep:

```bash
source .venv/bin/activate
python scripts/run_experiments.py --config configs/paper_core.json --device cuda --continue-on-error
```

That core config gives you three strong paper sections:

1. `quick_ablation`: linear probe vs full FT vs LoRA vs PiSSA-LoRA on `5k` images across `3` seeds
2. `data_regime`: LoRA vs PiSSA-LoRA on `1k`, `5k`, `10k`, and `50k` images across `3` seeds
3. `rank_ablation`: LoRA vs PiSSA-LoRA at ranks `4`, `8`, `16`, and `32` across `3` seeds

If time remains, run the extended plan:

```bash
source .venv/bin/activate
python scripts/run_experiments.py --config configs/paper_extended.json --device cuda --continue-on-error
```

This adds:

1. `target_module_ablation`: `qv` vs `qkv`
2. `long_run`: stronger `50k`, `5`-epoch adapter runs

## Purdue Cluster Usage

### Gautschi

1. Log in to `gautschi.rcac.purdue.edu`
2. Clone or copy this repo to your workspace
3. Create the environment once:

```bash
bash scripts/setup_env.sh "$HOME/.venvs/ece570-vit"
source "$HOME/.venvs/ece570-vit/bin/activate"
python scripts/prefetch_assets.py --cache-dir "$SCRATCH/hf-cache"
```

4. Submit:

```bash
sbatch -A <your_account> \
  --export=ALL,ROOT_DIR=$PWD,CONFIG=$PWD/configs/paper_core.json,VENV_DIR=$HOME/.venvs/ece570-vit \
  scripts/slurm/gautschi_ai.sbatch
```

Use `-q preemptible` if you want the cheaper preemptible queue and can tolerate restarts.

### Gilbreth

1. Log in to `gilbreth.rcac.purdue.edu`
2. Create the same virtualenv once
3. Check valid account/partition names:

```bash
slist
sinfo -o "%P %G"
```

4. Submit:

```bash
sbatch -A <your_account> -p <gilbreth_a100_partition> \
  --export=ALL,ROOT_DIR=$PWD,CONFIG=$PWD/configs/paper_core.json,VENV_DIR=$HOME/.venvs/ece570-vit \
  scripts/slurm/gilbreth_gpu.sbatch
```

## Outputs

Each run writes to:

```text
outputs/<plan>/<suite>/<experiment_name>/
```

Typical artifacts:

- `config.json`
- `history.json`
- `result.json`
- `confusion_matrix.png`

After aggregation, look in:

```text
outputs/<plan>/_summary/
```

for:

- `raw_results.csv`
- `summary_by_setting.csv`
- `summary.md`
- `plots/*.png`

## Suggested Paper Story

- `linear_probe` and `full_ft` establish cheap and expensive reference points
- `LoRA` is the standard PEFT baseline
- `PiSSA-LoRA` tests whether improved initialization gives better accuracy or faster convergence
- The multi-seed runs support stronger claims than a single Colab run
- The data-regime and rank sweeps let you discuss when PiSSA helps most

## Reproducibility Notes

- Keep the same seeds from the provided configs for the paper tables
- Report means and standard deviations across seeds, not only the best run
- Use the generated plots as draft figures, but double-check labels before final submission
- If you change hyperparameters, save a new config file instead of editing results by hand

