# ECE 5700 Project: ViT Adapter Tuning on CIFAR-10

This repository contains the final course-project code and paper assets for comparing four transfer-learning strategies on a pretrained Vision Transformer:

- `linear_probe`
- `full_ft`
- `lora`
- `pissa_lora`

The original notebooks are preserved for project history, but the recommended workflow is the script-based experiment pipeline under `src/`, `scripts/`, and `configs/`.

## Final Submission Files

- Paper source: `iclr2026-2/project_report.tex`
- Paper PDF: `iclr2026-2/project_report.pdf`
- Detailed result notes: `paper/detailed_experiment_report.md`
- Draft scaffold: `paper/report_draft.md`

## Why This Setup Is Better

- Multiple seeds instead of single-run anecdotes
- Clean experiment plans in JSON
- Structured artifacts per run (`config.json`, `history.json`, `result.json`)
- Automatic CSV summaries and plots
- Cluster-ready batch scripts for Purdue RCAC systems

## Repo Layout

- `configs/`: experiment plans used by the runner
- `scripts/run_experiments.py`: expands a JSON plan and executes all runs
- `scripts/summarize_results.py`: aggregates raw runs into CSV, markdown, and plots
- `scripts/prefetch_assets.py`: caches the ViT checkpoint and CIFAR-10 ahead of time
- `scripts/setup_env.sh`: creates a virtual environment and installs dependencies
- `scripts/slurm/`: Gautschi and Gilbreth batch scripts
- `src/ece570_vit_adapters/`: training, plan loading, and result reporting code
- `paper/`: markdown report drafts and result summaries
- `iclr2026-2/`: ICLR 2026 template assets plus the final report source

## Dependencies

Create the environment with:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

Key Python dependencies are listed in `requirements.txt`, including:

- `torch`
- `torchvision`
- `transformers`
- `datasets`
- `peft`
- `pandas`
- `matplotlib`

## Local Quick Start

```bash
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

The code automatically downloads the Hugging Face ViT checkpoint and the CIFAR-10 dataset when they are missing. If you are running on a shared cluster, `scripts/prefetch_assets.py` is the safest way to populate caches once before launching a sweep.

## Recommended Paper Experiments

Start with the core sweep:

```bash
source .venv/bin/activate
python scripts/run_experiments.py --config configs/paper_core.json --device cuda --continue-on-error
```

That core config gives the three main paper suites:

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

Use `-q preemptible` if you want the cheaper queue and can tolerate restarts.

### Gilbreth

1. Log in to `gilbreth.rcac.purdue.edu`
2. Create the same virtualenv once
3. Check valid account and partition names:

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

## Reproducing The Final Paper

The final report was written from the experiment summaries downloaded from Gautschi. The most important local result notes are in `paper/detailed_experiment_report.md`.

To rebuild the PDF from source:

```bash
cd iclr2026-2
tectonic project_report.tex
```

If you prefer a standard TeX workflow, `pdflatex` plus `bibtex` also works as long as the template support files in `iclr2026-2/` remain in place.

## Suggested Paper Story

- `linear_probe` and `full_ft` establish cheap and expensive reference points
- `LoRA` is the standard PEFT baseline
- `PiSSA-LoRA` tests whether improved initialization gives better accuracy or faster convergence
- The multi-seed runs support stronger claims than a single Colab run
- The data-regime and rank sweeps let you discuss when PiSSA helps most

## Authorship And Provenance

### Written for this submission

The reproducible experiment pipeline and report sources in this repository were written for this course project submission:

- `configs/`
- `scripts/`
- `src/ece570_vit_adapters/`
- `README.md`
- `paper/report_draft.md`
- `paper/detailed_experiment_report.md`
- `iclr2026-2/project_report.tex`
- `iclr2026-2/project_report.bib`

### Adapted or copied external materials

- `iclr2026-2/iclr2026_conference.sty`
- `iclr2026-2/iclr2026_conference.bst`
- `iclr2026-2/natbib.sty`
- `iclr2026-2/fancyhdr.sty`
- `iclr2026-2/math_commands.tex`

These template support files come from the official ICLR 2026 style package distributed by the course instructions.

The pretrained backbone and dataset are downloaded through public libraries at runtime:

- ViT backbone via Hugging Face `transformers`
- CIFAR-10 via Hugging Face `datasets`
- LoRA and PiSSA support via `peft`

The plots copied into `iclr2026-2/figures/` were generated by this repository from the experiment summaries.

### Exact line-number edits to prior repository code

No pre-existing tracked source files were edited for the final reproducible pipeline. The original notebooks:

- `ECE570Project.ipynb`
- `ECE570Project_Checkpoint2_Colab.ipynb`

were retained unchanged. All new functionality for the final submission was added in new files rather than by patching existing notebook cells.

## Reproducibility Notes

- Keep the same seeds from the provided configs for the paper tables
- Report means and standard deviations across seeds, not only the best run
- Use the generated plots as draft figures, but double-check labels before final submission
- If you change hyperparameters, save a new config file instead of editing results by hand
