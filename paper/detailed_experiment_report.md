# Detailed Experiment Report

## Status

All scheduled Gautschi experiments completed successfully.

- Total completed experiments: `60`
- Scheduler jobs:
  - `9194008` (`quick_ablation`) completed in `00:04:26`
  - `9194009` (`rank_ablation`) completed in `00:10:57`
  - `9194010` (`data_regime`) completed in `00:16:20`
- Exit status: all three jobs finished with `ExitCode=0:0`

## Local Artifact Locations

- Full downloaded experiment outputs: `outputs/gautschi_full/`
- Quick ablation summary: `outputs/gautschi_full/quick_ablation_20260409_171202/_summary/`
- Data-regime summary: `outputs/gautschi_full/data_regime_20260409_171308/_summary/`
- Rank-ablation summary: `outputs/gautschi_full/rank_ablation_20260409_171308/_summary/`
- Gautschi Slurm logs: `outputs/gautschi_full/logs/`

## Experimental Setup

### Hardware and Runtime

- Cluster: Purdue RCAC Gautschi
- Partition: `ai`
- GPU type: `H100`
- CPU allocation: `14` CPUs per GPU

### Model and Dataset

- Backbone: `google/vit-base-patch16-224-in21k`
- Dataset: `CIFAR-10`
- Seeds: `7`, `42`, `123`

### Methods

- `Linear Probe`: freeze the ViT encoder and train only the classifier head
- `Full FT`: fine-tune all parameters
- `LoRA`: low-rank adapters on attention `query` and `value`
- `PiSSA-LoRA`: same adapter structure as LoRA, but initialized with PiSSA

### Main Configurations

- Quick ablation: `5k` train / `2k` validation, `3` epochs
- Data regime: `1k`, `5k`, `10k`, `50k` train / `5k` validation, `3` epochs
- Rank ablation: `10k` train / `5k` validation, `3` epochs, ranks `4`, `8`, `16`, `32`

## Executive Summary

The clearest result is that PiSSA-LoRA won every suite. In the quick ablation it achieved the highest mean best validation accuracy, beating both standard LoRA and full fine-tuning while updating only `0.351%` of the model parameters. In the data-regime study, PiSSA-LoRA outperformed standard LoRA at every training-set size, with its largest gain appearing in the lowest-data `1k` setting. In the rank ablation, PiSSA-LoRA again beat standard LoRA at every tested rank, and its best overall mean came at rank `16`.

The strongest single mean result in the whole study was:

- `PiSSA-LoRA`, `50k` train, rank `8`: `98.60% +- 0.09%`

The best single run was:

- `PiSSA-LoRA`, `50k` train, rank `8`, seed `7`: `98.66%`

## Quick Ablation Results

This suite compares the four major tuning strategies on the same `5k`-example budget.

| Method | Trainable Params | Trainable % | Mean Best Val Acc | Std | Mean Train Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probe | 7,690 | 0.009% | 94.58% | 0.31 | 7.87 |
| Full FT | 85,806,346 | 100.000% | 97.12% | 0.33 | 19.38 |
| LoRA | 302,602 | 0.351% | 97.22% | 0.26 | 20.94 |
| PiSSA-LoRA | 302,602 | 0.351% | 97.53% | 0.33 | 21.15 |

### Interpretation

PiSSA-LoRA achieved the best average accuracy in this suite:

- PiSSA-LoRA vs LoRA: `+0.32` percentage points
- PiSSA-LoRA vs Full FT: `+0.42` percentage points
- PiSSA-LoRA vs Linear Probe: `+2.95` percentage points

The main efficiency result is that PiSSA-LoRA matched or exceeded full fine-tuning while training only `302,602` parameters instead of `85,806,346`. That is about `283.6x` fewer trainable parameters than full fine-tuning.

Linear probing was extremely parameter-efficient, using only `7,690` trainable parameters (`0.009%` of the model), but its accuracy lagged the stronger fine-tuning approaches by a noticeable margin.

One subtle but important observation is that parameter efficiency did not automatically translate into lower wall-clock training time. Full fine-tuning was slightly faster than LoRA and PiSSA-LoRA in this small `5k` setup. The likely reason is that all methods still run the full ViT forward pass, while adapters add a small software overhead. This means the main benefit of LoRA/PiSSA here is parameter efficiency and accuracy, not raw speed.

## Data-Regime Results

This suite evaluates whether PiSSA’s advantage changes as the amount of training data grows.

| Train Size | LoRA Mean Best Val Acc | PiSSA-LoRA Mean Best Val Acc | PiSSA Gain |
| --- | ---: | ---: | ---: |
| 1,000 | 83.02% +- 0.89 | 84.55% +- 0.51 | +1.53 |
| 5,000 | 96.99% +- 0.31 | 97.27% +- 0.21 | +0.28 |
| 10,000 | 97.72% +- 0.16 | 97.97% +- 0.31 | +0.25 |
| 50,000 | 98.47% +- 0.13 | 98.60% +- 0.09 | +0.13 |

### Interpretation

PiSSA-LoRA beat standard LoRA at every data scale.

The most important trend is that PiSSA’s advantage is largest in the low-data regime:

- At `1k`, the gain is `+1.53` points
- At `5k`, the gain drops to `+0.28`
- At `10k`, the gain is `+0.25`
- At `50k`, the gain is still positive at `+0.13`

This supports a strong paper claim: PiSSA initialization is most helpful when labeled data is limited, but it still provides a consistent edge even when the model sees the full `50k` training set.

The overall strongest mean setting in the entire study came from this suite:

- `PiSSA-LoRA`, `50k` train: `98.60% +- 0.09%`

The best single raw run also came from this regime:

- `PiSSA-LoRA`, `50k`, seed `7`: `98.66%`

Another useful practical point is that PiSSA-LoRA was slightly faster than LoRA at the largest scale in this experiment:

- LoRA at `50k`: `101.72s` mean training time
- PiSSA-LoRA at `50k`: `97.77s` mean training time

That difference is not the main scientific contribution, but it strengthens the overall efficiency story.

## Rank-Ablation Results

This suite studies whether the PiSSA-vs-LoRA gap changes as the adapter rank increases.

| Rank | Trainable Params | Trainable % | LoRA Mean Best Val Acc | PiSSA-LoRA Mean Best Val Acc | PiSSA Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 155,146 | 0.180% | 97.56% +- 0.19 | 97.83% +- 0.25 | +0.27 |
| 8 | 302,602 | 0.351% | 97.72% +- 0.16 | 97.97% +- 0.31 | +0.25 |
| 16 | 597,514 | 0.692% | 97.67% +- 0.11 | 98.15% +- 0.25 | +0.49 |
| 32 | 1,187,338 | 1.365% | 97.68% +- 0.15 | 97.95% +- 0.09 | +0.27 |

### Interpretation

PiSSA-LoRA again outperformed standard LoRA at every rank. The largest improvement appeared at rank `16`, where PiSSA gained `+0.49` percentage points.

This suite suggests three useful conclusions:

1. Increasing rank from `4` to `8` helps both methods.
2. PiSSA-LoRA reaches its best mean result at rank `16`.
3. Increasing rank further to `32` does not improve mean accuracy, so the accuracy/complexity tradeoff appears to peak around rank `16`.

The best setting in this suite was:

- `PiSSA-LoRA`, rank `16`: `98.15% +- 0.25%`

That setting still trains only `597,514` parameters, which is about `143.6x` fewer trainable parameters than full fine-tuning.

## Stability Across Seeds

The three-seed standard deviations were consistently small:

- Quick ablation standard deviations stayed between `0.26` and `0.33`
- Data-regime standard deviations shrank further at larger training sizes
- Rank-ablation standard deviations stayed below about `0.31`

This matters for the paper because it means the observed advantages are not coming from one lucky run. The gap is modest, but it is systematic and repeatable.

## Main Findings

### 1. PiSSA-LoRA was the strongest overall method

It achieved the best mean accuracy in the quick ablation, the best mean accuracy at every data scale, and the best mean accuracy at every tested rank.

### 2. PiSSA helped most when data was scarce

The `1k`-example setting showed the largest absolute gain over standard LoRA (`+1.53` points), which is exactly the kind of regime where initialization quality should matter most.

### 3. Full fine-tuning was not necessary to get top accuracy

In the `5k` quick ablation, PiSSA-LoRA slightly outperformed full fine-tuning while using `283.6x` fewer trainable parameters.

### 4. Rank `16` was the best adapter setting among the tested ranks

PiSSA-LoRA at rank `16` gave the strongest rank-ablation result, suggesting that moderate rank expansion is worthwhile, but pushing rank too far gives diminishing returns.

### 5. Parameter efficiency was the core benefit, not dramatic speedups

LoRA and PiSSA-LoRA reduced trainable parameters massively, but they did not always reduce wall-clock time relative to full fine-tuning in these small-scale H100 runs. The stronger story is accuracy per trainable parameter, not absolute runtime alone.

## Suggested Results-Section Wording

The experiments show that PiSSA-initialized LoRA consistently improves ViT fine-tuning performance on CIFAR-10 relative to standard LoRA. In the `5k`-example quick ablation, PiSSA-LoRA achieved the highest mean best validation accuracy (`97.53% +- 0.33`), outperforming both standard LoRA (`97.22% +- 0.26`) and full fine-tuning (`97.12% +- 0.33`) while updating only `0.351%` of the model parameters. This indicates that better adapter initialization can offset the usual accuracy gap between parameter-efficient tuning and full-model optimization.

The data-regime sweep further shows that PiSSA’s benefit is strongest when labeled data is limited. At `1k` training examples, PiSSA-LoRA improved over standard LoRA by `1.53` percentage points. The gain decreased as more data became available, but remained positive across all tested scales, including the full `50k`-example setting. This trend suggests that PiSSA provides a more favorable optimization starting point, especially in low-data conditions where adapter training is otherwise more fragile.

The rank-ablation results reinforce this conclusion. PiSSA-LoRA beat standard LoRA at every tested rank, and the best mean result occurred at rank `16` (`98.15% +- 0.25`). Increasing rank beyond `16` did not improve mean accuracy, indicating diminishing returns. Taken together, these results show that PiSSA-LoRA offers a strong accuracy/efficiency tradeoff and is a practical improvement over standard LoRA for vision fine-tuning.

## Recommended Tables and Figures For The Paper

Use these local files directly:

- Quick ablation figure: `outputs/gautschi_full/quick_ablation_20260409_171202/_summary/plots/quick_ablation.png`
- Data-regime figure: `outputs/gautschi_full/data_regime_20260409_171308/_summary/plots/data_regime.png`
- Rank-ablation figure: `outputs/gautschi_full/rank_ablation_20260409_171308/_summary/plots/rank_ablation.png`
- Quick raw table source: `outputs/gautschi_full/quick_ablation_20260409_171202/_summary/summary_by_setting.csv`
- Data raw table source: `outputs/gautschi_full/data_regime_20260409_171308/_summary/summary_by_setting.csv`
- Rank raw table source: `outputs/gautschi_full/rank_ablation_20260409_171308/_summary/summary_by_setting.csv`

## Caveats

- Full fine-tuning was only run in the quick ablation, not in the larger sweeps.
- Runtime numbers reflect measured training time inside the experiment runner, not queue wait time.
- The task is CIFAR-10, which is a useful class-project benchmark but still much smaller than many modern vision fine-tuning workloads.
- The conclusions are strongest for this specific ViT backbone and this adapter placement (`query` and `value`).

## Bottom Line

If the paper needs one sentence to summarize the whole study, use this:

PiSSA-LoRA consistently outperformed standard LoRA across all evaluated settings and slightly surpassed full fine-tuning in the `5k` quick ablation, while using roughly `284x` fewer trainable parameters than full-model training.

