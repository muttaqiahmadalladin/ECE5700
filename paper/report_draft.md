# Draft Report Scaffold

Use this as a working draft, then replace bracketed placeholders with your final numbers from `outputs/*/_summary/summary_by_setting.csv`.

## Title

PiSSA-Initialized LoRA for Vision Transformer Fine-Tuning on CIFAR-10

## Abstract

This project studies parameter-efficient fine-tuning for image classification by comparing a pre-trained Vision Transformer under four tuning strategies: linear probing, full fine-tuning, standard LoRA, and PiSSA-initialized LoRA. We evaluate the methods on CIFAR-10 under multiple data regimes and multi-seed settings to measure both effectiveness and parameter efficiency. Our results show that [best overall method] achieves [best mean accuracy] while training only [trainable percent] of the full model parameters, and that PiSSA [helps / does not help] most clearly in the [data regime or rank setting] regime. These findings suggest that [main takeaway].

## 1. Introduction

### Problem and Motivation

- Full fine-tuning of Vision Transformers is strong but expensive.
- Adapter-based methods such as LoRA reduce trainable parameters dramatically.
- PiSSA proposes a stronger adapter initialization, but its value for vision classification needs clearer empirical evidence in small-to-medium training regimes.

### Objectives

- Compare standard LoRA and PiSSA-LoRA on the same ViT backbone.
- Measure accuracy, trainable parameter count, runtime, and memory footprint.
- Test whether the advantage changes with training data size and adapter rank.

## 2. Related Work

You should cite and compare at least:

- The original Vision Transformer paper
- The original LoRA paper
- The PiSSA paper
- A Hugging Face PEFT image-classification example if you discuss implementation conventions

Questions to answer here:

- What does LoRA change relative to full fine-tuning?
- What does PiSSA change relative to standard LoRA?
- Why is ViT + CIFAR-10 a reasonable benchmark for this class project?

## 3. Methodology

### Backbone and Dataset

- Backbone: `google/vit-base-patch16-224-in21k`
- Dataset: `cifar10`
- Input preprocessing: random resized crop + horizontal flip for training, resize + center crop for evaluation

### Methods Compared

1. `linear_probe`: freeze the ViT encoder and train the classifier only
2. `full_ft`: fine-tune all model parameters
3. `lora`: inject LoRA adapters into attention projections
4. `pissa_lora`: same adapter structure, but PiSSA initialization

### Experimental Plans

1. `quick_ablation`
   Compare `linear_probe`, `full_ft`, `lora`, and `pissa_lora` on `5k` training examples and `2k` validation examples across `3` seeds.
2. `data_regime`
   Compare `lora` and `pissa_lora` on `1k`, `5k`, `10k`, and `50k` training examples across `3` seeds.
3. `rank_ablation`
   Compare `lora` and `pissa_lora` with ranks `4`, `8`, `16`, and `32` across `3` seeds.
4. `target_module_ablation` (optional)
   Compare `qv` and `qkv` target modules.
5. `long_run` (optional)
   Extend the strongest adapter settings to `5` epochs on the full `50k` CIFAR-10 training set.

### Metrics

- Best validation accuracy
- Final validation accuracy
- Validation loss
- Trainable parameter count and percentage
- Training time
- Throughput
- Peak GPU memory

## 4. Experimental Results and Analysis

### Table Plan

Use one table each for:

1. Quick ablation mean and standard deviation across seeds
2. Data-regime results by train size
3. Rank ablation results by rank

### Figure Plan

Use generated plots from `outputs/<plan>/_summary/plots/`:

1. `quick_ablation.png`
2. `data_regime.png`
3. `rank_ablation.png`
4. `parameter_efficiency.png`

### Result Discussion Template

- In the quick ablation, [method] reached the best mean accuracy of [x], while full fine-tuning required [y] times more trainable parameters.
- In the data-regime sweep, PiSSA-LoRA was most beneficial when the training set size was [x], suggesting that [interpretation].
- In the rank sweep, increasing rank from [x] to [y] led to [trend], indicating that [tradeoff].
- Runtime and memory results show that [adapter method] offers the best efficiency/accuracy tradeoff.

### Substantive Evaluation Paragraph

This project goes beyond a basic reimplementation by adding multi-seed evaluation, data-regime sweeps, rank ablations, and runtime/parameter-efficiency analysis. These additions help separate one-off lucky runs from stable trends and show more clearly where PiSSA initialization changes behavior relative to standard LoRA.

## 5. Conclusion and Contributions

This project reimplemented and extended adapter-based Vision Transformer fine-tuning for image classification. The main contribution is a reproducible comparison of standard LoRA and PiSSA-LoRA under multiple practical constraints: limited data, varied adapter rank, and repeated seeds. Empirically, we find that [main finding]. Practically, we also provide a reproducible experiment pipeline, automated summaries, and cluster-ready scripts for larger-scale evaluation.

## Formatting Checklist

- Use the ICLR 2026 template
- Stay within the page limit
- Verify every citation in the bibliography
- Make sure figures are legible in two-column format

## LLM Acknowledgement

Suggested wording:

This project used OpenAI Codex as a programming assistant to help restructure the codebase, automate experiments, and draft report language. All experiments were reviewed, selected, and interpreted by the authors.

