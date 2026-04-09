#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch the ViT checkpoint and CIFAR-10 dataset.")
    parser.add_argument("--model-name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--dataset-name", default="cifar10")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")

    print(f"prefetching model: {args.model_name}")
    AutoImageProcessor.from_pretrained(args.model_name)
    AutoModelForImageClassification.from_pretrained(args.model_name, ignore_mismatched_sizes=True)

    print(f"prefetching dataset: {args.dataset_name}")
    load_dataset(args.dataset_name)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

