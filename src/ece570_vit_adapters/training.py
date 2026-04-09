from __future__ import annotations

import gc
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .plan import ExperimentConfig


def run_single_experiment(
    config: ExperimentConfig,
    output_root: str | Path,
    device_preference: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = Path(output_root) / config.suite / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"

    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text())
        existing["status"] = "skipped_existing"
        return existing

    write_json(run_dir / "config.json", config.to_json_ready())
    set_seed(config.seed)

    device = resolve_device(device_preference)
    amp_dtype = resolve_amp_dtype(device)

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    train_loader, val_loader, label_names = make_dataloaders(processor, config)
    model, init_used = build_model(config, len(label_names), label_names)
    model = model.to(device)

    params = parameter_summary(model)
    eval_before = evaluate(model, val_loader, device, amp_dtype, len(label_names), config.max_eval_batches)
    history, train_stats = train_model(model, train_loader, val_loader, config, device, amp_dtype, len(label_names))
    eval_after = evaluate(model, val_loader, device, amp_dtype, len(label_names), config.max_eval_batches)

    best_epoch = max(history, key=lambda row: row["val_acc"]) if history else None
    confusion_plot_path = None
    if config.save_confusion_matrix and eval_after["confusion_matrix"] is not None:
        confusion_plot_path = run_dir / "confusion_matrix.png"
        save_confusion_matrix_plot(
            np.asarray(eval_after["confusion_matrix"]),
            label_names,
            confusion_plot_path,
            title=f"{config.method_label} | {config.suite}",
        )

    history_path = run_dir / "history.json"
    write_json(history_path, history)

    result = {
        **config.to_json_ready(),
        **params,
        "device": device,
        "amp_dtype": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        "init_used": init_used,
        "val_loss_before": eval_before["loss"],
        "val_acc_before": eval_before["acc"],
        "val_loss_after": eval_after["loss"],
        "val_acc_after": eval_after["acc"],
        "train_loss_last": history[-1]["train_loss"] if history else None,
        "best_epoch": None if best_epoch is None else best_epoch["epoch"],
        "best_val_acc": None if best_epoch is None else best_epoch["val_acc"],
        "best_val_loss": None if best_epoch is None else best_epoch["val_loss"],
        "train_seconds": train_stats["train_seconds"],
        "examples_seen": train_stats["examples_seen"],
        "optimizer_steps": train_stats["optimizer_steps"],
        "throughput_examples_per_second": train_stats["throughput_examples_per_second"],
        "peak_memory_mb": train_stats["peak_memory_mb"],
        "history_path": str(history_path),
        "confusion_matrix_path": None if confusion_plot_path is None else str(confusion_plot_path),
        "status": "completed",
    }

    write_json(result_path, result)

    del model
    del train_loader
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_preference: str) -> str:
    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_preference


def resolve_amp_dtype(device: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: str, amp_dtype: torch.dtype | None):
    if device != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def make_dataloaders(processor, config: ExperimentConfig):
    if config.dataset_name != "cifar10":
        raise ValueError("This project scaffold currently supports dataset_name='cifar10' only.")

    dataset = load_dataset(config.dataset_name)
    label_names = list(dataset["train"].features["label"].names)

    train_size = min(config.train_size, len(dataset["train"]))
    val_size = min(config.val_size, len(dataset["test"]))

    train_ds = dataset["train"].shuffle(seed=config.seed).select(range(train_size))
    val_ds = dataset["test"].shuffle(seed=config.seed).select(range(val_size))

    train_transform, eval_transform = build_transforms(processor)

    def preprocess_train(example):
        images = example["img"]
        if isinstance(images, list):
            example["pixel_values"] = [train_transform(image.convert("RGB")) for image in images]
        else:
            example["pixel_values"] = train_transform(images.convert("RGB"))
        return example

    def preprocess_eval(example):
        images = example["img"]
        if isinstance(images, list):
            example["pixel_values"] = [eval_transform(image.convert("RGB")) for image in images]
        else:
            example["pixel_values"] = eval_transform(images.convert("RGB"))
        return example

    train_ds = train_ds.with_transform(preprocess_train)
    val_ds = val_ds.with_transform(preprocess_eval)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, label_names


def build_transforms(processor):
    image_size = processor.size
    if hasattr(image_size, "get"):
        image_size = image_size.get("height") or image_size.get("shortest_edge")
    elif hasattr(image_size, "height"):
        image_size = image_size.height
    if not isinstance(image_size, int):
        raise ValueError(f"Could not infer image size from processor.size={processor.size!r}")

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    train_transform = Compose(
        [
            RandomResizedCrop(image_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    eval_transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def collate_batch(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def build_model(config: ExperimentConfig, num_labels: int, label_names: list[str]):
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for idx, label in enumerate(label_names)}

    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    if config.method == "full_ft":
        return model, "full_ft"

    if config.method == "linear_probe":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier")
        return model, "linear_probe"

    if config.method == "lora":
        init_candidates = [True]
    else:
        init_candidates = ["pissa_niter_4", "pissa"]

    last_error = None
    for init in init_candidates:
        try:
            lora_config = LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                bias="none",
                target_modules=list(config.target_modules),
                modules_to_save=["classifier"],
                init_lora_weights=init,
            )
            peft_model = get_peft_model(model, lora_config)
            return peft_model, str(init)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to build method={config.method} with target_modules={config.target_modules}: {last_error}"
    )


def parameter_summary(model) -> dict[str, float]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
    }


def evaluate(
    model,
    dataloader,
    device: str,
    amp_dtype: torch.dtype | None,
    num_labels: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=device == "cuda")
            labels = batch["labels"].to(device, non_blocking=device == "cuda")

            with autocast_context(device, amp_dtype):
                outputs = model(pixel_values=pixel_values, labels=labels)

            losses.append(float(outputs.loss.detach().cpu()))
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            label_cpu = labels.detach().cpu().numpy()
            pred_cpu = predictions.detach().cpu().numpy()
            for gold, pred in zip(label_cpu, pred_cpu):
                confusion[gold, pred] += 1

            if max_batches is not None and step >= max_batches:
                break

    per_class_denominator = confusion.sum(axis=1).clip(min=1)
    per_class_acc = (np.diag(confusion) / per_class_denominator).tolist()

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": correct / total if total else 0.0,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    config: ExperimentConfig,
    device: str,
    amp_dtype: torch.dtype | None,
    num_labels: int,
):
    lr = learning_rate_for_method(config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scaler_enabled = device == "cuda" and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device, enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    examples_seen = 0
    optimizer_steps = 0
    training_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        last_batch_index = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            last_batch_index = batch_index
            pixel_values = batch["pixel_values"].to(device, non_blocking=device == "cuda")
            labels = batch["labels"].to(device, non_blocking=device == "cuda")
            examples_seen += labels.size(0)

            with autocast_context(device, amp_dtype):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / config.grad_accum_steps

            epoch_losses.append(float((loss.detach() * config.grad_accum_steps).cpu()))
            scaler.scale(loss).backward()

            should_step = batch_index % config.grad_accum_steps == 0
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            if batch_index % config.log_every == 0:
                window = epoch_losses[-config.log_every :]
                print(
                    f"epoch {epoch:02d} | batch {batch_index:04d} | avg train loss {np.mean(window):.4f}",
                    flush=True,
                )

            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break

        if last_batch_index and last_batch_index % config.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        eval_metrics = evaluate(model, val_loader, device, amp_dtype, num_labels, config.max_eval_batches)
        epoch_record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": eval_metrics["loss"],
            "val_acc": eval_metrics["acc"],
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:02d} complete | train loss {train_loss:.4f} | val acc {eval_metrics['acc']:.4f}",
            flush=True,
        )

    train_seconds = time.perf_counter() - training_start
    peak_memory_mb = None
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    stats = {
        "train_seconds": train_seconds,
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
        "throughput_examples_per_second": examples_seen / train_seconds if train_seconds else None,
        "peak_memory_mb": peak_memory_mb,
    }
    return history, stats


def learning_rate_for_method(config: ExperimentConfig) -> float:
    if config.method == "full_ft":
        return config.lr_full_ft
    if config.method == "linear_probe":
        return config.lr_head_only
    return config.lr_lora


def save_confusion_matrix_plot(confusion: np.ndarray, label_names: list[str], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(label_names)))
    axis.set_xticklabels(label_names, rotation=45, ha="right")
    axis.set_yticks(range(len(label_names)))
    axis.set_yticklabels(label_names)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
