from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "linear_probe": "Linear Probe",
    "full_ft": "Full FT",
    "lora": "LoRA",
    "pissa_lora": "PiSSA-LoRA",
}

_METHODS = set(METHOD_LABELS)


@dataclass(frozen=True)
class ExperimentConfig:
    suite: str
    method: str
    seed: int
    model_name: str = "google/vit-base-patch16-224-in21k"
    dataset_name: str = "cifar10"
    train_size: int = 5000
    val_size: int = 2000
    epochs: int = 3
    micro_batch_size: int = 32
    eval_batch_size: int = 128
    grad_accum_steps: int = 2
    lr_full_ft: float = 5e-5
    lr_head_only: float = 2e-4
    lr_lora: float = 2e-4
    weight_decay: float = 0.01
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "value")
    num_workers: int = 4
    log_every: int = 50
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    save_confusion_matrix: bool = True

    @property
    def method_label(self) -> str:
        return METHOD_LABELS[self.method]

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def target_modules_tag(self) -> str:
        mapping = {"query": "q", "key": "k", "value": "v"}
        return "".join(mapping.get(module, module[:1]) for module in self.target_modules)

    @property
    def experiment_name(self) -> str:
        tokens = [
            self.suite,
            self.method,
            f"seed{self.seed}",
            f"train{self.train_size}",
            f"val{self.val_size}",
            f"ep{self.epochs}",
        ]
        if self.method in {"lora", "pissa_lora"}:
            tokens.extend([f"r{self.rank}", self.target_modules_tag])
        return "__".join(_sanitize_token(token) for token in tokens)

    def to_json_ready(self) -> dict[str, Any]:
        payload = {
            "suite": self.suite,
            "method": self.method,
            "method_label": self.method_label,
            "seed": self.seed,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "epochs": self.epochs,
            "micro_batch_size": self.micro_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_batch_size": self.effective_batch_size,
            "lr_full_ft": self.lr_full_ft,
            "lr_head_only": self.lr_head_only,
            "lr_lora": self.lr_lora,
            "weight_decay": self.weight_decay,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "target_modules_tag": self.target_modules_tag,
            "num_workers": self.num_workers,
            "log_every": self.log_every,
            "max_train_batches": self.max_train_batches,
            "max_eval_batches": self.max_eval_batches,
            "save_confusion_matrix": self.save_confusion_matrix,
            "experiment_name": self.experiment_name,
        }
        return payload


@dataclass(frozen=True)
class RunPlan:
    plan_name: str
    output_root: str
    experiments: tuple[ExperimentConfig, ...] = field(default_factory=tuple)


def load_plan(path: str | Path) -> RunPlan:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    shared = raw.get("shared", {})
    experiments: list[ExperimentConfig] = []

    for suite_spec in raw.get("suites", []):
        experiments.extend(_expand_suite(shared, suite_spec))

    return RunPlan(
        plan_name=raw.get("plan_name", config_path.stem),
        output_root=raw.get("output_root", f"outputs/{config_path.stem}"),
        experiments=tuple(experiments),
    )


def _expand_suite(shared: dict[str, Any], suite_spec: dict[str, Any]) -> list[ExperimentConfig]:
    suite_name = suite_spec["name"]
    base = dict(shared)
    base.update(suite_spec.get("base", {}))
    base["suite"] = suite_name

    grid = suite_spec.get("grid", {})
    if not grid:
        return [_coerce_experiment(base)]

    keys = list(grid)
    value_lists = [_ensure_list(grid[key]) for key in keys]
    experiments = []

    for values in product(*value_lists):
        candidate = dict(base)
        candidate.update(zip(keys, values))
        experiments.append(_coerce_experiment(candidate))

    return experiments


def _coerce_experiment(candidate: dict[str, Any]) -> ExperimentConfig:
    method = str(candidate["method"])
    if method not in _METHODS:
        raise ValueError(f"Unsupported method '{method}'. Expected one of {sorted(_METHODS)}.")

    target_modules = tuple(candidate.get("target_modules", ("query", "value")))
    return ExperimentConfig(
        suite=str(candidate["suite"]),
        method=method,
        seed=int(candidate.get("seed", 42)),
        model_name=str(candidate.get("model_name", "google/vit-base-patch16-224-in21k")),
        dataset_name=str(candidate.get("dataset_name", "cifar10")),
        train_size=int(candidate.get("train_size", 5000)),
        val_size=int(candidate.get("val_size", 2000)),
        epochs=int(candidate.get("epochs", 3)),
        micro_batch_size=int(candidate.get("micro_batch_size", 32)),
        eval_batch_size=int(candidate.get("eval_batch_size", 128)),
        grad_accum_steps=int(candidate.get("grad_accum_steps", 2)),
        lr_full_ft=float(candidate.get("lr_full_ft", 5e-5)),
        lr_head_only=float(candidate.get("lr_head_only", 2e-4)),
        lr_lora=float(candidate.get("lr_lora", 2e-4)),
        weight_decay=float(candidate.get("weight_decay", 0.01)),
        rank=int(candidate.get("rank", 8)),
        alpha=int(candidate.get("alpha", 16)),
        dropout=float(candidate.get("dropout", 0.05)),
        target_modules=target_modules,
        num_workers=int(candidate.get("num_workers", 4)),
        log_every=int(candidate.get("log_every", 50)),
        max_train_batches=_optional_int(candidate.get("max_train_batches")),
        max_eval_batches=_optional_int(candidate.get("max_eval_batches")),
        save_confusion_matrix=bool(candidate.get("save_confusion_matrix", True)),
    )


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _sanitize_token(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-").lower()

