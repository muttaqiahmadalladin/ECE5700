from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_results(results_root: str | Path) -> dict[str, Path]:
    results_root = Path(results_root)
    summary_dir = results_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_df = collect_results(results_root)
    if raw_df.empty:
        raise RuntimeError(f"No result.json files were found under {results_root}.")

    raw_path = summary_dir / "raw_results.csv"
    raw_df.sort_values(["suite", "method", "seed"]).to_csv(raw_path, index=False)

    summary_df = build_summary(raw_df)
    summary_path = summary_dir / "summary_by_setting.csv"
    summary_df.to_csv(summary_path, index=False)

    markdown_path = summary_dir / "summary.md"
    markdown_path.write_text(build_markdown_summary(raw_df, summary_df))

    plot_paths = generate_plots(summary_df, summary_dir / "plots")
    return {
        "raw_results": raw_path,
        "summary_by_setting": summary_path,
        "markdown_summary": markdown_path,
        **plot_paths,
    }


def collect_results(results_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(results_root.rglob("result.json")):
        if path.parent.name == "_summary":
            continue
        records.append(json.loads(path.read_text()))
    return pd.DataFrame(records)


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "suite",
        "method",
        "method_label",
        "train_size",
        "val_size",
        "epochs",
        "rank",
        "alpha",
        "target_modules_tag",
        "effective_batch_size",
    ]
    group_cols = [column for column in group_cols if column in raw_df.columns]

    metric_cols = [
        "val_acc_before",
        "val_acc_after",
        "best_val_acc",
        "val_loss_before",
        "val_loss_after",
        "best_val_loss",
        "train_loss_last",
        "train_seconds",
        "throughput_examples_per_second",
        "peak_memory_mb",
        "trainable_params",
        "trainable_pct",
    ]
    metric_cols = [column for column in metric_cols if column in raw_df.columns]

    summary_df = raw_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max", "count"])
    summary_df = summary_df.reset_index()
    summary_df.columns = _flatten_columns(summary_df.columns)
    return summary_df.sort_values(["suite", "method", "train_size", "rank"]).reset_index(drop=True)


def build_markdown_summary(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    top = raw_df.sort_values("best_val_acc", ascending=False).head(10)
    top_table_df = top[
        [
            "suite",
            "method_label",
            "seed",
            "train_size",
            "epochs",
            "rank",
            "target_modules_tag",
            "best_val_acc",
            "train_seconds",
        ]
    ]

    lines = [
        "# Experiment Summary",
        "",
        f"- Total completed runs: {len(raw_df)}",
        f"- Distinct suites: {raw_df['suite'].nunique()}",
        f"- Distinct methods: {raw_df['method'].nunique()}",
        "",
        "## Top Runs",
        "",
        _markdown_table(top_table_df),
        "",
        "## Files",
        "",
        "- `raw_results.csv`: one row per experiment.",
        "- `summary_by_setting.csv`: grouped mean/std/min/max across seeds.",
        "- `plots/`: suite-level visualizations plus a parameter-efficiency scatter plot.",
        "",
        "## How To Use This In The Paper",
        "",
        "- Quote means and standard deviations from `summary_by_setting.csv` rather than single-seed results.",
        "- Use `plots/` figures directly as draft visuals for the Results section.",
        "- Cross-check the best setting against `raw_results.csv` before finalizing claims.",
    ]
    return "\n".join(lines)


def generate_plots(summary_df: pd.DataFrame, plots_dir: Path) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    for suite_name, suite_df in summary_df.groupby("suite"):
        plot_path = plots_dir / f"{suite_name}.png"
        plot_suite(suite_name, suite_df, plot_path)
        generated[f"plot_{suite_name}"] = plot_path

    pareto_path = plots_dir / "parameter_efficiency.png"
    plot_parameter_efficiency(summary_df, pareto_path)
    generated["parameter_efficiency"] = pareto_path
    return generated


def plot_suite(suite_name: str, summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df["train_size"].nunique() > 1:
        x_col = "train_size"
    elif summary_df["rank"].nunique() > 1:
        x_col = "rank"
    elif summary_df["target_modules_tag"].nunique() > 1:
        x_col = "target_modules_tag"
    else:
        x_col = "method_label"

    fig, axis = plt.subplots(figsize=(8, 5))

    if x_col in {"train_size", "rank"}:
        for method_label, method_df in summary_df.groupby("method_label"):
            method_df = method_df.sort_values(x_col)
            axis.errorbar(
                method_df[x_col],
                method_df["best_val_acc_mean"],
                yerr=method_df["best_val_acc_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=method_label,
            )
        axis.set_xlabel(x_col.replace("_", " ").title())
    else:
        labels = summary_df[x_col].astype(str).tolist()
        if summary_df["method_label"].nunique() > 1 and x_col != "method_label":
            labels = [f"{row['method_label']}\n{label}" for label, (_, row) in zip(labels, summary_df.iterrows())]
        means = summary_df["best_val_acc_mean"].tolist()
        errors = summary_df["best_val_acc_std"].fillna(0.0).tolist()
        axis.bar(labels, means, yerr=errors, capsize=4)
        axis.set_xlabel(x_col.replace("_", " ").title())

    axis.set_ylabel("Best Validation Accuracy")
    axis.set_title(suite_name.replace("_", " ").title())
    axis.grid(axis="y", alpha=0.25)
    if x_col in {"train_size", "rank"}:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_parameter_efficiency(summary_df: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for method_label, method_df in summary_df.groupby("method_label"):
        axis.scatter(
            method_df["trainable_params_mean"],
            method_df["best_val_acc_mean"],
            s=60,
            label=method_label,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Trainable Parameters (mean, log scale)")
    axis.set_ylabel("Best Validation Accuracy (mean)")
    axis.set_title("Parameter Efficiency Frontier")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _flatten_columns(columns) -> list[str]:
    flattened = []
    for column in columns:
        if isinstance(column, tuple):
            flattened.append("_".join(str(part) for part in column if part))
        else:
            flattened.append(str(column))
    return flattened


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in frame.iterrows():
        rows.append([str(row[column]) for column in headers])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)
