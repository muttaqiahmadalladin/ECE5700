#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ece570_vit_adapters.plan import load_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible ViT adapter experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON run plan.")
    parser.add_argument("--output-root", help="Override the output directory from the JSON config.")
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Optional suite filter. Pass multiple times to keep more than one suite.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--max-experiments", type=int, help="Optional cap for quick testing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded plan and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run experiments even if result.json exists.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one experiment fails.",
    )
    parser.add_argument("--skip-summary", action="store_true", help="Skip CSV/plot aggregation at the end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.config)

    experiments = list(plan.experiments)
    if args.suite:
        keep = set(args.suite)
        experiments = [experiment for experiment in experiments if experiment.suite in keep]

    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]

    output_root = Path(args.output_root or plan.output_root)

    if not experiments:
        print("No experiments selected after filtering.")
        return 1

    print(f"plan: {plan.plan_name}")
    print(f"output_root: {output_root}")
    print(f"experiments: {len(experiments)}")
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"  [{index:02d}] suite={experiment.suite} method={experiment.method} seed={experiment.seed} "
            f"train={experiment.train_size} val={experiment.val_size} epochs={experiment.epochs}"
        )

    if args.dry_run:
        return 0

    from ece570_vit_adapters.training import run_single_experiment

    failures = []
    completed = 0

    for index, experiment in enumerate(experiments, start=1):
        print(f"\n[{index}/{len(experiments)}] {experiment.experiment_name}", flush=True)
        try:
            result = run_single_experiment(
                experiment,
                output_root=output_root,
                device_preference=args.device,
                overwrite=args.overwrite,
            )
            completed += 1
            print(
                f"completed status={result['status']} best_val_acc={result.get('best_val_acc')} "
                f"path={output_root / experiment.suite / experiment.experiment_name}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - failure path is for runtime debugging
            failures.append(
                {
                    "experiment": experiment.to_json_ready(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED: {experiment.experiment_name}: {exc}", flush=True)
            if not args.continue_on_error:
                break

    if failures:
        failure_path = output_root / "_failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"\nWrote failure log to {failure_path}", flush=True)

    if not args.skip_summary and completed > 0:
        try:
            from ece570_vit_adapters.reporting import summarize_results

            summary_files = summarize_results(output_root)
            print("\nSummary artifacts:")
            for label, path in summary_files.items():
                print(f"  {label}: {path}")
        except Exception as exc:  # pragma: no cover - runtime fallback
            print(f"\nWarning: runs completed, but summary generation failed: {exc}", flush=True)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
