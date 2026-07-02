"""Command-line interface for LAF-Boost experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import run_ablation, run_features, run_lafboost_ablation, run_main, run_smoke, run_transfer
from .utils import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        choices=["smoke", "main", "features", "ablation", "lafboost_ablation", "transfer", "all"],
        default="smoke",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/11726517"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--years", type=int, nargs="+", default=[2019])
    p.add_argument("--floors", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--input-len", type=int, default=120, help="History length in minutes.")
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60], help="Forecast horizons in minutes.")
    p.add_argument("--stride", type=int, default=30, help="Window stride in minutes.")
    p.add_argument("--models", nargs="+", default=["hgbr", "patchtst", "timesnet", "itransformer", "timemixer", "lafnet", "lafboost"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-sklearn-samples", type=int, default=120000)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"config_{args.experiment}.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    if args.experiment == "smoke":
        run_smoke(args)
    elif args.experiment == "main":
        run_main(args)
    elif args.experiment == "features":
        run_features(args)
    elif args.experiment == "ablation":
        run_ablation(args)
    elif args.experiment == "lafboost_ablation":
        run_lafboost_ablation(args)
    elif args.experiment == "transfer":
        run_transfer(args)
    elif args.experiment == "all":
        run_main(args)
        run_features(args)
        run_ablation(args)
        run_transfer(args)


if __name__ == "__main__":
    main()
