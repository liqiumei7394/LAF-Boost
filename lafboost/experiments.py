from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data import DataBundle, WindowDataset, compute_scaler, load_records, make_bundle, make_id_map, split_boundaries
from .training import train_hgbr, train_lafboost, train_torch_model


ABLATION_SETTINGS = {
    "full": {},
    "without_env": {"use_env": False},
    "without_related_load": {"use_related": False},
    "without_gate": {"use_gate": False},
    "without_decomposition": {"use_decomp": False},
}


def save_results(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    front = [c for c in ["experiment", "model", "feature_mode", "ablation", "train_floors", "test_floors"] if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(path, index=False)
    print(f"saved {path}")
    print(df.to_string(index=False))


def run_main(args: argparse.Namespace) -> List[Dict]:
    bundle = make_bundle(args.data_dir, args.years, args.floors, "full", args.input_len, args.horizons, args.stride)
    print(f"records={len(bundle.train.records)} train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)} features={bundle.feature_names}")
    rows = []
    for model_name in args.models:
        t0 = time.time()
        if model_name == "hgbr":
            row = train_hgbr(bundle, args)
        elif model_name == "lafboost":
            row = train_lafboost(bundle, args)
        else:
            row = train_torch_model(model_name, bundle, args)
        row.update({"experiment": "main", "seconds": round(time.time() - t0, 2)})
        rows.append(row)
    save_results(rows, args.output_dir / "main_results.csv")
    return rows


def run_features(args: argparse.Namespace) -> List[Dict]:
    rows = []
    for mode in ["history", "history_time", "history_env", "full"]:
        bundle = make_bundle(args.data_dir, args.years, args.floors, mode, args.input_len, args.horizons, args.stride)
        t0 = time.time()
        row = train_lafboost(bundle, args)
        row.update({"experiment": "features", "feature_mode": mode, "seconds": round(time.time() - t0, 2)})
        rows.append(row)
    save_results(rows, args.output_dir / "feature_results.csv")
    return rows


def run_ablation(args: argparse.Namespace) -> List[Dict]:
    """Run the LAF-Net core ablation used in the manuscript.

    The manuscript ablation intentionally evaluates the core LAF-Net network
    without the stacked residual calibrator, so the contribution of each
    branch/module is not hidden by the validation-set calibrator.
    """
    bundle = make_bundle(args.data_dir, args.years, args.floors, "full", args.input_len, args.horizons, args.stride)
    rows = []
    for name, kwargs in ABLATION_SETTINGS.items():
        t0 = time.time()
        row = train_torch_model("lafnet", bundle, args, kwargs)
        row.update({"experiment": "ablation_core", "ablation": name, "seconds": round(time.time() - t0, 2)})
        rows.append(row)
    save_results(rows, args.output_dir / "ablation_core_results.csv")
    return rows


def run_lafboost_ablation(args: argparse.Namespace) -> List[Dict]:
    """Optional full LAF-Boost ablation with the calibrator kept enabled."""
    bundle = make_bundle(args.data_dir, args.years, args.floors, "full", args.input_len, args.horizons, args.stride)
    rows = []
    for name, kwargs in ABLATION_SETTINGS.items():
        t0 = time.time()
        row = train_lafboost(bundle, args, kwargs)
        row.update({"experiment": "lafboost_ablation", "ablation": name, "seconds": round(time.time() - t0, 2)})
        rows.append(row)
    save_results(rows, args.output_dir / "ablation_results.csv")
    return rows


def run_transfer(args: argparse.Namespace) -> List[Dict]:
    # Train and validate on floors 2-5; test on floors 6-7.
    train_records = load_records(args.data_dir, args.years, [2, 3, 4, 5], "full")
    test_records = load_records(args.data_dir, args.years, [6, 7], "full")
    train_end, val_end = split_boundaries(args.years[-1])
    mean, std = compute_scaler(train_records, args.input_len, args.horizons, args.stride, train_end)
    floor_to_idx = make_id_map([rec.floor for rec in train_records + test_records])
    zone_to_idx = make_id_map([rec.zone for rec in train_records + test_records])

    train = WindowDataset(train_records, args.input_len, args.horizons, args.stride, "train", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    val = WindowDataset(train_records, args.input_len, args.horizons, args.stride, "val", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    # For cross-floor, use all target-floor windows after validation boundary as test.
    test = WindowDataset(test_records, args.input_len, args.horizons, args.stride, "test", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    bundle = DataBundle(
        train,
        val,
        test,
        train_records[0].feature_names,
        mean,
        std,
        float(mean[0]),
        float(std[0]),
        len(floor_to_idx),
        len(zone_to_idx),
        floor_to_idx,
        zone_to_idx,
    )

    rows = []
    for model_name in ["timesnet", "itransformer", "timemixer", "lafboost"]:
        t0 = time.time()
        if model_name == "lafboost":
            row = train_lafboost(bundle, args)
        else:
            row = train_torch_model(model_name, bundle, args)
        row.update({
            "experiment": "transfer",
            "train_floors": "2-5",
            "test_floors": "6-7",
            "seconds": round(time.time() - t0, 2),
        })
        rows.append(row)
    save_results(rows, args.output_dir / "transfer_results.csv")
    return rows


def run_smoke(args: argparse.Namespace) -> None:
    args.models = ["timesnet", "lafnet"]
    args.floors = [2]
    args.epochs = min(args.epochs, 1)
    args.stride = max(args.stride, 240)
    run_main(args)
