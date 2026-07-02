from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import DataBundle
from .metrics import inverse_target, metrics
from .models import MODEL_REGISTRY
from .utils import select_device


def evaluate_torch(model: nn.Module, loader: DataLoader, bundle: DataBundle, device: torch.device, horizons: Sequence[int]) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, floor, zone, seasonal_bases in loader:
            x = x.to(device)
            floor = floor.to(device)
            zone = zone.to(device)
            seasonal_bases = seasonal_bases.to(device)
            pred = model(x, floor, zone, seasonal_bases).detach().cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    y_true = inverse_target(np.concatenate(ys), bundle)
    y_pred = inverse_target(np.concatenate(ps), bundle)
    return metrics(y_true, y_pred, horizons)


def predict_torch_normalized(model: nn.Module, dataset: Dataset, args: argparse.Namespace, device: torch.device):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    ys, ps, bases = [], [], []
    with torch.no_grad():
        for x, y, floor, zone, seasonal_bases in loader:
            x = x.to(device)
            floor = floor.to(device)
            zone = zone.to(device)
            seasonal_bases = seasonal_bases.to(device)
            pred = model(x, floor, zone, seasonal_bases).detach().cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
            bases.append(seasonal_bases.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(bases)


def fit_torch_model(
    model_name: str,
    bundle: DataBundle,
    args: argparse.Namespace,
    model_kwargs: Optional[Dict] = None,
):
    device = select_device(args.device)
    model_kwargs = model_kwargs or {}
    cls = MODEL_REGISTRY[model_name]
    model = cls(
        input_dim=len(bundle.feature_names),
        output_dim=len(args.horizons),
        input_len=args.input_len,
        n_floors=bundle.n_floors,
        n_zones=bundle.n_zones,
        feature_names=bundle.feature_names,
        hidden=args.hidden,
        dropout=args.dropout,
        **model_kwargs,
    ).to(device)

    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0, reduction="none")
    best_state = None
    best_val = float("inf")
    patience = args.patience
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, floor, zone, seasonal_bases in train_loader:
            x, y = x.to(device), y.to(device)
            floor, zone = floor.to(device), zone.to(device)
            seasonal_bases = seasonal_bases.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, floor, zone, seasonal_bases)
            # Emphasize the longest horizon while still learning shorter horizons.
            weights = torch.linspace(0.2, 0.5, pred.shape[-1], device=device)
            weights = weights / weights.mean()
            loss = (loss_fn(pred, y) * weights.view(1, -1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        val_metrics = evaluate_torch(model, val_loader, bundle, device, args.horizons)
        val_score = val_metrics[f"rmse_{args.horizons[-1]}"]
        if args.verbose:
            print(f"{model_name} epoch={epoch:03d} train_loss={total/max(n,1):.5f} val_rmse_{args.horizons[-1]}={val_score:.5f}")
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, best_val


def train_torch_model(
    model_name: str,
    bundle: DataBundle,
    args: argparse.Namespace,
    model_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    model, device, best_val = fit_torch_model(model_name, bundle, args, model_kwargs)
    test_loader = DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    result = evaluate_torch(model, test_loader, bundle, device, args.horizons)
    result.update({"model": model_name, "params": sum(p.numel() for p in model.parameters()), "best_val_rmse": best_val})
    return result


def collect_sklearn_windows(bundle: DataBundle, split: str, max_samples: int, seed: int):
    ds = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[split]
    rng = np.random.default_rng(seed)
    indices = np.arange(len(ds))
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    feats, ys = [], []
    for idx in indices:
        x, y, floor, zone, seasonal_bases = ds[int(idx)]
        arr = x.numpy()
        # Compact tabular features: latest, mean, std, min, max, and floor/zone.
        tab = np.concatenate([
            arr[-1],
            arr.mean(axis=0),
            arr.std(axis=0),
            arr.min(axis=0),
            arr.max(axis=0),
            seasonal_bases.numpy().reshape(-1),
            [floor.item(), zone.item()],
        ])
        feats.append(tab)
        ys.append(y.numpy())
    return np.asarray(feats, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_hgbr(bundle: DataBundle, args: argparse.Namespace) -> Dict[str, float]:
    x_train, y_train = collect_sklearn_windows(bundle, "train", args.max_sklearn_samples, args.seed)
    x_test, y_test = collect_sklearn_windows(bundle, "test", args.max_sklearn_samples, args.seed + 1)
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=args.seed)
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = metrics(inverse_target(y_test, bundle), inverse_target(pred, bundle), args.horizons)
    out.update({"model": "hgbr", "params": 0, "best_val_rmse": float("nan")})
    return out


def fit_hgbr_predictor(bundle: DataBundle, args: argparse.Namespace):
    x_train, y_train = collect_sklearn_windows(bundle, "train", args.max_sklearn_samples, args.seed)
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=args.seed)
    )
    model.fit(x_train, y_train)
    return model


def predict_hgbr_normalized(model, bundle: DataBundle, split: str, args: argparse.Namespace):
    x, y = collect_sklearn_windows(bundle, split, 10**9, args.seed)
    pred = model.predict(x)
    return y, pred


def stack_features(laf_pred: np.ndarray, hgbr_pred: np.ndarray, bases: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            laf_pred,
            hgbr_pred,
            bases.reshape(bases.shape[0], -1),
            laf_pred - hgbr_pred,
            np.abs(laf_pred - hgbr_pred),
        ],
        axis=1,
    )


def train_lafboost(bundle: DataBundle, args: argparse.Namespace, model_kwargs: Optional[Dict] = None) -> Dict[str, float]:
    """Enhanced proposed method: LAF-Net with periodic residual calibration.

    The calibrator is part of the proposed method, not a baseline. It uses the
    validation split as a stacking set and is evaluated only on the held-out
    test split.
    """
    model_kwargs = model_kwargs or {}
    hgbr = fit_hgbr_predictor(bundle, args)
    laf, device, best_val = fit_torch_model("lafnet", bundle, args, model_kwargs)

    y_val, laf_val, bases_val = predict_torch_normalized(laf, bundle.val, args, device)
    y_test, laf_test, bases_test = predict_torch_normalized(laf, bundle.test, args, device)
    y_val_h, hgbr_val = predict_hgbr_normalized(hgbr, bundle, "val", args)
    y_test_h, hgbr_test = predict_hgbr_normalized(hgbr, bundle, "test", args)

    # The orders are produced by the same WindowDataset iteration; this protects
    # against accidental mismatch if the helper implementation changes later.
    if not np.allclose(y_val, y_val_h) or not np.allclose(y_test, y_test_h):
        raise RuntimeError("Stacking inputs are misaligned.")

    calibrator = MultiOutputRegressor(Ridge(alpha=0.1))
    calibrator.fit(stack_features(laf_val, hgbr_val, bases_val), y_val)
    ridge_val = calibrator.predict(stack_features(laf_val, hgbr_val, bases_val))
    ridge_test = calibrator.predict(stack_features(laf_test, hgbr_test, bases_test))

    val_candidates = {
        "laf": laf_val,
        "periodic_residual": hgbr_val,
        "stacked_calibrator": ridge_val,
    }
    test_candidates = {
        "laf": laf_test,
        "periodic_residual": hgbr_test,
        "stacked_calibrator": ridge_test,
    }
    pred = np.zeros_like(y_test)
    selected = []
    for i, h in enumerate(args.horizons):
        scores = {}
        for name, cand in val_candidates.items():
            mae = mean_absolute_error(y_val[:, i], cand[:, i])
            rmse = mean_squared_error(y_val[:, i], cand[:, i]) ** 0.5
            scores[name] = mae + rmse
        best_name = min(scores, key=scores.get)
        pred[:, i] = test_candidates[best_name][:, i]
        selected.append(f"{h}:{best_name}")
    out = metrics(inverse_target(y_test, bundle), inverse_target(pred, bundle), args.horizons)
    out.update({
        "model": "lafboost",
        "params": sum(p.numel() for p in laf.parameters()),
        "best_val_rmse": best_val,
        "selected_heads": "|".join(selected),
    })
    return out
