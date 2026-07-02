from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FEATURE_SETS


@dataclass
class SeriesRecord:
    floor: int
    zone: int
    dates: np.ndarray
    values: np.ndarray
    feature_names: List[str]


@dataclass
class DataBundle:
    train: "WindowDataset"
    val: "WindowDataset"
    test: "WindowDataset"
    feature_names: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    n_floors: int
    n_zones: int
    floor_to_idx: Dict[int, int]
    zone_to_idx: Dict[int, int]


def zone_columns(cols: Sequence[str]) -> List[int]:
    zones = set()
    for col in cols:
        if col.startswith("z") and "_" in col:
            prefix = col.split("_", 1)[0]
            if prefix[1:].isdigit():
                zones.add(int(prefix[1:]))
    return sorted(zones)


def read_floor_file(path: Path, floor: int, feature_mode: str) -> List[SeriesRecord]:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    records: List[SeriesRecord] = []
    wanted = FEATURE_SETS[feature_mode]

    for zone in zone_columns(df.columns):
        p = f"z{zone}"
        light = f"{p}_Light(kW)"
        plug = f"{p}_Plug(kW)"
        temp = f"{p}_S1(degC)"
        rh = f"{p}_S1(RH%)"
        lux = f"{p}_S1(lux)"
        ac_cols = [c for c in df.columns if c.startswith(f"{p}_AC") and c.endswith("(kW)")]

        required = [light]
        if "plug" in wanted:
            required.append(plug)
        if "temp" in wanted:
            required.append(temp)
        if "rh" in wanted:
            required.append(rh)
        if "lux" in wanted:
            required.append(lux)
        if "ac" in wanted and not ac_cols:
            continue
        if any(c not in df.columns for c in required):
            continue

        out = pd.DataFrame({"Date": df["Date"]})
        out["light"] = pd.to_numeric(df[light], errors="coerce")
        if "plug" in wanted:
            out["plug"] = pd.to_numeric(df[plug], errors="coerce")
        if "ac" in wanted:
            out["ac"] = df[ac_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if "lux" in wanted:
            out["lux"] = pd.to_numeric(df[lux], errors="coerce")
        if "temp" in wanted:
            out["temp"] = pd.to_numeric(df[temp], errors="coerce")
        if "rh" in wanted:
            out["rh"] = pd.to_numeric(df[rh], errors="coerce")

        if any(k in wanted for k in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "worktime"]):
            dt = out["Date"]
            minute_of_day = dt.dt.hour * 60 + dt.dt.minute
            out["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
            out["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
            out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
            out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
            out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
            out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
            out["worktime"] = ((dt.dt.dayofweek < 5) & (dt.dt.hour >= 8) & (dt.dt.hour < 18)).astype(float)

        feature_names = [c for c in wanted if c in out.columns]
        # Interpolate short gaps and remove unusable long gaps.
        for c in feature_names:
            out[c] = out[c].interpolate(limit=30).ffill().bfill()
        values = out[feature_names].to_numpy(np.float32)
        valid_ratio = np.isfinite(values).all(axis=1).mean()
        if valid_ratio < 0.95:
            continue
        records.append(
            SeriesRecord(
                floor=floor,
                zone=zone,
                dates=out["Date"].to_numpy(),
                values=values,
                feature_names=feature_names,
            )
        )
    return records


def load_records(data_dir: Path, years: Sequence[int], floors: Sequence[int], feature_mode: str) -> List[SeriesRecord]:
    records: List[SeriesRecord] = []
    for year in years:
        for floor in floors:
            path = data_dir / f"{year}Floor{floor}.csv"
            if not path.exists():
                continue
            records.extend(read_floor_file(path, floor, feature_mode))
    if not records:
        raise RuntimeError(f"No usable records found in {data_dir} for floors={floors}, years={years}, mode={feature_mode}")
    return records


def split_boundaries(year: int) -> Tuple[np.datetime64, np.datetime64]:
    train_end = np.datetime64(f"{year}-08-31T23:59:00")
    val_end = np.datetime64(f"{year}-10-31T23:59:00")
    return train_end, val_end


class WindowDataset(Dataset):
    def __init__(
        self,
        records: List[SeriesRecord],
        input_len: int,
        horizons: Sequence[int],
        stride: int,
        split: str,
        train_end: np.datetime64,
        val_end: np.datetime64,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        floor_to_idx: Optional[Dict[int, int]] = None,
        zone_to_idx: Optional[Dict[int, int]] = None,
    ):
        self.records = records
        self.input_len = input_len
        self.horizons = np.asarray(horizons, dtype=np.int64)
        self.stride = stride
        self.split = split
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.floor_to_idx = floor_to_idx or make_id_map([rec.floor for rec in records])
        self.zone_to_idx = zone_to_idx or make_id_map([rec.zone for rec in records])
        self.index: List[Tuple[int, int]] = []

        max_h = int(max(horizons))
        for rid, rec in enumerate(records):
            n = len(rec.values)
            last_start = n - input_len - max_h
            if last_start <= 0:
                continue
            for start in range(0, last_start, stride):
                input_end_time = rec.dates[start + input_len - 1]
                target_end_time = rec.dates[start + input_len + max_h - 1]
                if split == "train":
                    ok = target_end_time <= train_end
                elif split == "val":
                    ok = input_end_time > train_end and target_end_time <= val_end
                elif split == "test":
                    ok = input_end_time > val_end
                else:
                    raise ValueError(split)
                if ok:
                    self.index.append((rid, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start = self.index[idx]
        rec = self.records[rid]
        x = rec.values[start : start + self.input_len].copy()
        target_idx = start + self.input_len + self.horizons - 1
        y = rec.values[target_idx, 0].copy()
        last_idx = start + self.input_len - 1

        base_candidates = []
        for offset in [0, 1440, 10080]:
            if offset == 0:
                idxs = np.full_like(target_idx, last_idx)
            else:
                idxs = target_idx - offset
                idxs = np.where(idxs >= 0, idxs, last_idx)
            base_candidates.append(rec.values[idxs, 0])
        seasonal_bases = np.stack(base_candidates, axis=-1).astype(np.float32)

        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
            y = (y - self.feature_mean[0]) / self.feature_std[0]
            seasonal_bases = (seasonal_bases - self.feature_mean[0]) / self.feature_std[0]

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
            torch.tensor(self.floor_to_idx[rec.floor], dtype=torch.long),
            torch.tensor(self.zone_to_idx[rec.zone], dtype=torch.long),
            torch.from_numpy(seasonal_bases.astype(np.float32)),
        )


def make_id_map(ids: Sequence[int]) -> Dict[int, int]:
    return {value: idx for idx, value in enumerate(sorted(set(ids)))}


def compute_scaler(
    records: List[SeriesRecord],
    input_len: int,
    horizons: Sequence[int],
    stride: int,
    train_end: np.datetime64,
) -> Tuple[np.ndarray, np.ndarray]:
    sums = None
    sqs = None
    count = 0
    max_h = int(max(horizons))
    for rec in records:
        n = len(rec.values)
        last_start = n - input_len - max_h
        for start in range(0, max(0, last_start), stride):
            target_end_time = rec.dates[start + input_len + max_h - 1]
            if target_end_time <= train_end:
                x = rec.values[start : start + input_len]
                if sums is None:
                    sums = x.sum(axis=0, dtype=np.float64)
                    sqs = (x.astype(np.float64) ** 2).sum(axis=0)
                else:
                    sums += x.sum(axis=0, dtype=np.float64)
                    sqs += (x.astype(np.float64) ** 2).sum(axis=0)
                count += len(x)
    if sums is None or count == 0:
        raise RuntimeError("Training split has no windows.")
    mean = sums / count
    var = np.maximum(sqs / count - mean**2, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def make_bundle(
    data_dir: Path,
    years: Sequence[int],
    floors: Sequence[int],
    feature_mode: str,
    input_len: int,
    horizons: Sequence[int],
    stride: int,
) -> DataBundle:
    records = load_records(data_dir, years, floors, feature_mode)
    train_end, val_end = split_boundaries(years[-1])
    mean, std = compute_scaler(records, input_len, horizons, stride, train_end)
    floor_to_idx = make_id_map([rec.floor for rec in records])
    zone_to_idx = make_id_map([rec.zone for rec in records])
    train = WindowDataset(records, input_len, horizons, stride, "train", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    val = WindowDataset(records, input_len, horizons, stride, "val", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    test = WindowDataset(records, input_len, horizons, stride, "test", train_end, val_end, mean, std, floor_to_idx, zone_to_idx)
    feature_names = records[0].feature_names
    return DataBundle(
        train,
        val,
        test,
        feature_names,
        mean,
        std,
        float(mean[0]),
        float(std[0]),
        len(floor_to_idx),
        len(zone_to_idx),
        floor_to_idx,
        zone_to_idx,
    )
